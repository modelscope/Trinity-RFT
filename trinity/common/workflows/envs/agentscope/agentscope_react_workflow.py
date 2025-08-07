# -*- coding: utf-8 -*-
"""We include the customized math workflows in this file."""

from typing import List, Optional

import openai

from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.math_reward import MathBoxedRewardFn
from trinity.common.workflows import Task
from trinity.common.workflows.workflow import WORKFLOWS, Workflow
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@WORKFLOWS.register_module("agentscope_reactv2_gsm8k_workflow")
class AgentScopeReactV2Gsm8kWorkflow(Workflow):
    """
    This workflow serve as an example of how to use the agentscope framework with in the trinity workflow.
    """

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        # make sure that we have the corrct import
        try:
            import agentscope
            from agentscope.agents import ReActAgentV2
            from agentscope.service import ServiceToolkit, execute_python_code
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            logger.error(error_message)
            raise ImportError(error_message)

        # get openai client from model
        self.openai_client = model.get_openai_client()
        self.model_name = self.openai_client.model_path
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

        temperature = self.rollout_args.get("temperature", 1.0)
        max_tokens = self.rollout_args.get("max_tokens", 4096)

        agentscope.init(
            model_configs=[
                {
                    "model_type": "openai_chat",
                    "config_name": "my_model",
                    "model_name": self.model_name,
                    "api_key": "EMPTY",
                    "generate_args": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    "use_openai_formatter": True,
                }
            ],
            disable_saving=True,
        )
        toolkit = ServiceToolkit()
        toolkit.add(
            execute_python_code,
            timeout=300,
            use_docker=False,
            maximum_memory_bytes=None,
        )
        system_prompt = """
You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You should return your final answer within \\boxed{{}}.
"""
        self.agent = ReActAgentV2(
            name="math_react_agent",
            sys_prompt=system_prompt,
            model_config_name="my_model",  # replace by your model config name
            service_toolkit=toolkit,
            max_iters=5,
            verbose=False,
        )
        # we set the openai client to the agent's model
        self.agent.model.client = self.openai_client

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        # we get the answer from gsm8k dataset
        try:
            self.answer = self.truth.split("####")[1].strip()
        except Exception as e:
            logger.error(f"Error in getting answer from truth: {e}")
            self.answer = self.truth

        # we use the boxed format to evaluate the answer
        self.reward_fn = MathBoxedRewardFn()

    @property
    def resettable(self):
        return False

    @property
    def repeatable(self):
        return False

    def run(self):
        # make sure that we have the corrct import
        try:
            from agentscope.message import Msg
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            logger.error(error_message)
            raise ImportError(error_message)

        # provide the task to the react agent
        msg = Msg("user", self.task_desc, role="user")
        # Note that the main workflow can have arbitrary steps and include different logic
        content = self.agent.reply(msg).content

        # unify the response format to text
        if isinstance(content, list):
            response_text = content[0]["text"]
        else:
            response_text = content

        reward = self.reward_fn(response_text, self.answer)
        reward = sum(reward.values())
        logger.debug(f"Reward: {reward}")
        experiences = self.model.extract_experience_from_history(clear_history=True)
        logger.debug(f"Experiences extracted len: {len(experiences)}")
        import uuid

        run_id = uuid.uuid4().hex[:6]
        for i, experience in enumerate(experiences):
            experience.eid.step = i
            experience.eid.run = run_id
            experience.reward = reward
        logger.debug(
            f"return experience len: {len(experiences)}, run_id: {str(experiences[-1].eid.run)}, final step reward: {experiences[-1].reward}"
        )
        return experiences
