# -*- coding: utf-8 -*-
"""We include the agentscope react workflows in this file."""

from typing import List, Optional

from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.math_reward import MathBoxedRewardFn
from trinity.common.workflows.workflow import Task, Workflow
from trinity.utils.annotations import Deprecated


@Deprecated
class AgentScopeV0ReactMathWorkflow(Workflow):
    """
    This workflow serves as an example of how to use the agentscope framework within the trinity workflow.
    We use the AgentScope V0 version here. The code will be deprecated soon.
    """

    can_reset: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        # make sure that we have the correct import
        try:
            import agentscope
            from agentscope.service import ServiceToolkit, execute_python_code
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        # get openai client from model
        self.openai_client = model.get_openai_client()
        self.model_name = self.openai_client.model_path

        agentscope.init(
            model_configs=[
                {
                    "model_type": "openai_chat",
                    "config_name": "my_model",
                    "model_name": self.model_name,
                    "api_key": "EMPTY",
                    "generate_args": {
                        "temperature": self.task.rollout_args.temperature,
                        "max_tokens": self.task.rollout_args.max_tokens or 4096,
                    },
                    "use_openai_formatter": True,
                }
            ],
            disable_saving=True,
        )
        self.toolkit = ServiceToolkit()
        self.toolkit.add(
            execute_python_code,
            timeout=300,
            use_docker=False,
            maximum_memory_bytes=None,
        )
        self.reset(task)

    def reset(self, task: Task):
        self.system_prompt = """
You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}.
"""
        try:
            from agentscope.agents import ReActAgentV2
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)
        self.agent = ReActAgentV2(
            name="math_react_agent",
            sys_prompt=self.system_prompt,
            model_config_name="my_model",  # replace by your model config name
            service_toolkit=self.toolkit,
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
            if isinstance(self.truth, str) and "####" in self.truth:
                # GSM8K dataset
                self.answer = self.truth.split("####")[1].strip()
            else:
                self.answer = str(self.truth)
        except Exception as e:
            self.logger.debug(f"Error in getting answer from truth: {str(e)}")
            self.answer = str(self.truth)

        # we use the boxed format to evaluate the answer
        self.reward_fn = MathBoxedRewardFn()

    def run(self):
        # make sure that we have the correct import
        try:
            from agentscope.message import Msg
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        # provide the task to the react agent
        msg = Msg("user", self.task_desc, role="user")
        # Note that the main workflow can have arbitrary steps and include different logic
        content = self.agent.reply(msg).content

        # unify the response format to text
        try:
            if isinstance(content, list):
                response_text = content[0]["text"]
            else:
                response_text = content
        except Exception as e:
            error_message = f"Error in processing the response: {e}"
            self.logger.info(error_message)
            response_text = str(content)

        reward = self.reward_fn(response_text, self.answer)
        reward = sum(reward.values())
        self.logger.debug(f"Reward: {reward}")
        experiences = self.model.extract_experience_from_history(clear_history=True)
        self.logger.debug(f"Experiences extracted len: {len(experiences)}")
        for i, experience in enumerate(experiences):
            experience.eid.step = i
            experience.reward = reward
            agent_metrics = {"react_memory_length": len(self.agent.memory.get_memory())}
            if experience.metrics is None:
                experience.metrics = {}
            experience.metrics.update(agent_metrics)
        self.logger.debug(
            f"return experience len: {len(experiences)}, run_id: {str(experiences[-1].eid.run)}, final step reward: {experiences[-1].reward}"
        )
        return experiences
