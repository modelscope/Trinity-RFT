# -*- coding: utf-8 -*-
"""We include the customized math workflows in this file."""

from typing import List, Optional

from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.math_reward import MathBoxedRewardFn
from trinity.common.workflows.workflow import Task, Workflow


class AgentScopeReactMathWorkflow(Workflow):
    """
    This workflow serves as an example of how to use the agentscope framework within the trinity workflow.
    We use the AgentScope V1 version here.
    """

    can_reset: bool = True
    is_async: bool = True

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
            from agentscope.formatter import OpenAIChatFormatter
            from agentscope.model import OpenAIChatModel
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        # get openai client from model
        self.openai_async_client = model.get_openai_async_client()
        self.model_name = self.openai_async_client.model_path

        self.agent_model = OpenAIChatModel(
            api_key="EMPTY",
            model_name=self.model_name,
            stream=False,
            generate_kwargs={
                "temperature": self.task.rollout_args.temperature,
                "max_tokens": self.task.rollout_args.max_tokens or 4096,
            },
        )
        self.agent_model.client = self.openai_async_client
        self.agent_model_formatter = OpenAIChatFormatter()
        self.reset(task)

    def reset(self, task: Task):
        self.system_prompt = """
You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}.
"""
        try:
            from agentscope.agent import ReActAgent
            from agentscope.memory import InMemoryMemory
            from agentscope.tool import Toolkit, execute_python_code
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)
        self.toolkit = Toolkit()
        self.toolkit.register_tool_function(execute_python_code)
        self.agent = ReActAgent(
            name="math_react_agent",
            sys_prompt=self.system_prompt,
            model=self.agent_model,
            formatter=self.agent_model_formatter,
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
        )
        # we set the openai client to the agent's model
        self.agent.model.client = self.openai_async_client

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

    async def run_async(self):
        # make sure that we have the correct import
        try:
            from agentscope.message import Msg
            from pydantic import BaseModel, Field
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        # provide the task to the react agent
        msg = Msg("user", self.task_desc, role="user")

        # Note that the main workflow can have arbitrary steps and include different logic
        class FinalResult(BaseModel):
            result: str = Field(
                description="Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}"
            )

        def extract_final_answer(result) -> str:
            """Extract the final answer from the agent's response."""
            try:
                if (
                    hasattr(result, "metadata")
                    and isinstance(result.metadata, dict)
                    and "result" in result.metadata
                ):
                    return result.metadata["result"]
                if hasattr(result, "content"):
                    if isinstance(result.content, dict) and "result" in result.content:
                        return result.content["result"]
                    return str(result.content)
                return str(result)
            except Exception as e:
                self.logger.warning(f"Extract final answer error: {e}. Raw: {result}")
                return str(result)

        result = await self.agent.reply(msg, structured_model=FinalResult)

        final_answer = extract_final_answer(result)

        reward = self.reward_fn(final_answer, self.answer)
        reward = sum(reward.values())
        self.logger.debug(f"Reward: {reward}")
        experiences = self.model.extract_experience_from_history(clear_history=True)
        self.logger.debug(f"Experiences extracted len: {len(experiences)}")
        for i, experience in enumerate(experiences):
            experience.eid.step = i
            experience.reward = reward
            agent_metrics = {"react_memory_length": len(self.agent.memory.content)}
            if experience.metrics is None:
                experience.metrics = {}
            experience.metrics.update(agent_metrics)
        self.logger.debug(
            f"return experience len: {len(experiences)}, run_id: {str(experiences[-1].eid.run)}, final step reward: {experiences[-1].reward}"
        )
        return experiences
