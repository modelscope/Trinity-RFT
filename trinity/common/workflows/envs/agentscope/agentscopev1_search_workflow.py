# -*- coding: utf-8 -*-
"""We include simple react deep search workflows in this file. We use AgentScope V1 framework."""

import os
import re
from typing import List, Optional

from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow


class AgentScopeV1ReactSearchWorkflow(Workflow):
    """
    This workflow serves as an example of how to use the agentscope framework within the trinity workflow.
    """

    can_reset: bool = True
    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):  # get openai client from model
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
        self.workflow_args = task.workflow_args
        self.max_turns = int(self.workflow_args.get("max_turns", 10))
        self.search_client_type = self.workflow_args.get("search_client_type", "searxng")
        self.max_model_tokens = int(self.workflow_args.get("max_model_tokens", 24000))
        if self.search_client_type not in ["searxng", "tavily"]:
            raise ValueError(
                f"search_client_type must be one of ['searxng', 'tavily'], but got {self.search_client_type}"
            )
        self.system_prompt = "You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions."

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

    def judge_result(self, result, question, correct_answer, judge_model=None) -> bool:
        """Use LLM to judge whether the answer is correct or not."""
        if result is None:
            return False

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

        final_answer = extract_final_answer(result)

        judge_prompt = f"""Judge whether the following [response] to [question] is correct or not based on the [correct_answer] below.

[question]: {question}

[response]: {final_answer}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

1. **Reasoning**: Explain why the [response] is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the response. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

2. **Correctness**: Answer exactly "YES" if [response] matches the [correct_answer] given above, or is within a small margin of error for numerical problems and small format issue for text problems (for example, with or without a hyphen should be considered the same). Answer exactly "NO" otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the [response] is incorrect.
"""
        messages = [
            {"role": "system", "content": "You evaluate correctness."},
            {"role": "user", "content": judge_prompt},
        ]
        completion = judge_model.chat.completions.create(
            model=judge_model.model_path, messages=messages, stream=False
        )
        judge_output = completion.choices[0].message.content

        self.logger.info(
            f"[judge_result] prompt:\n{judge_prompt}\n\n[judge_result] LLM output:\n{judge_output}"
        )

        # Yes if the response is correct, No otherwise
        match = re.search(r"Correctness.*?YES", judge_output, re.IGNORECASE)
        return match is not None

    async def run_async(self):
        try:
            from agentscope.agent import ReActAgent
            from agentscope.mcp import StdIOStatefulClient
            from agentscope.memory import InMemoryMemory
            from agentscope.message import Msg
            from pydantic import BaseModel, Field
        except ImportError as e:
            error_message = f"AgentScope V1 is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        self.agent = ReActAgent(
            name="Friday",
            sys_prompt=self.system_prompt,
            model=self.agent_model,
            formatter=self.agent_model_formatter,
            memory=InMemoryMemory(),
            max_iters=self.max_turns,
        )
        self.agent.model.client = self.openai_async_client

        if self.search_client_type == "tavily":
            tavily_api_key = os.getenv("TAVILY_API_KEY", "")
            if not tavily_api_key:
                raise ValueError(
                    "TAVILY_API_KEY environment variable is not set. Please set it to use the Tavily search tool."
                )

            self.search_client = StdIOStatefulClient(
                name="tavily_mcp",
                command="npx",
                args=["-y", "tavily-mcp@latest"],
                env={"TAVILY_API_KEY": tavily_api_key},
            )
        elif self.search_client_type == "searxng":
            searxng_url = os.getenv("SEARXNG_URL", "")
            if not searxng_url:
                raise ValueError(
                    "SEARXNG_URL environment variable is not set. Please set it to use the SearXNG search tool."
                )
            self.search_client = StdIOStatefulClient(  # refer to https://github.com/ihor-sokoliuk/mcp-searxng for more details
                name="searxng_mcp",
                command="npx",
                args=["-y", "mcp-searxng"],
                env={"SEARXNG_URL": searxng_url},
            )
        else:
            raise ValueError(
                f"search_client_type must be one of ['searxng', 'tavily'], but got {self.search_client_type}"
            )

        instruction = Msg("user", content=self.task_desc, role="user")

        class FinalResult(BaseModel):
            result: str = Field(description="The final result to the initial user query")

        try:
            await self.search_client.connect()
            await self.agent.toolkit.register_mcp_client(self.search_client)
            result = await self.agent.reply(instruction, structured_model=FinalResult)
        except Exception as e:
            self.logger.error(f"Error during agent reply: {e}")
            result = None
        finally:
            if self.search_client and self.search_client.is_connected:
                await self.search_client.close()

        # Reward calculation (judge_result can stay sync if your judge_model only has sync chat, otherwise you need to make it async)
        try:
            judge_model = self.auxiliary_models[0] if self.auxiliary_models else None
            assert judge_model is not None, "Please provide a judge model for reward calculation."
            reward = 1 if self.judge_result(result, self.task_desc, self.truth, judge_model) else 0
        except Exception as e:
            self.logger.error(f"Error in judge_model judging: {e}")
            reward = 0

        self.logger.debug(f"Reward: {reward}")
        experiences = self.model.extract_experience_from_history(clear_history=True)
        return_experiences = []
        self.logger.debug(f"Experiences extracted len: {len(experiences)}")
        for i, experience in enumerate(experiences):
            experience.eid.step = i
            experience.reward = reward
            agent_metrics = {
                "react_turns": len(self.agent.memory.content) // 2,
                "max_turns": self.max_turns,
            }
            if experience.metrics is None:
                experience.metrics = {}
            experience.metrics.update(agent_metrics)
            if len(experience.tokens) > self.max_model_tokens:
                continue
            return_experiences.append(experience)
        if return_experiences:
            self.logger.debug(
                f"return experience len: {len(return_experiences)}, run_id: {str(return_experiences[-1].eid.run)}, final step reward: {return_experiences[-1].reward}"
            )
        else:
            self.logger.info("No valid experiences to return (all filtered out).")
        return return_experiences
