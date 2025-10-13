"""An example workflow using AgentScope's ReAct agent to solve tasks.

This workflow is a demonstration of how to integrate the AgentScope framework within the Trinity-RFT workflow system with minimal modifications.
"""

from typing import Dict, List, Optional, Union

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("as_react_workflow")
class AgentScopeReActWorkflow(Workflow):
    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.model_client = model.get_openai_async_client()
        self.reset(task)

    def reset(self, task: Task):
        from trinity.common.workflows.agentscope.react.templates import TEMPLATE_MAP

        task_type = task.workflow_args.get("type", "gsm8k")
        self.logger.info(f"task_type: {task_type}")
        template = TEMPLATE_MAP.get(task_type, None)
        if template is None:
            raise ValueError(
                f"Unsupported task type {task_type} for AgentScope ReAct Agent, please add a template first."
            )
        # extract the query and the answer from the task
        self.query = task.raw_task.get(task.format_args.prompt_key)  # type: ignore [index]
        self.answer = task.raw_task.get(task.format_args.response_key)  # type: ignore [index]
        self.reward_fn = template.reward_fn_cls(**task.reward_fn_args)
        self.toolkit_manager = template.toolkit_manager(task=task)

        system_prompt = (
            template.system_prompt
            if isinstance(template.system_prompt, str)
            else template.system_prompt(task)
        )

        # import here to avoid the import error if agentscope is not installed and this workflow is not used
        try:
            from trinity.common.workflows.agentscope.react.react_agent import (
                AgentScopeReActAgent,
            )
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)
        self.agent = AgentScopeReActAgent(
            model_name=self.model_client.model_path,
            openai_client=self.model_client,
            system_prompt=system_prompt,
            generate_kwargs={
                "temperature": self.rollout_args.get("temperature", 1.0),
                "max_tokens": self.rollout_args.get("max_tokens", 4096),
            },
            response_structure=template.response_structure,
            toolkit=self.toolkit_manager.toolkit,
        )

    async def run_async(self):
        """Run the workflow asynchronously."""
        # Step 1: call the react agent to solve the task
        response = await self.agent.reply(self.query)
        # Step 2: extract the experience
        exps = self.model.extract_experience_from_history()
        # Step 3: calculate the reward based on the response
        reward = await self.calculate_reward(response, exps)
        # Step 4: construct experiences from the interaction history and return them
        return self.construct_experiences(reward, exps)

    async def calculate_reward(self, response, exps) -> Union[float, Dict[str, float]]:
        """Calculate the reward for the workflow.

        Returns:
            Union[float, Dict[str, float]]: The reward value or a dictionary of reward value.
        """
        return self.reward_fn(
            response=response,
            truth=self.answer,
            auxiliary_models=self.auxiliary_models,
            num_turns=len(exps),
            **self.toolkit_manager.get_status(),
        )

    def construct_experiences(
        self, reward: Union[float, Dict[str, float]], exps
    ) -> List[Experience]:
        """Construct experiences from the agent's interaction history.

        Args:
            reward (Union[float, Dict[str, float]]): The reward value to assign to each experience.

        Returns:
            List: A list of Experience objects.
        """
        reward_value = reward if isinstance(reward, float) else sum(reward.values())
        react_memory_length = len(self.agent.agent.memory.content)
        num_turns = len(exps)
        for i, exp in enumerate(exps):
            exp.eid.step = i
            exp.reward = reward_value
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["react_memory_length"] = react_memory_length
            exp.metrics["num_turns"] = num_turns
            # record detailed reward if available
            if isinstance(reward, dict):
                exp.metrics.update(reward)
        return exps

    @property
    def asynchronous(self):
        """AgentScope's ReAct agent only supports asynchronous calls, so we set this to True."""
        return True

    @property
    def repeatable(self):
        """This workflow is not repeatable."""
        return False

    @property
    def resettable(self):
        return True
