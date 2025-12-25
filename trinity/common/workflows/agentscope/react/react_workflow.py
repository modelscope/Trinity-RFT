"""An example workflow using AgentScope's ReAct agent to solve tasks.

This workflow is a demonstration of how to integrate the AgentScope framework within the Trinity-RFT workflow system with minimal modifications.
"""

from typing import Dict, List, Optional, Union

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow

from .templates import TEMPLATE_MAP


class AgentScopeReActWorkflow(Workflow):
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
        self.model_client = model.get_openai_async_client()

        task_type = task.workflow_args.get("type", "gsm8k")
        template = TEMPLATE_MAP.get(task_type, None)
        if template is None:
            raise ValueError(
                f"Unsupported task type {task_type} for AgentScope ReAct Agent, please add a template first."
            )
        # extract the query and the answer from the task
        self.query = task.raw_task.get(task.format_args.prompt_key)  # type: ignore [index]
        self.answer = task.raw_task.get(task.format_args.response_key)  # type: ignore [index]
        self.reward_fn = template.reward_fn_cls(**task.reward_fn_args)

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
            system_prompt=template.system_prompt,
            generate_kwargs={
                "temperature": self.task.rollout_args.temperature,
                "max_tokens": self.task.rollout_args.max_tokens or 4096,
            },
            response_structure=template.response_structure,
        )

    async def run_async(self):
        """Run the workflow asynchronously."""
        # Step 1: call the react agent to solve the task
        response = await self.agent.reply(self.query)
        # Step 2: calculate the reward based on the response
        reward = await self.calculate_reward(response)
        # Step 3: construct experiences from the interaction history and return them
        return self.construct_experiences(reward)

    async def calculate_reward(self, response) -> Union[float, Dict[str, float]]:
        """Calculate the reward for the workflow.

        Returns:
            Union[float, Dict[str, float]]: The reward value or a dictionary of reward value.
        """
        return self.reward_fn(response=response, truth=self.answer)

    def construct_experiences(self, reward: Union[float, Dict[str, float]]) -> List[Experience]:
        """Construct experiences from the agent's interaction history.

        Args:
            reward (Union[float, Dict[str, float]]): The reward value to assign to each experience.

        Returns:
            List: A list of Experience objects.
        """
        exps = self.model.extract_experience_from_history()
        for exp in exps:
            exp.reward = reward if isinstance(reward, float) else sum(reward.values())
            exp.metrics = {"react_memory_length": len(self.agent.agent.memory.content)}
            # record detailed reward if available
            if isinstance(reward, dict):
                exp.metrics.update(reward)
        return exps
