from typing import Awaitable, Callable, Dict, List, Optional

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("agentscope_workflow_adapter")
class AgentScopeWorkflowAdapter(Workflow):
    """Adapter to wrap a agentscope trainable workflow function into a Trinity Workflow."""

    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        """Initialize the adapter with the task and model."""
        try:
            from agentscope.model import TrinityChatModel
        except ImportError:
            raise ImportError(
                "This workflow requires agentscope >= 0.1.6, please install "
                "it via `pip install agentscope>=0.1.6`",
            )

        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.workflow_func: Callable[
            [Dict, TrinityChatModel], Awaitable[float]
        ] = task.workflow_args.get("workflow_func", None)

        if self.workflow_func is None:
            raise ValueError(
                "The 'workflow_func' is not provided.",
            )

        self.chat_model: TrinityChatModel = TrinityChatModel(
            model.get_openai_async_client(),
            generate_kwargs={
                "temperature": self.task.rollout_args.temperature,
                "max_tokens": self.task.rollout_args.max_tokens or 4096,
                "top_logprobs": self.task.rollout_args.logprobs,
            },
        )

    def construct_experiences(
        self,
        reward: float,
    ) -> List[Experience]:
        """Construct experiences from the agent's interaction history.

        Args:
            reward (float): The reward value to assign to each experience.

        Returns:
            List: A list of Experience objects.
        """
        exps = self.model.extract_experience_from_history()
        for exp in exps:
            exp.reward = reward
        return exps

    async def run_async(self) -> List[Experience]:
        """Run the workflow asynchronously and return experiences."""
        reward = await self.workflow_func(self.task.raw_task, self.chat_model)  # type: ignore [arg-type]
        return self.construct_experiences(reward)
