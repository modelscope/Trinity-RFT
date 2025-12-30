from typing import Awaitable, Callable, Dict, List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow


class AgentScopeWorkflowAdapter(Workflow):
    """Adapter to wrap a agentscope trainable workflow function into a Trinity Workflow."""

    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        """Initialize the adapter with the task and model."""
        try:
            from agentscope.model import TrinityChatModel
        except ImportError:
            raise ImportError(
                "This workflow requires agentscope >= 1.0.7, please install "
                "it via `pip install agentscope>=1.0.7`",
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
                "top_p": self.task.rollout_args.top_p,
                "max_tokens": self.task.rollout_args.max_tokens or 4096,
                "logprobs": True,
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


class AgentScopeWorkflowAdapterV1(Workflow):
    """A more general adapter to wrap agentscope trainable workflow and judge functions into a Trinity Workflow."""

    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        """Initialize the adapter with the task and model."""
        try:
            from agentscope.model import TrinityChatModel
        except ImportError:
            raise ImportError(
                "This workflow requires agentscope >= 1.0.11, please install "
                "it via `pip install agentscope>=1.0.11`",
            )

        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.workflow_func = task.workflow_args.get("workflow_func", None)
        self.judge_func = task.workflow_args.get("judge_func", None)

        if self.workflow_func is None:
            raise ValueError(
                "The 'workflow_func' is not provided.",
            )

        self.chat_model: TrinityChatModel = TrinityChatModel(
            model.get_openai_async_client(),
            generate_kwargs={
                "temperature": self.task.rollout_args.temperature,
                "top_p": self.task.rollout_args.top_p,
                "max_tokens": self.task.rollout_args.max_tokens or 4096,
                "logprobs": True,
                "top_logprobs": self.task.rollout_args.logprobs,
            },
        )

        # TODO: customize generate_kwargs for auxiliary models if needed
        if self.auxiliary_model_wrappers is not None and self.auxiliary_models is not None:
            self.auxiliary_chat_models = {
                aux_model_wrapper.model_name
                or f"auxiliary_model_{i}": TrinityChatModel(openai_async_client=aux_model)
                for i, (aux_model_wrapper, aux_model) in enumerate(
                    zip(self.auxiliary_model_wrappers, self.auxiliary_models)
                )
            }
        else:
            self.auxiliary_chat_models = {}

    def construct_experiences(
        self,
        reward: float,
        metrics: Dict,
    ) -> List[Experience]:
        """Construct experiences from the agent's interaction history.

        Args:
            reward (float): The reward value to assign to each experience.
            metrics (Dict): A dictionary of metrics to be attached to the last experience.

        Returns:
            List: A list of Experience objects.
        """
        exps = self.model.extract_experience_from_history()
        for exp in exps:
            exp.reward = reward
        # only attach metrics to the last experience
        if len(exps) > 0:
            exps[-1].metrics = metrics
        return exps

    async def run_async(self) -> List[Experience]:
        """Run the workflow asynchronously and return experiences."""
        try:
            from agentscope.tuner import JudgeOutput, WorkflowOutput
        except ImportError:
            raise ImportError(
                "Fail to import agentscope tuner related types. Please ensure agentscope>=1.0.11 is installed."
            )

        metrics = {}
        workflow_output: WorkflowOutput = await self.workflow_func(
            self.task.raw_task, self.chat_model, self.auxiliary_chat_models
        )  # type: ignore [arg-type]
        metrics.update(workflow_output.metrics or {})
        if self.judge_func is not None:
            assert (
                workflow_output.response is not None
            ), "Workflow must provide response for judging."
            judge_output: JudgeOutput = await self.judge_func(
                self.task.raw_task, workflow_output.response, self.auxiliary_chat_models
            )  # type: ignore [arg-type]
            reward = judge_output.reward
            metrics.update(judge_output.metrics or {})
        else:
            assert (
                workflow_output.reward is not None
            ), "Either workflow or judge must provide reward."
            reward = workflow_output.reward
        return self.construct_experiences(reward, metrics)
