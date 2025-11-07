"""An example workflow using AgentScope's ReAct agent to solve tasks.

This workflow is a demonstration of how to integrate the AgentScope framework within the Trinity-RFT workflow system with minimal modifications.
"""
import uuid
import openai
from typing import Dict, List, Optional, Union
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow
from transformers import AutoTokenizer
from .templates import TEMPLATE_MAP

@WORKFLOWS.register_module("as_react_workflow")
class AgentScopeReActWorkflow(Workflow):
    is_async: bool = True

    def __init__(
        self,
        config,
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


@WORKFLOWS.register_module("agentopia_workflow")
class AgentopiatWorkflowWrap(Workflow):
    is_async: bool = True
    def __init__(
        self,
        config,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.config = config
        self.task = task

        # 模拟openai的异步客户端
        self.model_client = model.get_openai_async_client()
        # task_type 用于获取奖励函数
        # extract the query and the answer from the task
        self.query = task.raw_task.get(task.format_args.prompt_key)  # type: ignore [index]
        self.answer = task.raw_task.get(task.format_args.response_key)  # type: ignore [index]
        self.task.workflow_args = {
            "env_type": "appworld",
            "task_id": self.task.task_id,
            "instance_id": uuid.uuid4().hex,
        }

    async def run_async(self):
        cmt_tokenized = {}
        from agentopia.trinity_compat_env import TrinityCompatWorkflow
        from agentopia.schema.trajectory import Sample
        from omegaconf import OmegaConf

        #   config_path: '/mnt/data/qingxu.fu/ba-verl-advance/launcher/appworld_linear_base/git-appworld-qwen2-agentscope-bz32-tp4-linear.yaml'
        config = OmegaConf.load(self.config.agentopia_configuration.config_path)
        config.trainer.experiment_name = "dummy"
        cmt = TrinityCompatWorkflow(
            task=self.task,
            llm_handle=self.model_client,
            tokenizer=AutoTokenizer.from_pretrained(self.model_client.model_path),
            config=config,
        ).run_in_new_thread()

        from vsdb import bp
        bp("DEV3")

        sample_final = []
        try:
            sample_arr = cmt.group_tokenize()
        except Exception as e:
            cmt.generate_log(global_step=-1)
            raise e
        cmt.generate_log(global_step=-1)
        sample_final += sample_arr


        exps = []
        for index, sample in enumerate(sample_final):
            sample: Sample
            input_ids = sample.input_ids
            prompt_ids = sample.prompt_ids
            response_ids = sample.response_ids
            attention_mask = sample.attention_mask
            prompt_attention_mask = sample.prompt_attention_mask
            response_attention_mask = sample.response_attention_mask
            loss_mask = sample.loss_mask
            prompt_loss_mask = sample.prompt_loss_mask
            response_loss_mask = sample.response_loss_mask
            position_ids = sample.position_ids
            prompt_position_ids = sample.prompt_position_ids
            response_position_ids = sample.response_position_ids
            # cmt_tokenized["step_reward"] = self.reward_structure.step_reward[index]

            logprobs = sample.response_logprobs
            try:
                reward = cmt.reward_structure.step_reward
                if isinstance(reward, list):
                    reward = reward[0]
            except Exception as e:
                reward = cmt.reward_structure.raw_reward
            if not isinstance(reward, (float, int)): # if reward is still not a float or int, set it to 0.0
                reward = cmt.reward_structure.raw_reward

            if len(response_ids) + len(prompt_ids) == len(input_ids) and len(logprobs) == len(response_ids) and len(logprobs) > 0:
                exp = Experience(
                    # eid=uuid.uuid4().hex,
                    tokens = input_ids,     # [seq_length] prompt + response
                    prompt_length = len(prompt_ids),  # Length of the prompt in tokens, used for generating attention masks
                    logprobs = logprobs,   # [resp_length]
                    reward = reward,  #
                    # advantages=None,
                    # returns=None,
                    info = {},
                    metrics = {},   # for wandb logging (must be string:float)
                    response_text = "", # optional
                    prompt_text = "", # optional
                    #### for multi-turn experiences
                    action_mask = response_loss_mask,  # 1 是训练
                    messages=sample.messages,    #
                    # tools,
                    #### for dpo experiences
                    # chosen,
                    # rejected,
                    # chosen_messages,
                    # rejected_messages,
                    #### for multi-modal data
                    # multi_modal_inputs
                )
                exps += [exp]
            else:
                from vsdb import bp
                bp("BUGX")
        return exps
