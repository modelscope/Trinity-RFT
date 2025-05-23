# -*- coding: utf-8 -*-
"""veRL Trainer Class

Modified from verl/trainer/ppo/ray_trainer.py
"""
import os
from typing import Tuple

import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from trinity.common.config import Config
from trinity.common.constants import AlgorithmType
from trinity.common.experience import Experiences
from trinity.trainer.trainer import TrainEngineWrapper
from trinity.trainer.verl.ray_trainer import (
    DataProto,
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    find_latest_ckpt_path,
    np,
    pprint,
    reduce_metrics,
)
from trinity.utils.monitor import Monitor


class _InternalDataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.index = 0
        self.experience_buffer = None

    def state_dict(self):
        return None

    def load_state_dict(self, *args, **kwargs):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        raise StopIteration


class VerlPPOTrainerWrapper(RayPPOTrainer, TrainEngineWrapper):
    """A wrapper for verl.trainer.ppo.RayPPOTrainer."""

    def __init__(
        self,
        global_config: Config,
    ):
        train_config = global_config.trainer
        config = OmegaConf.structured(train_config.trainer_config)
        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # instantiate tokenizer

        tokenizer = hf_tokenizer(local_path)

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from trinity.trainer.verl.fsdp_workers import (
                ActorRolloutRefWorker,
                CriticWorker,
            )

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            raise NotImplementedError("Not support megatron for now.")

        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
        )
        self.init_workers()
        self.algorithm_type = (
            AlgorithmType.PPO
        )  # TODO: initialize algorithm_type according to config
        self.logger = Monitor(
            project=config.trainer.project_name,
            name=config.trainer.experiment_name,
            role="trainer",
            config=global_config,
        )
        self.reset_experiences_example_table()

    def reset_experiences_example_table(self):
        self.experiences_example_table = pd.DataFrame(
            columns=["step", "reward", "prompt", "response"]
        )

    def prepare(self):
        self.actor_rollout_wg.setup_weight_sync_group()

        self.global_steps = 0
        self.sft_warmup_step_num = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.sft_warmup_step_num = min(self.global_steps, self.config.trainer.sft_warmup_steps)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

    def _create_dataloader(self):
        self.train_dataloader = _InternalDataLoader(self.config)
        # TODO: compute total training steps
        # if self.algorithm_type.is_dpo():
        #     train_batch_size = self.config.buffer.read_batch_size
        #     total_epochs = self.config.trainer.total_epochs
        #     from math import ceil

        #     self.total_training_steps = ceil(
        #         self.train_dataloader.size() // train_batch_size * total_epochs
        #     )
        #     if not self.config.actor_rollout_ref.actor.optim.total_training_steps > 0:
        #         self.config.actor_rollout_ref.actor.optim.total_training_steps = (
        #             self.total_training_steps
        #         )
        #     if not self.config.critic.optim.total_training_steps > 0:
        #         self.config.critic.optim.total_training_steps = self.total_training_steps
        # else:
        self.total_training_steps = float("inf")

    def train_dpo_step(self, experiences: Experiences) -> Tuple[bool, int]:
        metrics = {}
        timing_raw = {}

        with _timer("step", timing_raw):
            # generate a batch
            attention_mask = experiences.attention_masks
            cumsum = torch.cumsum(attention_mask, dim=-1)
            position_ids = torch.clip(cumsum - 1, 0, None).long()

            batch = DataProto.from_single_dict(
                {
                    "uid": np.array(experiences.run_ids),  # useless
                    "position_ids": position_ids,
                    "input_ids": experiences.tokens.long(),
                    "responses": experiences.tokens[:, experiences.prompt_length :].long(),
                    "attention_mask": attention_mask.long(),
                    "response_mask": (
                        experiences.action_masks[:, experiences.prompt_length :].long()
                        if hasattr(experiences, "action_masks")
                        and experiences.action_masks is not None
                        else attention_mask[:, experiences.prompt_length :].long()
                    ),
                }
            )
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            # self._balance_batch(batch, metrics=metrics)  # _balance_batch will shuffle the batch, which will break DPO
            # TODO: implement a new _balance_batch for DPO

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1
            ).tolist()

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # update actor
            with _timer("update_actor", timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)

        # collect metrics
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        self.logger.log(data=metrics, step=self.global_steps)

        # save checkpoint
        if (
            self.config.trainer.save_freq > 0
            and self.global_steps % self.config.trainer.save_freq == 0
        ):
            with _timer("save_checkpoint", timing_raw):
                self._save_checkpoint()

        self.global_steps += 1
        return True, self.global_steps - 1

    def train_sft_step(self, experiences: Experiences) -> Tuple[bool, int]:
        if self.sft_warmup_step_num >= self.config.trainer.sft_warmup_steps:
            return False, self.global_steps - 1
        metrics = {}
        timing_raw = {}

        with _timer("step", timing_raw):
            # generate a batch
            attention_mask = experiences.attention_masks
            cumsum = torch.cumsum(attention_mask, dim=-1)
            position_ids = torch.clip(cumsum - 1, 0, None).long()

            batch = DataProto.from_single_dict(
                {
                    "uid": np.array(experiences.run_ids),
                    "position_ids": position_ids,
                    "input_ids": experiences.tokens.long(),
                    "responses": experiences.tokens[:, experiences.prompt_length :].long(),
                    "attention_mask": attention_mask.long(),
                    "response_mask": (
                        experiences.action_masks[:, experiences.prompt_length :].long()
                        if hasattr(experiences, "action_masks")
                        and experiences.action_masks is not None
                        else attention_mask[:, experiences.prompt_length :].long()
                    ),
                }
            )
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            self._balance_batch(batch, metrics=metrics)  # TODO this may affect multi-turn

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1
            ).tolist()

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # update actor
            with _timer("update_actor", timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)

        # collect metrics
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        # TODO: log as sft metrics
        self.logger.log(data=metrics, step=self.global_steps)
        self.sft_warmup_step_num += 1
        self.global_steps += 1
        if self.sft_warmup_step_num == self.config.trainer.sft_warmup_steps:
            self.logger.log(
                data={"sft_warmup_steps": self.sft_warmup_step_num},
                step=self.global_steps - 1,
            )
            with _timer("save_checkpoint", timing_raw):
                self._save_checkpoint()
            return False, self.global_steps - 1
        return True, self.global_steps - 1

    def train_rft_step(self, experiences: Experiences) -> Tuple[bool, int]:
        metrics = {}
        timing_raw = {}

        with _timer("step", timing_raw):
            # Convert rewards to token_level_rewards
            attention_mask = experiences.attention_masks
            token_level_rewards = torch.zeros(attention_mask.shape, dtype=experiences.rewards.dtype)
            cumsum = torch.cumsum(attention_mask, dim=-1)
            eos_mask_idx = cumsum.argmax(dim=-1)
            position_ids = torch.clip(cumsum - 1, 0, None).long()
            token_level_rewards[
                torch.arange(experiences.batch_size), eos_mask_idx
            ] = experiences.rewards
            token_level_rewards = token_level_rewards[:, experiences.prompt_length :]

            batch = DataProto.from_single_dict(
                {
                    "uid": np.array(experiences.run_ids),
                    "position_ids": position_ids,
                    "input_ids": experiences.tokens.long(),
                    "responses": experiences.tokens[:, experiences.prompt_length :].long(),
                    "attention_mask": attention_mask.long(),
                    "response_mask": (
                        experiences.action_masks[:, experiences.prompt_length :].long()
                        if hasattr(experiences, "action_masks")
                        and experiences.action_masks is not None
                        else attention_mask[:, experiences.prompt_length :].long()
                    ),
                    "token_level_scores": token_level_rewards,
                    "old_log_probs": experiences.logprobs[:, experiences.prompt_length :],  # type: ignore
                }
            )
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)  # TODO this may affect multi-turn

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1
            ).tolist()

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            with _timer("adv", timing_raw):
                # compute rewards. apply_kl_penalty if available
                if not self.config.actor_rollout_ref.actor.get("use_kl_loss", False):
                    batch, kl_metrics = apply_kl_penalty(
                        batch,
                        kl_ctrl=self.kl_ctrl,
                        kl_penalty=self.config.algorithm.kl_penalty,
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process
                kwargs = {}
                algorithm_type = self.config.actor_rollout_ref.actor.get(
                    "algorithm_type", AlgorithmType.PPO
                )
                if algorithm_type == AlgorithmType.OPMD:
                    tau = self.config.actor_rollout_ref.actor.get("tau", 0.0)
                    opmd_baseline = self.config.actor_rollout_ref.actor.get("opmd_baseline", "mean")
                    kwargs = {
                        "algorithm_type": algorithm_type,
                        "tau": tau,
                        "opmd_baseline": opmd_baseline,
                    }
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    **kwargs,
                )

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    # TODO add send weight explorer
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            if (
                self.config.trainer.save_freq > 0
                and self.global_steps % self.config.trainer.save_freq == 0
            ):
                with _timer("save_checkpoint", timing_raw):
                    self._save_checkpoint()

        # collect metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(
            compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
        )

        if self.config.enable_preview:
            self._log_experiences(experiences)

        # TODO: make a canonical logger that supports various backend
        self.logger.log(data=metrics, step=self.global_steps)

        self.global_steps += 1

        if self.global_steps >= self.total_training_steps:
            if (
                self.config.trainer.save_freq > 0
                and (self.global_steps - 1) % self.config.trainer.save_freq != 0
            ):
                with _timer("save_checkpoint", timing_raw):
                    self._save_checkpoint()
            # stop training
            return False, self.global_steps - 1
        else:
            # continue
            return True, self.global_steps - 1

    def _log_single_experience(
        self, experiences: Experiences, idx: int, skip_special_tokens: bool
    ) -> None:
        reward = experiences.rewards[idx]
        attn_mask = experiences.attention_masks[idx].bool()
        prompt_token = experiences.tokens[idx][: experiences.prompt_length][
            attn_mask[: experiences.prompt_length]
        ]
        response_token = experiences.tokens[idx][experiences.prompt_length :][
            attn_mask[experiences.prompt_length :]
        ]
        prompt_text = self.tokenizer.decode(prompt_token, skip_special_tokens=skip_special_tokens)
        response_text = self.tokenizer.decode(
            response_token, skip_special_tokens=skip_special_tokens
        )
        new_row = pd.DataFrame(
            {
                "step": [self.global_steps],
                "reward": [reward],
                "prompt": [prompt_text],
                "response": [response_text],
            }
        )
        self.experiences_example_table = pd.concat(
            [self.experiences_example_table, new_row], ignore_index=True
        )

    def _log_experiences(self, experiences: Experiences) -> None:
        skip_special_tokens = False
        reward_max_id = torch.argmax(experiences.rewards)
        self._log_single_experience(experiences, reward_max_id, skip_special_tokens)

        reward_min_id = torch.argmin(experiences.rewards)
        self._log_single_experience(experiences, reward_min_id, skip_special_tokens)

        if self.global_steps % self.config.trainer.sync_freq == 0:
            self.logger.log_table(
                "rollout_examples", self.experiences_example_table, self.global_steps
            )
            self.reset_experiences_example_table()

    def save_checkpoint(self) -> None:
        self._save_checkpoint()

    def sync_weight(self) -> None:
        self.actor_rollout_wg.sync_weight()

    def set_mode(self, algorithm_type: AlgorithmType = AlgorithmType.PPO) -> None:
        self.actor_rollout_wg.set_mode(algorithm_type)
        if self.algorithm_type.is_sft() and (not algorithm_type.is_sft()):
            self.sft_to_rft()
        self.algorithm_type = algorithm_type

    def sft_to_rft(self) -> None:
        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(
                    self.config.trainer.resume_mode, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_mode
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        print(f"Loading actor from {actor_path} to ref_policy_wg")
        self.ref_policy_wg.load_checkpoint(actor_path, del_local_after_load=False)
        self.actor_rollout_wg.clear_optimizer_state()
        if self.use_critic:
            self.critic_wg.clear_optimizer_state()
        print("sft to rft finished")

    def shutdown(self) -> None:
        pass
