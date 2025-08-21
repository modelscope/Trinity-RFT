# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor.
Modified from https://github.com/volcengine/verl/blob/v0.4.1/verl/workers/actor/dp_actor.py
"""

import logging
import os

import torch
from torch import nn
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor as DPActor

from trinity.algorithm import ENTROPY_LOSS_FN, KL_FN, POLICY_LOSS_FN
from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import DummyEntropyLossFn
from trinity.algorithm.kl_fn.kl_fn import DummyKLFn
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import AlgorithmConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(DPActor):
    def __init__(
        self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config, actor_module, actor_optimizer)
        self.policy_loss_fn = None
        self.kl_loss_fn = None
        self.entropy_loss_fn = None

    def set_algorithm(self, algorithm_config: AlgorithmConfig):
        self.policy_loss_fn = POLICY_LOSS_FN.get(algorithm_config.policy_loss_fn)(
            backend="verl", **algorithm_config.policy_loss_fn_args
        )
        self.kl_loss_fn = KL_FN.get(algorithm_config.kl_loss_fn)(**algorithm_config.kl_loss_fn_args)
        self.entropy_loss_fn = ENTROPY_LOSS_FN.get(algorithm_config.entropy_loss_fn)(
            **algorithm_config.entropy_loss_fn_args
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):  # noqa: C901
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid silent error
        select_keys = [
            "input_ids",
            "position_ids",
            "attention_mask",
            "responses",
            "response_mask",
        ]
        select_keys.extend(self.policy_loss_fn.select_keys)
        if not isinstance(self.kl_loss_fn, DummyKLFn):
            select_keys.append("ref_log_prob")
        select_keys = list(set(select_keys))
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = (
                            self.config.ppo_max_token_len_per_gpu
                            * self.ulysses_sequence_parallel_size
                        )
                        (
                            rearranged_text_micro_batches_tds,
                            textual_indices,
                        ) = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )

                        for current_original_indices, text_mb_td in zip(
                            textual_indices, rearranged_text_micro_batches_tds
                        ):
                            current_mm_inputs_list = [
                                all_multi_modal_inputs_list[idx] for idx in current_original_indices
                            ]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size
                            // self.config.ppo_micro_batch_size_per_gpu
                        )
                        num_micro_batches = (
                            mini_batch.batch.batch_size[0]
                            // self.config.ppo_micro_batch_size_per_gpu
                        )
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(
                            num_micro_batches
                        )
                elif self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    micro_batch_metrics = {}

                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [
                                    {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()}
                                    for item_dict in v
                                ]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = data["response_mask"]
                    assert response_mask.shape == attention_mask[:, -response_length:].shape

                    # all return: (bsz, response_length)
                    calculate_entropy = self.entropy_loss_fn != DummyEntropyLossFn
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    pg_loss, pg_loss_metrics = self.policy_loss_fn(  # type: ignore
                        logprob=log_prob, **data
                    )
                    prefix_metrics(
                        src_metrics=pg_loss_metrics, prefix="actor", dst_metrics=micro_batch_metrics
                    )

                    # compute entropy loss from entropy
                    entropy_loss, entropy_loss_metrics = self.entropy_loss_fn(  # type: ignore
                        entropy=entropy,
                        action_mask=response_mask,
                        **data,
                    )
                    prefix_metrics(
                        src_metrics=entropy_loss_metrics,
                        prefix="actor",
                        dst_metrics=micro_batch_metrics,
                    )

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss

                    kl_loss, kl_loss_metrics = self.kl_loss_fn.calculate_kl_loss(
                        logprob=log_prob,
                        ref_logprob=data.get("ref_log_prob", None),
                        response_mask=response_mask,
                    )
                    prefix_metrics(
                        src_metrics=kl_loss_metrics,
                        prefix="actor",
                        dst_metrics=micro_batch_metrics,
                    )
                    policy_loss = policy_loss + kl_loss

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
