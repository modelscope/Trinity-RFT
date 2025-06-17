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
Modified from https://github.com/volcengine/verl/blob/0758489422e8d41a89e6c36d4c477714520f0dcc/verl/workers/actor/dp_actor.py
"""

import itertools
import logging
import os

import torch
from torch import nn
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor as DPActor

from trinity.algorithm import ENTROPY_LOSS_FN, KL_FN, POLICY_LOSS_FN
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
            **algorithm_config.policy_loss_fn_args
        )
        self.kl_loss_fn = KL_FN.get(algorithm_config.kl_loss_fn)(**algorithm_config.kl_loss_fn_args)
        self.entropy_loss_fn = ENTROPY_LOSS_FN.get(algorithm_config.entropy_loss_fn)(
            **algorithm_config.entropy_loss_fn_args
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(
                num_micro_batches
            )
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(
                batch=batch, max_token_len=max_token_len
            )
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]  # type: ignore

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
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
        select_keys_verl2trinity = {
            "old_log_probs": "old_logprob",
            "ref_log_prob": "ref_logprob",
            "response_mask": "action_mask",
            "advantages": "advantages",
        }
        select_keys_trinity2verl = {value: key for key, value in select_keys_verl2trinity.items()}
        for trinity_key in self.policy_loss_fn.select_keys:
            verl_key = select_keys_trinity2verl[trinity_key]
            select_keys.append(verl_key)
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

        # TODO: for pairwise_opmd and use_uid, is it necessary to somehow sort samples within batch by uid,
        # to ensure that there are samples with the same uid within each micro-batch
        # (at which level pairwise loss is computed)?
        # (In comparison, advantage is computed at the level of batch, same for opmd, grpo, etc.)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = (
                        mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
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
                        data = {
                            **data.batch.to(get_torch_device().current_device()),
                            **data.non_tensor_batch,
                        }
                    else:
                        data = data.to(
                            get_torch_device().current_device()
                        )  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = data["response_mask"]
                    assert response_mask.shape == attention_mask[:, -response_length:].shape

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=True
                    )

                    kwargs = {
                        select_keys_verl2trinity[verl_key]: value
                        for verl_key, value in data.items()
                        if verl_key in select_keys_verl2trinity
                    }
                    pg_loss, pg_loss_metrics = self.policy_loss_fn(  # type: ignore
                        logprob=log_prob,
                        **kwargs,
                    )
                    prefix_metrics(
                        src_metrics=pg_loss_metrics, prefix="actor", dst_metrics=micro_batch_metrics
                    )

                    # compute entropy loss from entropy
                    entropy_loss, entropy_loss_metrics = self.entropy_loss_fn(  # type: ignore
                        entropy=entropy,
                        action_mask=response_mask,
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
