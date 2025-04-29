# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Modified from dp_actor.py
"""

import itertools
from typing import Tuple

import torch
import verl.utils.torch_functional as verl_F
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

from trinity.common.constants import AlgorithmType
from trinity.trainer.verl import core_algos

__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.algorithm_type = AlgorithmType.PPO

    def set_mode(self, algorithm_type: AlgorithmType = AlgorithmType.PPO):
        self.algorithm_type = algorithm_type

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(
                            rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                        )
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(
                    input_ids_rmpad, shifts=-1, dims=1
                )  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(
                    0
                )  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                if self.algorithm_type.is_sft():  # SFT
                    loss_fct = nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                    if self.use_ulysses_sp:
                        loss = gather_outpus_and_unpad(
                            loss, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    response_mask = attention_mask[:, -response_length:].bool()
                    # pad back to (bsz, seqlen)
                    full_loss = pad_input(
                        hidden_states=loss.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    ).squeeze(-1)
                    full_loss = torch.where(
                        response_mask, full_loss[:, -response_length - 1 : -1], 0.0
                    )
                    full_loss = full_loss.sum(-1) / response_mask.sum(-1)
                    full_loss = full_loss.mean()
                    return full_loss

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(
                    logits_rmpad
                )  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    entropy_rmpad = gather_outpus_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                entropy = full_entropy.squeeze(-1)[
                    :, -response_length - 1 : -1
                ]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[
                    :, -response_length - 1 : -1
                ]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                if self.algorithm_type.is_sft():
                    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                    response_mask = attention_mask[:, -response_length:].bool()
                    response_labels = torch.where(
                        response_mask, input_ids[:, -response_length:], -100
                    )
                    response_logits = logits[:, -response_length - 1 : -1, :]
                    loss = loss_fct(
                        response_logits.reshape(-1, response_logits.shape[-1]),
                        response_labels.reshape(-1),
                    )
                    loss = loss.view(response_labels.shape)
                    loss = loss.sum(-1) / response_mask.sum(-1)
                    loss = loss.mean()
                    return loss
                logits.div_(temperature)
                logits = logits[
                    :, -response_length - 1 : -1, :
                ]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
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
        ]  # temperature must be in the data.meta_info to avoid slient error
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
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):  # noqa: C901
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error

        algorithm_type: AlgorithmType = self.config.get("algorithm_type", AlgorithmType.PPO)
        if self.algorithm_type.is_rft():
            select_keys = [
                "responses",
                "input_ids",
                "attention_mask",
                "position_ids",
                "old_log_probs",
                "advantages",
                "response_mask",
            ]
            if self.config.use_kl_loss:
                select_keys.append("ref_log_prob")

            if algorithm_type == AlgorithmType.PAIRWISE_OPMD:
                select_keys.append("token_level_scores")
        elif self.algorithm_type.is_dpo():
            select_keys = [
                "attention_mask",
                "input_ids",
                "position_ids",
                "response_mask",
                "responses",
                "ref_log_prob",
            ]
        else:  # sft
            select_keys = [
                "attention_mask",
                "input_ids",
                "position_ids",
                "response_mask",
                "responses",
            ]
        use_uid = self.config.get("use_uid", False)

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs or ((algorithm_type == AlgorithmType.PAIRWISE_OPMD) and use_uid):
            # TODO: for now, we treat algorithm_type == AlgorithmType.PAIRWISE_OPMD in the same way that
            # has_multi_modal_inputs was treated originally (to handle non_tensor_select_keys);
            # need to double check if this is the best approach.
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = []
            if has_multi_modal_inputs:
                non_tensor_select_keys.append("multi_modal_inputs")
            if (algorithm_type == AlgorithmType.PAIRWISE_OPMD) and use_uid:
                non_tensor_select_keys.append("uid")
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
                if has_multi_modal_inputs or (
                    (algorithm_type == AlgorithmType.PAIRWISE_OPMD) and use_uid
                ):
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
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {
                            **data.batch.to(torch.cuda.current_device()),
                            **data.non_tensor_batch,
                        }
                    else:
                        data = data.to(
                            torch.cuda.current_device()
                        )  # actor device is cpu when using offload

                    # TODO: it is better to unify the returns of several modes (sft, dpo)
                    if self.algorithm_type.is_sft():
                        policy_loss = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature
                        )

                    elif self.algorithm_type.is_dpo():
                        response_mask = data["response_mask"]

                        _, log_prob = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature
                        )
                        if self.config.use_kl_loss:
                            ref_log_prob = data["ref_log_prob"]
                        else:
                            ref_log_prob = None

                        (
                            policy_loss,
                            chosen_reward,
                            rejected_reward,
                        ) = core_algos.compute_policy_loss_dpo(
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            eos_mask=response_mask,
                            beta=self.config.kl_loss_coef,
                            # label_smoothing=self.config.label_smoothing # TODO: add configs for dpo
                        )

                    else:  # rft
                        responses = data["responses"]
                        response_length = responses.size(1)
                        attention_mask = data["attention_mask"]
                        # response_mask = attention_mask[:, -response_length:]
                        response_mask = data["response_mask"]
                        assert response_mask.shape == attention_mask[:, -response_length:].shape
                        old_log_prob = data["old_log_probs"]
                        advantages = data["advantages"]

                        clip_ratio = self.config.clip_ratio
                        entropy_coeff = self.config.entropy_coeff

                        tau = self.config.get("tau", 1.0)
                        token_level_scores = None
                        index = None
                        if algorithm_type == AlgorithmType.PAIRWISE_OPMD:
                            token_level_scores = data["token_level_scores"]
                            if use_uid:
                                index = data["uid"]

                        # all return: (bsz, response_length)
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature
                        )

                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            eos_mask=response_mask,
                            algorithm_type=algorithm_type,
                            advantages=advantages,
                            cliprange=clip_ratio,
                            # for opmd / pairwise_opmd
                            tau=tau,
                            token_level_scores=token_level_scores,
                            index=index,
                        )
                        # compute entropy loss from entropy
                        entropy_loss = verl_F.masked_mean(entropy, response_mask)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = data["ref_log_prob"]
                            # compute kl loss
                            kld = core_algos.kl_penalty(
                                logprob=log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = masked_mean(kld, response_mask)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    if self.algorithm_type.is_rft():
                        data = {
                            "actor/entropy_loss": entropy_loss.detach().item(),
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                        }
                    elif self.algorithm_type.is_dpo():
                        data = {
                            "dpo/loss": policy_loss.detach().item(),
                            "dpo/loss_mean": loss.detach().item(),
                            "dpo/chosen_reward": chosen_reward.detach().mean().item(),
                            "dpo/rejected_reward": rejected_reward.detach().mean().item(),
                            "dpo/accuracy_mean": (chosen_reward > rejected_reward)
                            .float()
                            .mean()
                            .item(),
                        }
                    else:
                        data = {
                            "sft/loss": loss.detach().item(),
                        }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
