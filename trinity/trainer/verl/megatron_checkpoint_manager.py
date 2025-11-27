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
Megatron Checkpoint Manager.
Modified from https://github.com/volcengine/verl/blob/v0.5.0/verl/utils/checkpoint/megatron_checkpoint_manager.py
"""

import json
from collections.abc import Callable
from dataclasses import asdict

import ray
import torch
import torch.distributed
from megatron.core.transformer.enums import AttnBackend
from transformers import GenerationConfig
from verl.utils.checkpoint.megatron_checkpoint_manager import (
    MegatronCheckpointManager as OldMegatronCheckpointManager,
)
from verl.utils.checkpoint.megatron_checkpoint_manager import logger
from verl.utils.fs import local_mkdir_safe
from verl.utils.logger import log_with_rank
from verl.utils.megatron.dist_checkpointing import save_dist_checkpointing
from verl.utils.megatron_utils import (
    get_dist_checkpoint_path,
    get_hf_model_checkpoint_path,
    get_transformer_config_checkpoint_path,
)

from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.verl_trainer import CheckpointMonitor


class MegatronCheckpointManager(OldMegatronCheckpointManager):
    """
    An enhanced version of the original FSDP checkpoint manager that:

    1. Uploads model state dicts to a remote Synchronizer actor (either directly or via checkpoints).
    """

    def __init__(
        self,
        *args,
        ray_namespace: str = "",
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.synchronizer = Synchronizer.get_actor(namespace=ray_namespace)
        self.checkpoint_monitor = CheckpointMonitor.get_actor(
            namespace=ray_namespace,
        )
        self.previous_state_dict_step = None

    def _save_state_dict(self, local_path, global_step):
        dist_checkpoint_path = get_dist_checkpoint_path(local_path)
        hf_ckpt_path = get_hf_model_checkpoint_path(local_path)

        if self.use_dist_checkpointing:
            # Generate state dict for saving
            state_dict = self.generate_state_dict()
            # log_with_rank(f"Generated state dict for saving: {state_dict.keys()}", rank=self.rank, logger=logger)
            # for vpp_rank, model in enumerate(self.model):
            #     if len(self.model) > 1:
            #         model_i_keys = state_dict[f"model{vpp_rank}"].keys()
            #         log_with_rank(f"Generated state dict for saving: {model_i_keys}", rank=self.rank, logger=logger)
            #     else:
            #         log_with_rank(
            #             f"Generated state dict for saving: {state_dict['model'].keys()}", rank=self.rank, logger=logger
            #         )
            # Start Async save if enabled
            async_save_request = save_dist_checkpointing(
                sharded_state_dict=state_dict,
                ckpt_path=dist_checkpoint_path,
                async_save=self.checkpoint_config.async_save,
            )

            if self.rank == 0:
                # Save huggingface config
                self.hf_config.save_pretrained(hf_ckpt_path)

            # Synchronize all async save requests
            if not self.checkpoint_config.async_save:
                assert (
                    async_save_request is None
                ), "Async save request should be None when not using async save."
                torch.distributed.barrier()
        else:
            assert (
                self.use_hf_checkpoint
            ), "use_hf_checkpoint should be True when not using dist checkpointing"
            log_with_rank(
                f"Saving HF model checkpoint to {local_path} with bridge",
                rank=self.rank,
                logger=logger,
            )
            self.bridge.save_weights(self.model, hf_ckpt_path)
            log_with_rank(
                f"Saved bridge checkpoint to {hf_ckpt_path}", rank=self.rank, logger=logger
            )

        if self.rank == 0:
            if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                try:
                    generation_config = GenerationConfig.from_pretrained(
                        self.hf_config.name_or_path
                    )
                    generation_config.save_pretrained(hf_ckpt_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

        def finalize_save_fn():
            # Rank 0 uploads checkpoint to HDFS if hdfs_path is provided
            runtime_context = ray.get_runtime_context()
            node_id = runtime_context.get_node_id()
            job_id = runtime_context.get_job_id()
            ray.get(self.checkpoint_monitor.notify_started.remote(node_id=node_id, job_id=job_id))
            log_with_rank(
                f"Dist checkpointing save completed for {dist_checkpoint_path}",
                rank=self.rank,
                logger=logger,
            )
            ray.get(self.checkpoint_monitor.notify_finished.remote(global_step, True))

        if self.checkpoint_config.async_save:
            assert (
                async_save_request is not None
            ), "Async save request should not be None when using async save."
            async_save_request.add_finalize_fn(finalize_save_fn)
        else:
            finalize_save_fn()

        self.previous_state_dict_step = global_step

    def save_state_dict(  # noqa: C901
        self,
        local_path: str,
        global_step: int = 0,
    ):
        if self.previous_state_dict_step is None:
            # First sync in trainer.prepare
            self.previous_state_dict_step = global_step
            return
        elif self.previous_state_dict_step == global_step:
            # No need to save for sync again
            return

        local_path = local_mkdir_safe(local_path)
        self._save_state_dict(local_path, global_step)
        ray.get(
            self.checkpoint_monitor.register_thread_count.remote(
                global_step, state_dict_thread_count=1
            )
        )

    def save_checkpoint(  # noqa: C901
        self,
        local_path: str,
        global_step: int = 0,
        max_ckpt_to_keep=None,
        save_as_hf: bool = False,
    ):
        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        if (
            max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep  # type: ignore
        ):
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1  # type: ignore
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])  # type: ignore
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]  # type: ignore

        local_path = local_mkdir_safe(local_path)

        state_dict_thread_count = 0
        if self.should_save_model:
            if self.previous_state_dict_step != global_step:
                self._save_state_dict(local_path, global_step)
                state_dict_thread_count += 1

        # Only rank 0 saves the hf config and tokenizer to huggingface path
        # No matter whether we save hf model or not
        if self.rank == 0:
            # Save tokenizer
            hf_config_tokenizer_path = get_hf_model_checkpoint_path(local_path)
            self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved Huggingface tokenizer to {hf_config_tokenizer_path}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

        if self.should_save_extra:
            if self.rank == 0:
                # Save transformer config
                log_with_rank(
                    f"Transformer config: {self.transformer_config}", rank=self.rank, logger=logger
                )
                transformer_config_dict = asdict(self.transformer_config)
                to_convert_types = {torch.dtype: str, AttnBackend: str}
                ignore_types = [Callable]
                pop_keys = []
                for key, value in transformer_config_dict.items():
                    if type(value) in to_convert_types:
                        transformer_config_dict[key] = to_convert_types[type(value)](value)
                    if type(value) in ignore_types:
                        pop_keys.append(key)
                    if callable(value):
                        pop_keys.append(key)
                for key in pop_keys:
                    transformer_config_dict.pop(key)
                transformer_config_path = get_transformer_config_checkpoint_path(local_path)
                with open(transformer_config_path, "w") as f:
                    json.dump(transformer_config_dict, f, indent=2)

        if self.should_save_hf_model or save_as_hf:
            try:
                # wait for everyone to dump to local
                state_dict = self.weight_saver(
                    self.model,
                    self.hf_config,
                    dtype=self.param_dtype,
                    is_value_model=self.is_value_model,
                    tie_word_embeddings=self.share_embeddings_and_output_weights,
                )

                torch.distributed.barrier()
                if self.rank == 0:
                    # TODO: async save or use mbridge to save hf model
                    hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                    import warnings

                    from accelerate import init_empty_weights

                    with init_empty_weights(), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if "mistral7b-rm" in self.config.model.path:
                            from transformers import MistralForSequenceClassification

                            model = MistralForSequenceClassification.from_pretrained(
                                self.config.model.path, torch_dtype=torch.bfloat16
                            )  # use score head instead of lm_head
                            state_dict["score.weight"] = state_dict["score.weight"]
                        else:
                            from transformers import AutoModelForCausalLM

                            model = AutoModelForCausalLM.from_pretrained(
                                self.config.model.path, torch_dtype=torch.bfloat16
                            )
                    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
                    model.save_pretrained(hf_model_ckpt_path, state_dict=state_dict)
                    log_with_rank(
                        f"Saved Huggingface config and tokenizer to {hf_model_ckpt_path}",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )
            except Exception:
                logger.error(
                    f"Failed to save Huggingface model to {local_path}, you can try to set `use_mbridge=true` to save it.",
                    exc_info=True,
                )

        ray.get(
            self.checkpoint_monitor.register_thread_count.remote(
                global_step, state_dict_thread_count=state_dict_thread_count
            )
        )
        self.previous_saved_paths.append(local_path)
