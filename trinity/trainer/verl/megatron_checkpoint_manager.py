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

from trinity.common.config import SynchronizerConfig
from trinity.common.constants import SyncMethod
from trinity.manager.synchronizer import Synchronizer


class MegatronCheckpointManager(OldMegatronCheckpointManager):
    """
    An enhanced version of the original FSDP checkpoint manager that:

    1. Uploads model state dicts to a remote Synchronizer actor (either directly or via checkpoints).
    """

    def __init__(
        self,
        *args,
        sync_config: SynchronizerConfig = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.synchronizer_config = sync_config
        if sync_config is not None:
            # Retrieve the remote Synchronizer actor using the provided namespace
            self.synchronizer = Synchronizer.get_actor(namespace=sync_config.ray_namespace)
        else:
            self.synchronizer = None

    def _notify_synchronizer_with_step_num(self, global_step):
        """
        Notifies the Synchronizer actor about the current training step number,
        used when SyncMethod is CHECKPOINT.

        Args:
            global_step (int): The current global training step.
        """
        if getattr(self.synchronizer_config, "sync_method", None) == SyncMethod.CHECKPOINT:
            ray.get(
                self.synchronizer.set_model_state_dict_with_step_num.remote(
                    global_step, self.world_size
                )
            )

    def save_checkpoint(  # noqa: C901
        self,
        local_path: str,
        hdfs_path: str = None,
        global_step: int = 0,
        max_ckpt_to_keep=None,
        model_state_dict_only: bool = False,
    ):
        # TODO: if resume from checkpoint, synchronization will save model again, which is unnecessary.
        if global_step == 0 and model_state_dict_only:
            if self.rank == 0:
                ray.get(self.synchronizer.set_model_state_dict.remote(None, global_step))
            return

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
        dist_checkpoint_path = get_dist_checkpoint_path(local_path)

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
            hf_ckpt_path = get_hf_model_checkpoint_path(local_path)
            self.bridge.save_weights(self.model, hf_ckpt_path)
            log_with_rank(
                f"Saved bridge checkpoint to {hf_ckpt_path}", rank=self.rank, logger=logger
            )

        if self.should_save_model:
            # Only rank 0 saves the hf config and tokenizer to huggingface path
            # No matter whether we save hf model or not
            if self.rank == 0:
                # Save tokenizer
                hf_config_tokenizer_path = get_hf_model_checkpoint_path(local_path)
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
                # Save huggingface config
                self.hf_config.save_pretrained(hf_config_tokenizer_path)
                if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                    try:
                        generation_config = GenerationConfig.from_pretrained(
                            self.hf_config.name_or_path
                        )
                        generation_config.save_pretrained(hf_config_tokenizer_path)
                    except Exception:
                        # if the generation config isn't available, we don't save it
                        pass
                log_with_rank(
                    f"Saved Huggingface config and tokenizer to {hf_config_tokenizer_path}",
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

        if self.should_save_hf_model:
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
                hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                import warnings

                from accelerate import init_empty_weights

                with init_empty_weights(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if "mistral7b-rm" in self.config.model.path:
                        from transformers import MistralForSequenceClassification

                        model = MistralForSequenceClassification.from_pretrained(
                            self.config.model.path
                        )  # use score head instead of lm_head
                        state_dict["score.weight"] = state_dict["score.weight"]
                    else:
                        from transformers import AutoModelForCausalLM

                        model = AutoModelForCausalLM.from_pretrained(
                            self.config.model.path, torch_dtype="auto"
                        )
                model.save_pretrained(hf_model_ckpt_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved Huggingface config and tokenizer to {hf_model_ckpt_path}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )

                if hdfs_path is not None:
                    log_with_rank(
                        f"Uploading checkpoint to {hdfs_path}",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )
                    from verl.utils import hdfs_io

                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=hf_model_ckpt_path, dst=hdfs_path, dirs_exist_ok=True)
                    log_with_rank(
                        f"HDFS checkpoint uploaded to {hdfs_path}",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )

        def finalize_save_fn():
            # Rank 0 uploads checkpoint to HDFS if hdfs_path is provided
            log_with_rank(
                f"Dist checkpointing save completed for {dist_checkpoint_path}",
                rank=self.rank,
                logger=logger,
            )
            self._notify_synchronizer_with_step_num(global_step)
            if self.rank == 0:
                if hdfs_path is not None:
                    log_with_rank(
                        f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger
                    )
                    from verl.utils import hdfs_io

                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=dist_checkpoint_path, dst=hdfs_path, dirs_exist_ok=True)
                    hdfs_io.copy(src=hf_config_tokenizer_path, dst=hdfs_path, dirs_exist_ok=True)

        if self.checkpoint_config.async_save:
            assert (
                async_save_request is not None
            ), "Async save request should not be None when using async save."
            async_save_request.add_finalize_fn(finalize_save_fn)
        else:
            finalize_save_fn()

        self.previous_saved_paths.append(local_path)
