import json
import os
import threading
import warnings
from dataclasses import asdict
from typing import Optional

import ray
import torch
from accelerate import init_empty_weights
from torch.distributed.fsdp import (
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from transformers import GenerationConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import (
    FSDPCheckpointManager as OldFSDPCheckpointManager,
)
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPConfig, logger
from verl.utils.device import is_cuda_available
from verl.utils.fs import local_mkdir_safe
from verl.utils.fsdp_utils import (
    fsdp_version,
    get_fsdp_full_state_dict,
    get_fsdp_state_ctx,
)
from verl.utils.logger import log_with_rank

from trinity.common.synchronizer import Synchronizer


class FSDPCheckpointManager(OldFSDPCheckpointManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs.pop("config", None)
        if config is not None:
            self.synchronizer = Synchronizer.get_actor(namespace=config.ray_namespace)
        else:
            self.synchronizer = None
        self._model_state_dict_thread = None
        self._optimizer_state_dict_thread = None
        self._extra_state_dict_thread = None
        self._save_model_thread = None

    def upload_state_dict(self, trainer_step: int):
        assert self.synchronizer is not None
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
            state_dict = self.model.state_dict()
        if self.rank == 0:
            ray.get(self.synchronizer.set_model_state_dict.remote(state_dict, trainer_step))

    def save_checkpoint(  # noqa: C901
        self,
        local_path: str,
        hdfs_path: str = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        model_state_dict_only: bool = False,
    ):
        """
        modified from verl.utils.checkpoint.fsdp_checkpoint_manager.py:save_checkpoint
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path, only rank 0 should do this
        if (
            self.rank == 0
            and max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep  # type: ignore
        ):
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1  # type: ignore
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])  # type: ignore
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]  # type: ignore

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert (
                self.model is not None
            ), "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert (
                self.optimizer is not None
            ), "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(
                self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
            ):
                model_path = os.path.join(
                    local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                optim_path = os.path.join(
                    local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
                )
                extra_path = os.path.join(
                    local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
                )

                if self.should_save_model:
                    model_state_dict = self.model.state_dict()
                    # torch.save(model_state_dict, model_path)
                    if self._model_state_dict_thread is not None:
                        self._model_state_dict_thread.join()

                    def _save_model_state_dict():
                        torch.save(model_state_dict, model_path)
                        log_with_rank(
                            f"Saved model to {os.path.abspath(model_path)}",
                            rank=self.rank,
                            logger=logger,
                        )
                        ray.get(
                            self.synchronizer.set_model_state_dict_with_step_num.remote(
                                global_step, self.world_size
                            )
                        )

                    self._model_state_dict_thread = threading.Thread(
                        target=_save_model_state_dict,
                        # target=torch.save,
                        # args=(model_state_dict, model_path),
                    )
                    self._model_state_dict_thread.start()
                    # log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer and not model_state_dict_only:
                    optimizer_state_dict = self.optimizer.state_dict()
                    # torch.save(optimizer_state_dict, optim_path)
                    if self._optimizer_state_dict_thread is not None:
                        self._optimizer_state_dict_thread.join()

                    def _save_optimizer_state_dict():
                        torch.save(optimizer_state_dict, optim_path)
                        log_with_rank(
                            f"Saved optim to {os.path.abspath(optim_path)}",
                            rank=self.rank,
                            logger=logger,
                        )

                    self._optimizer_state_dict_thread = threading.Thread(
                        target=_save_optimizer_state_dict,
                        # target=torch.save,
                        # args=(optimizer_state_dict, optim_path),
                    )
                    self._optimizer_state_dict_thread.start()
                    # log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra and not model_state_dict_only:
                    lr_scheduler_state_dict = (
                        self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    )
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    # torch.save(extra_state_dict, extra_path)
                    if self._extra_state_dict_thread is not None:
                        self._extra_state_dict_thread.join()

                    def _save_extra_state_dict():
                        torch.save(extra_state_dict, extra_path)
                        log_with_rank(
                            f"Saved extra_state to {os.path.abspath(extra_path)}",
                            rank=self.rank,
                            logger=logger,
                        )

                    self._extra_state_dict_thread = threading.Thread(
                        target=_save_extra_state_dict,
                        # target=torch.save,
                        # args=(extra_state_dict, extra_path),
                    )
                    self._extra_state_dict_thread.start()
                    # log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            if (
                unwrap_model.can_generate()
                and hasattr(model_config, "name_or_path")
                and model_config.name_or_path
            ):
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                generation_config.save_pretrained(hf_config_tokenizer_path)
            else:
                generation_config = None

            model_config.save_pretrained(hf_config_tokenizer_path)
            self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model and not model_state_dict_only:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(
                        f"Unknown architecture {model_config['architectures']}"
                    )

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(
                        model_config, torch_dtype=torch.bfloat16
                    )
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model."
                        )

                # save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                if self._save_model_thread is not None:
                    self._save_model_thread.join()

                def _save_model():
                    save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                    log_with_rank(
                        f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )

                self._save_model_thread = threading.Thread(
                    target=_save_model,
                    # target=save_model.save_pretrained,
                    # args=(hf_local_path, state_dict),
                )
                self._save_model_thread.start()
                self.processing_class.save_pretrained(hf_local_path)
                # log_with_rank(f"Saved hf_model to {os.path.abspath(hf_local_path)}", rank=self.rank, logger=logger, log_only_rank_0=True)
                # del state_dict
                # del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if not model_state_dict_only:
            self.previous_saved_paths.append(local_path)
