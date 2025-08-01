# -*- coding: utf-8 -*-
"""Custom vLLM Worker."""
import ray
import torch
import torch.distributed

from trinity.common.synchronizer import Synchronizer
from trinity.utils.distributed import init_process_group
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class WorkerExtension:
    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        state_dict_meta: list = None,
        explorer_name: str = None,
        namespace: str = None,
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"
        self._state_dict_meta = state_dict_meta
        self._weight_update_rank = torch.distributed.get_rank() + rank_offset
        logger.info(
            f"vLLM starting init_process_group:\n"
            f"  > address={master_address}:{master_port}\n"
            f"  > rank={torch.distributed.get_rank()}\n"
            f"  > rank_offset={rank_offset}\n"
            f"  > world_size={world_size}"
        )
        self._model_update_group = init_process_group(
            host=master_address,
            port=master_port,
            group_name=group_name,
            backend=backend,
            timeout=timeout,
            world_size=world_size,
            rank=self._weight_update_rank,
        )
        torch.distributed.barrier(group=self._model_update_group)
        logger.info("vLLM init_process_group finished.")
        self._explorer_name = explorer_name
        self._namespace = namespace
        self.synchronizer = Synchronizer.get_actor(namespace=self._namespace)

    def update_weight(self):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if self._state_dict_meta is None:
            self._state_dict_meta = ray.get(self.synchronizer.get_state_dict_meta.remote())
        if self._weight_update_rank == 0:
            state_dict, _ = ray.get(self.synchronizer.get_model_state_dict.remote())
        for name, dtype_str, shape in self._state_dict_meta:
            if self._weight_update_rank == 0:
                weight = state_dict[name]
                weight = weight.to(self.device)
            else:
                dtype = getattr(torch, dtype_str.split(".")[-1])
                weight = torch.empty(shape, dtype=dtype, device=self.device)
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)
            weight = weight.type(self.model_config.dtype)
            self.model_runner.model.load_weights(weights=[(name, weight)])
            del weight
        torch.distributed.barrier(group=self._model_update_group)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
