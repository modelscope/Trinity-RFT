# -*- coding: utf-8 -*-
"""Custom vLLM Worker."""
import ray
import torch
import torch.distributed

from trinity.common.constants import EXPLORER_NAME
from trinity.utils.distributed import init_process_group, is_ipv6_address
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
        update_with_checkpoint: bool = True,
        state_dict_meta: list = None,
        namespace: str = None,
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"
        self.set_state_dict_meta(state_dict_meta)
        self._update_with_checkpoint = update_with_checkpoint
        self._weight_update_rank = torch.distributed.get_rank() + rank_offset
        logger.info(
            f"vLLM starting init_process_group ({'checkpoint' if self._update_with_checkpoint else 'nccl'}):\n"
            f"  > address={master_address}:{master_port}\n"
            f"  > rank={torch.distributed.get_rank()}\n"
            f"  > rank_offset={rank_offset}\n"
            f"  > world_size={world_size}"
        )
        if is_ipv6_address(master_address):
            # using tcp://ipv6:port will lead to ValueError
            init_method = f"tcp://[{master_address}]:{master_port}"
        else:
            init_method = f"tcp://{master_address}:{master_port}"

        self._model_update_group = init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timeout,
            world_size=world_size,
            rank=self._weight_update_rank,
            group_name=group_name,
        )
        logger.info("vLLM init_process_group finished.")
        self._namespace = namespace
        self._explorer_actor = None

    def set_state_dict_meta(self, state_dict_meta):
        self._state_dict_meta = state_dict_meta

    def update_weight(self):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        assert self._state_dict_meta is not None
        if self._explorer_actor is None:
            self._explorer_actor = ray.get_actor(name=EXPLORER_NAME, namespace=self._namespace)
        for name, dtype_str, shape in self._state_dict_meta:
            if self._weight_update_rank == 0:
                weight = ray.get(self._explorer_actor.get_weight.remote(name))
                weight = weight.to(self.device)
            else:
                dtype = getattr(torch, dtype_str.split(".")[-1])
                weight = torch.empty(shape, dtype=dtype, device=self.device)
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)
            weight = weight.type(self.model_config.dtype)
            self.model_runner.model.load_weights(weights=[(name, weight)])
            del weight
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
