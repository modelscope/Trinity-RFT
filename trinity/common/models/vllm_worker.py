# -*- coding: utf-8 -*-
"""Custom vLLM Worker."""
import ray
import torch
import torch.distributed

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
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"
        self._update_with_checkpoint = update_with_checkpoint
        if self._update_with_checkpoint:
            logger.info(
                f"init_process_group (checkpoint): address={master_address}:{master_port}, rank={torch.distributed.get_rank()}, rank_offset={rank_offset}, world_size={world_size}"
            )
            self._weight_update_rank = torch.distributed.get_rank() + rank_offset
        else:
            logger.info(
                f"init_process_group (nccl): rank={torch.distributed.get_rank()}, rank_offset={rank_offset}, world_size={world_size}"
            )
            self._weight_update_rank = torch.distributed.get_rank() + rank_offset

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
        logger.info(
            f"init_process_group: master_address={master_address}, master_port={master_port}, "
            f"rank={self._weight_update_rank}, world_size={world_size}, group_name={group_name}"
        )
        self._explorer_actor = None

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if self._weight_update_rank == 0:
            if self._explorer_actor is None:
                self._explorer_actor = ray.get_actor(name="explorer")
            weight = ray.get(self._explorer_actor.get_weight.remote(name))
            weight = weight.to(self.device)
        else:
            weight = torch.empty(shape, dtype=dtype, device="cuda")

        torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        weight = weight.type(self.model_config.dtype)

        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight
