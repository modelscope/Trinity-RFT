from collections import defaultdict
from typing import List

from trinity.common.config import Config
from trinity.common.models.model import InferenceModel


def create_rollout_models(
    config: Config,
) -> List[InferenceModel]:
    """Create `engine_num` rollout models.

    Each model has `tensor_parallel_size` workers.
    """
    import ray
    from ray.util.placement_group import placement_group, placement_group_table
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from trinity.common.models.vllm_async_model import vLLMAysncRolloutModel
    from trinity.common.models.vllm_model import vLLMRolloutModel

    engine_num = config.explorer.engine_num
    tensor_parallel_size = config.explorer.tensor_parallel_size
    is_multi_process = config.explorer.tensor_parallel_size > 1

    vllm_engines = []

    if config.explorer.engine_type == "vllm":
        engine_cls = vLLMRolloutModel
    elif config.explorer.engine_type == "vllm_async":
        engine_cls = vLLMAysncRolloutModel
    else:
        raise ValueError(f"Unknown engine type: {config.explorer.engine_type}")

    bundles = [{"GPU": 1} for _ in range(engine_num * tensor_parallel_size)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())

    vllm_engines = []

    # to address https://github.com/ray-project/ray/issues/51117
    # aggregate bundles belonging to the same node
    bundle_node_map = placement_group_table(pg)["bundles_to_node_id"]
    node_bundle_map = defaultdict(list)
    for bundle_id, node_id in bundle_node_map.items():
        node_bundle_map[node_id].append(bundle_id)

    for node_id, bundle_ids in node_bundle_map.items():
        assert len(bundle_ids) % tensor_parallel_size == 0, (
            f"Node {node_id} has {len(bundle_ids)} bundles, "
            f"which is not divisible by tensor_parallel_size({tensor_parallel_size})"
        )
        for i in range(len(bundle_ids) // tensor_parallel_size):
            bundles_for_engine = bundle_ids[
                i * tensor_parallel_size : (i + 1) * tensor_parallel_size
            ]
            config.explorer.bundle_indices = ",".join([str(bid) for bid in bundles_for_engine])
            vllm_engines.append(
                engine_cls.options(
                    num_cpus=0,
                    num_gpus=0 if is_multi_process else 1,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundles_for_engine[0],
                    ),
                ).remote(
                    config=config,
                )
            )
    return vllm_engines
