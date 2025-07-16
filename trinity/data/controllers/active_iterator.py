import os
import threading
import traceback
from copy import deepcopy
from functools import partial
from numbers import Number
from typing import Any, Dict, List, Union

import ray
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import add_suffix_to_filename

from trinity.common.config import BufferConfig, DataPipelineConfig, RewardShapingConfig
from trinity.common.constants import DataProcessorPipelineType, OpType, StorageType
from trinity.data.controllers.default_ops import DIMENSION_STATS_KEYS
from trinity.data.controllers.task_parser import DataTaskParser
from trinity.data.core.dataset import RftDataset, read_from_buffer, write_to_buffer
from trinity.data.processors.cleaner import DataCleaner
from trinity.data.processors.human_annotator import DataHumanAnnotator
from trinity.data.processors.synthesizer import DataSynthesizer
from trinity.data.utils import add_temp_file_buffer, get_temp_file_buffer_path
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class DataActiveIterator:
    """
    Manager for active learning iterations and dataset management.
    """

    def __init__(
        self,
        config: DataPipelineConfig,
        buffer_config: BufferConfig,
        pipeline_type: Union[DataProcessorPipelineType, str] = DataProcessorPipelineType.TASK,
        updated_api_info: Dict[str, Any] = None,
    ):
        """
        The initialization method.

        :param config: the data pipeline config.
        :param buffer_config: the buffer config.
        :param pipeline_type: the type of the activated pipeline.
        :param updated_api_info: the updated API info for OPs. Not available for Data-Juicer config agents.
        """
        self.config = config
        self.buffer_config = buffer_config
        self.pipeline_type = pipeline_type
        if isinstance(self.pipeline_type, str):
            self.pipeline_type = DataProcessorPipelineType(pipeline_type)
        self.updated_api_info = updated_api_info if updated_api_info is not None else {}

        # check if the llm agent is required
        if self.config.agent_model_name is not None:
            # get the api key
            api_key = os.environ.get("OPENAI_API_KEY")
            # initialize the agent
            from agentscope.models import DashScopeChatWrapper

            self.llm_agent = DashScopeChatWrapper(
                config_name="_",
                model_name=self.config.agent_model_name,
                api_key=api_key,
                stream=False,
            )
        else:
            self.llm_agent = None

        # init task parser
        self.task_parser = DataTaskParser(config, self.llm_agent)

        # Priority weights
        # larger positive values means larger scores --> higher priority
        # smaller negative values means lower scores --> higher priority
        self.priority_weights = self.config.priority_weights or {
            "difficulty": -0.7,
            "diversity": 0.8,
            "usage_frequency": -0.5,
            "quality": 1.0,
        }
        self.min_priority_score = self.config.min_priority_score

        # sort with only the top_k
        # keep all data by default
        self.top_k = -1

        # Statistics tracking
        self.state = {"iterations": 0, "samples_selected": 0, "avg_priority_score": 0.0}

        # make key mapping for OPs
        # consider:
        #   1. text_key: only prompt_key
        #   2. input_keys: [prompt_key, response_key] if they are available
        #   3. field_names: [prompt_key, response_key] if they are available
        self.updated_op_args = {
            "text_key": self.config.format.prompt_key,
            "input_keys": [
                self.config.format.prompt_key,
            ],
            "field_names": [
                self.config.format.prompt_key,
            ],
        }
        if self.config.format.response_key != "":
            self.updated_op_args["input_keys"].append(self.config.format.response_key)
            self.updated_op_args["field_names"].append(self.config.format.response_key)

        # if it's a dynamic task pipeline, update the model_params and model_path as well
        if self.pipeline_type == DataProcessorPipelineType.TASK and self.config.dynamic:
            # update the api model path
            if self.updated_api_info.get("model", None) is not None:
                self.updated_op_args["api_or_hf_model"] = self.updated_api_info["model"]
            # update the model_params
            self.updated_op_args["model_params"] = {
                "base_url": self.updated_api_info.get("base_url", None),
                "api_key": self.updated_api_info.get("api_key", None),
            }

            # besides, try to use a temp buffer to store the left task set with dynamic priorities
            temp_file_buffer_name = self.config.input_buffers[0].name + "_temp_file_buffer"
            temp_file_buffer_path = add_suffix_to_filename(self.config.input_buffers[0].path, "_temp_file_buffer") + '.jsonl'

            self.temp_file_buffer_config = deepcopy(self.config.input_buffers[0])
            self.temp_file_buffer_config.name = temp_file_buffer_name
            self.temp_file_buffer_config.path = temp_file_buffer_path
            self.temp_file_buffer_config.storage_type = StorageType.FILE
            self.temp_file_buffer_config.raw = True

            # update batch size -- top_k
            self.top_k = self.buffer_config.read_batch_size
        else:
            self.temp_file_buffer_config = None

    # flake8: noqa: C901
    def run(self, thread_event: threading.Event = None, recompute: bool = True):
        """Run the active iterator."""
        # step 1. parse the dj config
        logger.info("Parsing the Data-Juicer config...")
        try:
            (
                dj_config,
                hit_cleaner,
                hit_synthesizer,
                hit_human_annotator,
            ) = self.task_parser.parse_to_dj_config(self.updated_op_args)
        except Exception:
            traceback.print_exc()
            return 1, "config parsing failed."

        # step 2. prepare rft-dataset from the input buffers
        logger.info("Preparing Rft-Dataset from input buffers...")
        try:
            dataset = RftDataset(self.config, self.buffer_config)
        except Exception:
            traceback.print_exc()
            return 2, "RftDataset loading failed."

        # step 3. load processor
        logger.info("Loading data processors...")
        try:
            if hit_cleaner:
                cleaner = DataCleaner(
                    dj_config,
                    clean_strategy=self.config.clean_strategy,
                    min_size_ratio=self.config.min_size_ratio,
                    data_dist=self.config.data_dist,
                )
            if hit_synthesizer:
                synthesizer = DataSynthesizer(
                    dj_config,
                )
            if hit_human_annotator:
                human_annotator = DataHumanAnnotator(
                    dj_config,
                )
        except Exception:
            traceback.print_exc()
            return 3, "DataCleaner loading failed."

        while True:
            # if a stop event is set, stop!
            if thread_event and thread_event.is_set():
                logger.info("Stop event is set, stopping the pipeline...")
                break

            # step 4. load data from the input buffers for the next batch
            logger.info("Loading data from input buffers for the next batch...")
            try:
                if self.temp_file_buffer_config is not None:
                    # it's a dynamic task pipeline and the temp_file_buffer already exists,
                    # read from the temp_file_buffer
                    temp_file_buffer_path = get_temp_file_buffer_path(self.temp_file_buffer_config.name)
                    if temp_file_buffer_path is not None:
                        # the temp_file_buffer already exists, so read from it
                        self.temp_file_buffer_config.path = temp_file_buffer_path
                        dataset.data = read_from_buffer(self.temp_file_buffer_config, self.buffer_config)
                    else:
                        # add this new temp_file_buffer and load from the original input buffers
                        add_temp_file_buffer(self.temp_file_buffer_config.name, self.temp_file_buffer_config.path)
                        dataset.read_from_buffer()
                else:
                    dataset.read_from_buffer()
                if "priority" not in dataset.data.features:
                    recompute = True
                elif recompute:
                    # remove the "priority" column and computed stats & meta
                    dataset = dataset.remove_columns(["priority", Fields.stats, Fields.meta])
            except StopIteration:
                break
            except Exception:
                traceback.print_exc()
                return 4, "RftDataset loading from buffers failed."

            # step 5. apply processors to calculate scores of different dimensions
            logger.info("Applying data processors to calculate stats...")
            try:
                res_dataset = dataset
                if recompute:
                    if hit_cleaner:
                        res_dataset = cleaner.process([res_dataset])
                    if hit_synthesizer:
                        res_dataset = synthesizer.process([res_dataset])
                    if hit_human_annotator:
                        res_dataset = human_annotator.process([res_dataset])
            except Exception:
                traceback.print_exc()
                return 5, "DataProcessors processing failed."

            # step 6. calculate the average and final scores, including priority
            try:
                if recompute and hit_cleaner:
                    logger.info("Calculating the average and final scores...")
                    scored_dataset = self._group_scores(res_dataset)
                    scored_dataset = self._compute_priority_scores(scored_dataset)
                else:
                    scored_dataset = res_dataset
            except Exception:
                traceback.print_exc()
                return 6, "Grouping and computing priority score failed."

            # step 7. reward shaping. Only available for experience pipeline and the reward shaping config is set
            try:
                if (
                    self.pipeline_type == DataProcessorPipelineType.EXPERIENCE
                    and self.config.reward_shaping is not None
                    and len(self.config.reward_shaping) > 0
                ):
                    logger.info("Rewarding shaping...")
                    reshaped_dataset = self._reward_shaping(scored_dataset)
                else:
                    reshaped_dataset = scored_dataset
            except Exception:
                traceback.print_exc()
                return 7, "Reward shaping failed."

            # step 8. track lineage if they are changed
            try:
                res_dataset = reshaped_dataset
            except Exception:
                traceback.print_exc()
                return 8, "Tracking lineage failed."

            # step 9, sort the dataset by the computed priority
            try:
                if "priority" in res_dataset.data.features:
                    logger.info("Sorting samples by priority...")
                    res_dataset, left_data = res_dataset.sort_by("priority", reverse=True, top_k=self.top_k)
                    if self.temp_file_buffer_config is not None and get_temp_file_buffer_path(self.temp_file_buffer_config.name) is not None:
                        # write left data to the temp buffer
                        self.temp_file_buffer_config.path = get_temp_file_buffer_path(self.temp_file_buffer_config.name)
                        write_to_buffer(self.temp_file_buffer_config, self.buffer_config, left_data)
            except Exception:
                traceback.print_exc()
                return 9, "Sorting results by priority failed."

            # step 10. export the result to the output buffer
            try:
                logger.info("Writing processed data to output buffer...")
                res_dataset.write_to_buffer()
            except Exception:
                traceback.print_exc()
                return 10, "Exporting result to output buffer failed."

            # only one iter for task pipelines
            if self.pipeline_type == DataProcessorPipelineType.TASK:
                break

        try:
            if self.pipeline_type != DataProcessorPipelineType.TASK or not self.config.dynamic:
                # do not release the output buffer if it's a dynamic task pipeline
                dataset.release_output_buffer()
        except Exception:
            traceback.print_exc()
            return -1, "Releasing output buffer failed."

        return 0, "success"

    def _group_scores(self, dataset: RftDataset) -> RftDataset:
        # for perplexity, normalize them with the max value.
        stats_min_max = {}
        for stats in dataset.data.features[Fields.stats]:
            all_stats = [
                sample[Fields.stats][stats] for sample in dataset.data if Fields.stats in sample
            ]
            if len(all_stats) > 0 and isinstance(all_stats[0], Number):
                stats_min_max[stats] = [min(all_stats), max(all_stats)]

        def _group_single(sample):
            stats = sample[Fields.stats]
            for group_score, related_stats in DIMENSION_STATS_KEYS.items():
                total_score = 0.0
                hit_cnt = 0
                details = {}
                for stats_key in related_stats:
                    stats_meta = related_stats[stats_key]
                    if stats_key in stats:
                        # min-max normalization
                        min_val, max_val = stats_meta["range"]
                        if min_val is None or max_val is None:
                            min_val, max_val = stats_min_max[stats_key]
                        current_score = (stats[stats_key] - min_val) / (max_val - min_val)
                        if stats_meta["better"] == "lower":
                            current_score = 1.0 - current_score
                        total_score += current_score
                        hit_cnt += 1
                        # record original stats
                        details[stats_key] = stats[stats_key]
                        # record normalized score
                        details[f"normalized_{stats_key}"] = current_score
                final_score = total_score / hit_cnt if hit_cnt > 0 else 0.0
                sample[Fields.stats][group_score] = final_score
                sample[group_score] = final_score
                sample[f"{group_score}_detail"] = details
            return sample

        dataset.data = dataset.data.map(_group_single)
        return dataset

    def _compute_combined_score(
        self,
        sample,
    ) -> float:
        """Combine different factors into final priority score"""
        if "priority" in sample:
            return sample

        from data_juicer.utils.constant import Fields

        stats = sample[Fields.stats]
        if isinstance(stats, list):
            stats = stats[0]
        score = 0.0

        # Usage frequency penalty
        if "usage_frequency" in self.priority_weights:
            freq = stats.get("consumed_cnt", 0)
            # normalized_freq = min(freq / 10.0, 1.0)  # Normalize to [0,1]
            score += self.priority_weights["usage_frequency"] * freq

        # TODO: Sample diversity (using embedding distance)
        if "diversity" in self.priority_weights:
            diversity = self._compute_diversity_score()
            score += self.priority_weights["diversity"] * diversity

        # Data quality score
        if "quality" in self.priority_weights:
            quality = stats.get("quality_score", 0.5)
            score += self.priority_weights["quality"] * quality

        # Data difficulty score
        if "difficulty" in self.priority_weights:
            difficulty = stats.get("difficulty_score", 0.5)
            score += self.priority_weights["difficulty"] * difficulty

        sample["priority"] = [score] if isinstance(sample[Fields.stats], list) else score
        return sample

    def _compute_diversity_score(self) -> float:
        """Compute diversity score based on embedding distance"""
        return 0.0  # TODO: Implement

    def _compute_priority_scores(self, dataset: RftDataset) -> RftDataset:
        """Compute utility scores for all samples in dataset"""
        dataset.data = dataset.data.map(self._compute_combined_score)
        return dataset

    def _reward_shaping_single(self, sample, reward_shaping_config: RewardShapingConfig):
        tgt_stats = reward_shaping_config.stats_key
        op_type = reward_shaping_config.op_type
        # if the target stats does not exist, skip this stats and return the original sample
        if tgt_stats not in sample[Fields.stats]:
            return sample
        if op_type == OpType.ADD:
            sample[self.config.format.reward_key] += (
                reward_shaping_config.weight * sample[Fields.stats][tgt_stats]
            )
        elif op_type == OpType.MUL:
            sample[self.config.format.reward_key] *= (
                reward_shaping_config.weight * sample[Fields.stats][tgt_stats]
            )
        elif op_type == OpType.SUB:
            sample[self.config.format.reward_key] -= (
                reward_shaping_config.weight * sample[Fields.stats][tgt_stats]
            )
        elif op_type == OpType.DIV:
            sample[self.config.format.reward_key] /= (
                reward_shaping_config.weight * sample[Fields.stats][tgt_stats]
            )
        return sample

    def _reward_shaping(self, rft_dataset: RftDataset) -> RftDataset:
        dataset = rft_dataset.data
        # check if there is a reward column in the dataset. If not, skip!
        if self.config.format.reward_key not in dataset.features:
            return rft_dataset
        # get reward shaping configs
        reward_shaping_configs = self.config.reward_shaping
        for reward_shaping_config in reward_shaping_configs:
            dataset = dataset.map(
                partial(self._reward_shaping_single, reward_shaping_config=reward_shaping_config)
            )

        rft_dataset.data = dataset
        return rft_dataset

    @ray.method(num_returns=1)
    def select_batch(self, dataset: RftDataset, batch_size: int) -> List[Dict[str, Any]]:
        """Select a batch of samples for training"""

        # Get utility scores for all samples
        dataset = self._compute_priority_scores(dataset)

        # Filter by minimum utility threshold
        dataset.data = dataset.data.filter(lambda s: s["priority"] >= self.min_priority_score)

        # Select top-k samples
        dataset.sort_by("priority", reverse=True, top_k=batch_size)
        selected_samples = dataset.data.to_list()

        # Update state
        self._update_state(selected_samples, dataset.data["priority"])

        return selected_samples

    def _update_state(self, selected_samples: List[Dict[str, Any]], scores: List[float]):
        """Update iterator state"""
        self.state["iterations"] += 1
        self.state["samples_selected"] += len(selected_samples)
        self.state["avg_priority_score"] = (
            self.state["avg_priority_score"] * (self.state["iterations"] - 1)
            + sum(scores) / len(scores)
        ) / self.state["iterations"]
        # TODO: support priority scores in different granularity, sample-level,
        #  batch-level, dataset-level

    def get_state(self) -> Dict[str, Any]:
        """Get iterator state"""
        return self.state.copy()
