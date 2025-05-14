import os
import traceback
from typing import Any, Dict, List

import ray

from trinity.common.config import Config
from trinity.data.controllers.default_ops import DIMENSION_STATS_KEYS
from trinity.data.controllers.task_parser import DataTaskParser
from trinity.data.core.dataset import RftDataset
from trinity.data.core.dataset_db import RftDatasetDB
from trinity.data.processors.cleaner import DataCleaner
from trinity.data.processors.human_annotator import DataHumanAnnotator
from trinity.data.processors.synthesizer import DataSynthesizer


class DataActiveIterator:
    """
    Manager for active learning iterations and dataset management.
    """

    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.data_config = config.data
        if (
            self.data_config.agent_model_name is not None
            and self.data_config.agent_model_config is not None
        ):
            # get the api key
            api_key = os.environ.get("OPENAI_API_KEY")
            # initialize the agent
            import agentscope
            from agentscope.models import DashScopeChatWrapper

            agentscope.init(model_configs=[self.data_config.agent_model_config])
            self.llm_agent = DashScopeChatWrapper(
                config_name="_",
                model_name=self.data_config.agent_model_name,
                api_key=api_key,
                stream=False,
            )
        else:
            self.llm_agent = None
        self.task_parser = DataTaskParser(config, self.llm_agent)
        self.dsdb = RftDatasetDB(self.data_config)

        # Priority weights
        # larger positive values means larger scores --> higher priority
        # smaller negative values means lower scores --> higher priority
        self.priority_weights = self.data_config.priority_weights or {
            "difficulty": -0.7,
            "diversity": 0.8,
            "usage_frequency": -0.5,
            "quality": 1.0,
        }
        self.min_priority_score = self.data_config.min_priority_score

        # Statistics tracking
        self.state = {"iterations": 0, "samples_selected": 0, "avg_priority_score": 0.0}

        # make key mapping for OPs
        # consider:
        #   1. text_key: only prompt_key
        #   2. input_keys: [prompt_key, response_key] if they are available
        #   3. field_names: [prompt_key, response_key] if they are available
        self.updated_op_args = {
            "text_key": self.data_config.format.prompt_key,
            "input_keys": [
                self.data_config.format.prompt_key,
            ],
            "field_names": [
                self.data_config.format.prompt_key,
            ],
        }
        if self.data_config.format.response_key != "":
            self.updated_op_args["input_keys"].append(self.data_config.format.response_key)
            self.updated_op_args["field_names"].append(self.data_config.format.response_key)

    # flake8: noqa: C901
    def run(self):
        """Run the active iterator."""
        # step 1. parse the dj config
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

        # step 2. load dataset
        try:
            dataset = RftDataset(self.data_config)
        except Exception:
            traceback.print_exc()
            return 2, "RftDataset loading failed."

        # step 3. load cleaner
        try:
            if hit_cleaner:
                cleaner = DataCleaner(
                    dj_config,
                    clean_strategy=self.data_config.clean_strategy,
                    min_size_ratio=self.data_config.min_size_ratio,
                    data_dist=self.data_config.data_dist,
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

        # step 4. apply processors to calculate scores of different dimensions
        try:
            res_dataset = dataset
            if hit_cleaner:
                res_dataset = cleaner.process([res_dataset])
            if hit_synthesizer:
                res_dataset = synthesizer.process([res_dataset])
            if hit_human_annotator:
                res_dataset = human_annotator.process([res_dataset])
        except Exception:
            traceback.print_exc()
            return 4, "DataProcessors processing failed."

        # step 5. calculate the average and final scores, including priority
        try:
            if hit_cleaner:
                scored_dataset = self._group_scores(res_dataset)
                scored_dataset = self._compute_priority_scores(scored_dataset)
            else:
                scored_dataset = res_dataset
        except Exception:
            traceback.print_exc()
            return 5, "Grouping and computing priority score failed."

        # step 6. track lineage if they are changed
        try:
            res_dataset = scored_dataset
        except Exception:
            traceback.print_exc()
            return 6, "Tracking lineage failed."

        # step 7. export the result to the database
        try:
            self.dsdb.add_entries(res_dataset)
        except Exception:
            traceback.print_exc()
            return 7, "Exporting result to database failed."

        return 0, "success"

    def _group_scores(self, dataset: RftDataset) -> RftDataset:
        # for perplexity, normalize them with the max value.
        from data_juicer.utils.constant import Fields

        stats_min_max = {}
        for stats in dataset.data.features[Fields.stats]:
            all_stats = [
                sample[Fields.stats][stats] for sample in dataset.data if Fields.stats in sample
            ]
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

        sample["priority"] = [score]
        return sample

    def _compute_diversity_score(self) -> float:
        """Compute diversity score based on embedding distance"""
        return 0.0  # TODO: Implement

    def _compute_priority_scores(self, dataset: RftDataset) -> RftDataset:
        """Compute utility scores for all samples in dataset"""
        dataset.data = dataset.data.map(self._compute_combined_score)
        return dataset

    def _select_top_k(self, dataset: RftDataset, k: int) -> List:
        """Select top-k samples based on utility scores"""
        return dataset.data.sort("priority", reverse=True).take(k).to_list()

    @ray.method(num_returns=1)
    def select_batch(self, dataset: RftDataset, batch_size: int) -> List[Dict[str, Any]]:
        """Select a batch of samples for training"""

        # Get utility scores for all samples
        dataset = self._compute_priority_scores(dataset)

        # Filter by minimum utility threshold
        dataset.data = dataset.data.filter(lambda s: s["priority"] >= self.min_priority_score)

        # Select top-k samples
        selected_samples = self._select_top_k(dataset, batch_size)

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
