import copy
import math
from typing import Any, Dict, List, Optional, Union

from data_juicer.core import Analyzer
from data_juicer.core.data import NestedDataset
from data_juicer.utils.constant import StatsKeys
from jsonargparse import Namespace, namespace_to_dict
from jsonargparse.typing import PositiveFloat, PositiveInt
from scipy.stats import norm

from trinity.data.controllers.default_ops import DEFAULT_CLEANER
from trinity.data.core.dataset import RftDataset
from trinity.data.processors.base import BaseDataProcessor
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class DataCleaner(BaseDataProcessor):
    """
    Data cleaner that iteratively cleans dataset until targets are met.
    If min_size_ratio is not None, directly apply the threshold of each OP
    in dj_cfg.

    Features:
    1. Iterative cleaning with max tries
    2. Target-based cleaning criteria
    3. Progress tracking
    4. Automatic metric tracking
    """

    def __init__(
        self,
        dj_cfg: Optional[Namespace],
        clean_strategy: str = "iterative",
        min_size_ratio: PositiveFloat = None,
        data_dist: Optional[str] = "gaussian",
        op_weights: dict = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param dj_cfg: Inited Data-Juicer config.
        :param clean_strategy: The strategy to clean the data.
        :param min_size_ratio: The minimum ratio of data to be retained.
        :param data_dist: The assumption data distribution to estimate the
            threshold for each op.
        """
        self.dj_cfg = self.keep_cleaner_op_cfg(dj_cfg)
        super().__init__(self.dj_cfg, **kwargs)
        self.clean_strategy: Union[None, str] = clean_strategy
        self.clean_history: List[Dict[str, Any]] = []
        self.min_size_ratio = min_size_ratio
        self.data_dist = data_dist
        self.op_name_to_stats_key = {}
        self.op_weights = op_weights

    def keep_cleaner_op_cfg(self, dj_cfg):
        """Only consider cleaner op in data-juicer configs."""
        dj_cfg = copy.deepcopy(dj_cfg)
        for i in range(len(dj_cfg.process)):
            if isinstance(dj_cfg.process[i], Namespace):
                dj_cfg.process[i] = namespace_to_dict(dj_cfg.process[i])
        new_process = []
        for op in dj_cfg.process:
            op_name = list(op.keys())[0]
            for dimension in DEFAULT_CLEANER:
                if op_name in DEFAULT_CLEANER[dimension]:
                    new_process.append(op)
                    break
        dj_cfg.process = new_process
        return dj_cfg

    def calculate_thresholds(self, mean, std, proportion):
        """
        Given the data distribution assumption, calculate the
        thresholds given the mean, std and proportion.
        """
        proportion = max(0, min(1, proportion))
        if self.data_dist == "gaussian":  # Gaussian distribution
            lower_threshold = norm.ppf(proportion, loc=mean, scale=std)
            upper_threshold = norm.ppf(1 - proportion, loc=mean, scale=std)
        else:  # Uniform distribution
            range_width = std * math.sqrt(12)
            a = mean - range_width / 2
            b = mean + range_width / 2

            lower_threshold = a + proportion * (b - a)
            upper_threshold = b - proportion * (b - a)

        return lower_threshold, upper_threshold

    def update_op_threshold(
        self,
        cur_step: PositiveInt,
        stats_key_to_mean: dict,
        stats_key_to_std: dict,
        op_name_to_stats_key: dict,
        clean_ratio_per_iter: PositiveFloat = 0.1,
    ):
        """
        Update threshold for each iteration. Assume the distribution is a
        Gaussian distribution and estimate the threshold to remove
        `cur_step * clean_ratio_per_iter * op_weight` ratio data for each
        op.
        """
        exe_cfg = self.dj_executor.cfg
        for i in range(len(exe_cfg.process)):
            if isinstance(exe_cfg.process[i], Namespace):
                exe_cfg.process[i] = namespace_to_dict(exe_cfg.process[i])

        update_record = {}
        for process in exe_cfg.process:
            op_name, args = list(process.items())[0]
            op_weight = self.op_weights.get(op_name, 1)
            update_record[op_name] = {}

            temp_args = copy.deepcopy(args)
            if op_name not in op_name_to_stats_key:
                # skip the op such as `clean_email_mapper`
                continue

            stats_keys = op_name_to_stats_key[op_name]
            for stats_key in stats_keys:
                if stats_key in stats_key_to_mean:
                    cur_mean = stats_key_to_mean[stats_key]
                    cur_std = stats_key_to_std[stats_key]
                    cur_proportion = cur_step * clean_ratio_per_iter * op_weight
                    min_th, max_th = self.calculate_thresholds(cur_mean, cur_std, cur_proportion)
                    for arg_name in temp_args.keys():
                        new_val = None
                        if "min" in arg_name:
                            new_val = min_th
                        if "max" in arg_name:
                            new_val = max_th
                        if new_val is not None and str(new_val) != "nan":
                            logger.info(
                                f"Iterative cleaning for op {op_name}, "
                                f"changed its threshold "
                                f"{arg_name}={args[arg_name]} into "
                                f"{arg_name}={new_val}"
                            )
                            args[arg_name] = new_val
                            update_record[op_name][arg_name] = new_val

            logger.info("Arguments have been updated: " + str(update_record))

    def process(
        self, datasets: List[RftDataset], max_tries: PositiveInt = 5, **kwargs: Any
    ) -> RftDataset:
        """Clean dataset iteratively until targets are met"""

        rft_dataset = datasets[0]
        dataset = NestedDataset.from_dict(rft_dataset.data.to_dict())

        original_size = len(dataset)
        self.clean_history = []

        # min_size_ratio is None, execute dj_executor one time directly.
        if self.min_size_ratio is None:
            max_tries = 1
        # min_size_ratio is not None, need to execute analyzer to get mean,
        # std of each stat, and the mapping between op name and stat key.
        else:
            logger.info("Executing Data-Juicer analyzer...")
            analyzer = Analyzer(self.dj_cfg)
            analyzer.run(dataset, skip_export=True)
            df = analyzer.overall_result
            mean_series = df[df.index == "mean"]
            stats_key_to_mean = mean_series.iloc[0, :].to_dict()
            std_series = df[df.index == "std"]
            stats_key_to_std = std_series.iloc[0, :].to_dict()

            tmp_cfg = copy.deepcopy(self.dj_cfg)
            self.op_name_to_stats_key = StatsKeys.get_access_log(dj_cfg=tmp_cfg, dataset=dataset)

        for try_idx in range(max_tries):
            logger.info(f"Cleaning iteration {try_idx + 1}/{max_tries}")

            # if min_size_ratio is not None, need to step the OP thresholds.
            if self.min_size_ratio is not None:
                self.update_op_threshold(
                    try_idx, stats_key_to_mean, stats_key_to_std, self.op_name_to_stats_key
                )

            # Apply cleaning operators
            dataset = self.dj_executor.run(dataset=dataset)

            # Track metrics
            self.trace_interested_metrics(dataset)
            # TODO: catch dataset state
            # current_metrics = dataset.state.copy()
            # self.clean_history.append(current_metrics)

            # TODO: Check if targets are met
            # if self._check_clean_targets(dataset, clean_targets):
            #     logger.info("Cleaning targets met")
            #     break

            # Check if dataset size reduced too much
            if self.min_size_ratio and len(dataset) < original_size * self.min_size_ratio:
                logger.warning(
                    f"Dataset size reduced by more than {self.min_size_ratio * 100}%. Stopping cleaning."
                )
                break

        self.trace_data_lineage(dataset)
        rft_dataset.data = dataset
        return rft_dataset

    def _check_clean_targets(self, dataset: RftDataset, clean_targets: Dict[str, Any]) -> bool:
        """Check if cleaning targets are met"""
        for metric, (target, comparator) in clean_targets.items():
            try:
                # TODO: compute dataset metrics
                current_value = dataset.compute_metrics([metric])[metric]
                if not comparator(current_value, target):
                    logger.info(
                        f"Target not met for {metric}: " f"current={current_value}, target={target}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to check target for {metric}: {str(e)}")
                return False
        return True

    def get_cleaning_history(self) -> List[Dict[str, Any]]:
        """Get history of metrics during cleaning"""
        return self.clean_history
