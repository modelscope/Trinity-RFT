from typing import Dict, List

import numpy as np
import torch

from trinity.buffer.reader.file_reader import _HFBatchReader
from trinity.buffer.selector.diff_estimator import InterpolationBetaPREstimator
from trinity.common.config import DataSelectorConfig
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

SELECTORS = Registry("selectors")


class BaseSelector:
    def __init__(self, data_source: _HFBatchReader, config: DataSelectorConfig):
        self.data_source = data_source
        self.config = config

    def get_indices(self, batch_size: int, return_extra_info: bool = False) -> List[int]:
        raise NotImplementedError

    def update(self, indices: List[int], values: List[float]) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict) -> None:
        raise NotImplementedError


@SELECTORS.register_module("sequential")
class SequentialSelector(BaseSelector):
    def __init__(self, data_source: _HFBatchReader, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.num_per_epoch = data_source.num_per_epoch
        self.current_index = 0

    def get_indices(self, batch_size: int, return_extra_info: bool = False) -> List[int]:
        start = self.current_index % self.num_per_epoch
        end = start + batch_size
        assert end <= self.num_per_epoch, f"Batch size ({batch_size}) is too large"
        self.current_index += batch_size
        return list(range(start, end))

    def update(self, indices: List[int], values: List[float]) -> None:
        pass

    def state_dict(self) -> Dict:
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get("current_index", 0)


@SELECTORS.register_module("shuffle")
class ShuffleSelector(BaseSelector):
    def __init__(self, data_source: _HFBatchReader, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.dataset_size = data_source.dataset_size
        self.num_per_epoch = data_source.num_per_epoch
        self.current_index = 0
        self.seed = config.seed
        self.order = self._get_order()

    def _get_order(self) -> List[int]:
        rng = np.random.default_rng(self.seed + self.current_index // self.num_per_epoch)
        return rng.choice(self.dataset_size, self.num_per_epoch, replace=False)

    def get_indices(self, batch_size: int, return_extra_info: bool = False) -> List[int]:
        start = self.current_index % self.num_per_epoch
        end = start + batch_size
        assert end <= self.num_per_epoch, f"Batch size ({batch_size}) is too large"
        ret = self.order[start:end]
        self.current_index += batch_size
        if self.current_index % self.num_per_epoch == 0:
            self.order = self._get_order()
        return ret

    def update(self, indices: List[int], values: List[float]) -> None:
        pass

    def state_dict(self) -> Dict:
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get("current_index", 0)
        self.order = self._get_order()


@SELECTORS.register_module("random")
class RandomSelector(BaseSelector):
    def __init__(self, data_source: _HFBatchReader, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.dataset_size = data_source.dataset_size
        self.num_per_epoch = data_source.num_per_epoch
        self.current_index = 0
        self.seed = config.seed

    def get_indices(self, batch_size, return_extra_info=False):
        rng = np.random.default_rng(self.seed + self.current_index)
        selected_indices = rng.choice(self.dataset_size, batch_size, replace=False)
        self.current_index += batch_size
        if return_extra_info:
            return selected_indices, {}
        else:
            return selected_indices

    def update(self, indices: List[int], values: List[float]) -> None:
        pass

    def state_dict(self) -> Dict:
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get("current_index", 0)


@SELECTORS.register_module("offline_easy2hard")
class OfflineEasy2HardSelector(BaseSelector):
    def __init__(self, data_source, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.logger = get_logger("offline_easy2hard_selector")

        feature_keys = config.feature_keys
        self.features = np.concatenate(
            [np.array(list(data_source.dataset[k]))[:, None] for k in feature_keys], axis=1
        )
        features_with_index = [list(self.features[i]) + [i] for i in range(len(self.features))]
        features_with_index = sorted(features_with_index)[::-1]
        self.logger.debug(f"OfflineEasy2HardSelector, sorted {features_with_index[:20]}")
        self.sorted_index = np.array([i[-1] for i in features_with_index])

        self.num_per_epoch = data_source.num_per_epoch
        self.current_index = 0

    def update(self, indices: List[int], values: List[float]) -> None:
        pass

    def get_indices(self, batch_size, return_extra_info=False):
        start = self.current_index % self.num_per_epoch
        end = start + batch_size
        assert end <= self.num_per_epoch, f"Batch size ({batch_size}) is too large"
        self.current_index += batch_size
        selected_indices = self.sorted_index[start:end]
        if not return_extra_info:
            return selected_indices
        else:
            extra_info = {
                "indices": selected_indices.tolist(),
                "feat1": self.features[selected_indices, 0].tolist(),
                "feat2": self.features[selected_indices, 1].tolist(),
            }
            return selected_indices, extra_info

    def state_dict(self) -> Dict:
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get("current_index", 0)


@SELECTORS.register_module("diff_based")
class DiffBasedSelector(BaseSelector):
    def __init__(self, data_source, config: DataSelectorConfig) -> None:
        super().__init__(data_source, config)
        self.logger = get_logger("diff_based_selector")
        self.diff_estimator = self.build_diff_estimator(data_source.dataset, config)
        self.current_index = 0
        self.seed = config.seed

    def build_diff_estimator(self, dataset, config: DataSelectorConfig):
        self.logger.debug(f"{config=}")
        feature_keys = config.feature_keys
        assert len(feature_keys) == 2
        features = np.concatenate(
            [np.array(list(dataset[k]))[:, None] for k in feature_keys], axis=1
        )
        self.logger.debug(f"{features.shape=}")
        self.logger.debug(f"{features[:5]=}")
        adaptive_rho = hasattr(config, "adaptive_rho") and config.adaptive_rho
        return InterpolationBetaPREstimator(
            features=features,
            m=config.m,
            lamb=config.lamb,
            rho=config.rho,
            adaptive_rho=adaptive_rho,
        )

    def update(self, indices: List[int], values: List[float]) -> None:
        self.diff_estimator.update(indices, values)

    def get_scores(self) -> List[float]:
        rng = np.random.default_rng(self.seed + self.current_index)
        predicted_pr = self.diff_estimator.predict_pr(rng=rng, do_sample=self.config.do_sample)
        scores = -np.abs(self.config.target_reward - predicted_pr)
        return scores

    def get_indices(self, batch_size, return_extra_info=False):
        sampling_scores = self.get_scores()
        sampling_scores = torch.from_numpy(sampling_scores)
        if self.config.tau == 0:
            selected_indices = torch.topk(sampling_scores, batch_size).indices
        else:
            sampling_logits = sampling_scores / self.config.tau
            sampling_logits -= sampling_logits.max()
            sampling_probabilities = torch.softmax(sampling_logits, dim=0)
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.current_index)
            selected_indices = torch.multinomial(
                sampling_probabilities,
                batch_size,
                replacement=False,
                generator=rng,
            )
        self.logger.debug(f"{selected_indices=}")
        self.logger.debug(f"{sampling_scores=}")
        self.logger.debug(f"{sampling_scores[selected_indices]=}")
        self.current_index += batch_size

        if return_extra_info:
            selected_indices_list = selected_indices.tolist()
            alphas = self.diff_estimator.alphas[selected_indices_list]
            betas = self.diff_estimator.betas[selected_indices_list]
            point_est = alphas / (alphas + betas)
            extra_info = {
                "indices": selected_indices_list,
                "scores": sampling_scores[selected_indices].tolist(),
                "alphas": alphas.tolist(),
                "betas": betas.tolist(),
                "point": point_est.tolist(),
            }
            return selected_indices, extra_info
        else:
            return selected_indices

    def state_dict(self) -> Dict:
        return {
            "current_index": self.current_index,
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict.get("current_index", 0)
