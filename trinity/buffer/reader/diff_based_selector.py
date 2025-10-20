from typing import List

import numpy as np
import torch

from trinity.common.config import DataSelectorConfig, StorageConfig

from .diff_estimator import InterpolationBetaPREstimator


def build_diff_estimator(dataset, config: DataSelectorConfig):
    print(f"[DEBUG]: {config=}")
    feature_keys = config.feature_keys
    features = np.concat([np.array(list(dataset[k]))[:, None] for k in feature_keys], axis=1)
    print(f"[DEBUG]: {features.shape=}")
    print(f"[DEBUG]: {features[:5]=}")
    adaptive_rho = hasattr(config, "adaptive_rho") and config.adaptive_rho
    return InterpolationBetaPREstimator(
        features=features, m=config.m, lamb=config.lamb, rho=config.rho, adaptive_rho=adaptive_rho
    )


class BaseSelector:
    def __init__(self, data_source, config: DataSelectorConfig):
        self.data_source = data_source
        self.config = config

    def get_indices(self, batch_size: int, return_extra_info: bool = False):
        raise NotImplementedError

    def update(self, indices: List[int], values: List[float]):
        raise NotImplementedError


class RandomSelector(BaseSelector):
    def __init__(self, data_source, config: DataSelectorConfig):
        super().__init__(data_source, config)
        self.n = len(data_source)
        print(f"[DEBUG]: RandomSelector-{self.n=}")

    def get_indices(self, batch_size, return_extra_info=False):
        selected_indices = torch.from_numpy(np.random.permutation(self.n)[:batch_size])
        print(f"[DEBUG]: RandomSelector-{selected_indices=}")
        if return_extra_info:
            return selected_indices, {}
        else:
            return selected_indices

    def update(self, *args, **kwargs):
        pass


class OfflineEasy2HardSelector(BaseSelector):
    def __init__(self, data_source, config: DataSelectorConfig):
        super().__init__(data_source, config)

        feature_keys = config.feature_keys
        self.features = np.concat(
            [np.array(list(data_source[k]))[:, None] for k in feature_keys], axis=1
        )
        features_with_index = [list(self.features[i]) + [i] for i in range(len(self.features))]
        features_with_index = sorted(features_with_index)[::-1]
        print(f"[DEBUG]: OfflineEasy2HardSelector, sorted {features_with_index[:20]}")
        self.sorted_index = np.array([i[2] for i in features_with_index])

        self.n = len(data_source)
        self.current_position = 0

    def update(self, *args, **kwargs) -> None:
        pass

    def get_indices(self, batch_size, return_extra_info=False):
        if self.current_position + batch_size > self.n:
            new_position = self.current_position + batch_size - self.n
            selected_indices = np.concatenate(
                [self.sorted_index[self.current_position :], self.sorted_index[:new_position]]
            )
        else:
            new_position = self.current_position + batch_size
            selected_indices = self.sorted_index[self.current_position : new_position]
        self.current_position = new_position
        if not return_extra_info:
            return selected_indices
        else:
            extra_info = {
                "indices": selected_indices.tolist(),
                "feat1": self.features[selected_indices, 0].tolist(),
                "feat2": self.features[selected_indices, 1].tolist(),
            }
            return selected_indices, extra_info


class DiffBasedSelector(BaseSelector):
    def __init__(self, data_source, config: DataSelectorConfig) -> None:
        super().__init__(data_source, config)
        self.diff_estimator = build_diff_estimator(data_source, config)

    def update(self, indices: List[int], values: List[float]) -> None:
        self.diff_estimator.update(indices, values)

    def get_scores(self) -> List[float]:
        predicted_pr = self.diff_estimator.predict_pr(do_sample=self.config.do_sample)
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
            selected_indices = torch.multinomial(
                sampling_probabilities, batch_size, replacement=False
            )
        print(f"[DEBUG]: {selected_indices=}")
        print(f"[DEBUG]: {sampling_scores=}")
        print(f"[DEBUG]: {sampling_scores[selected_indices]=}")

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


def build_selector(dataset, config: StorageConfig) -> BaseSelector:
    selector_config = config.task_selector
    assert selector_config is not None
    selector_type = selector_config.selector_type
    if selector_type == "random":
        return RandomSelector(dataset, selector_config)
    elif selector_type == "diff":
        return DiffBasedSelector(dataset, selector_config)
    elif selector_type == "offline":
        return OfflineEasy2HardSelector(dataset, selector_config)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")
