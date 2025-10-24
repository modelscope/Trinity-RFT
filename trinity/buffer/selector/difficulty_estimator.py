from typing import List

import numpy as np

from trinity.utils.log import get_logger


class BaseBetaPREstimator:
    n: int
    m: int
    lamb: float
    rho: float
    alphas: np.ndarray
    betas: np.ndarray

    def __init__(self, n: int, m: int = 16, lamb: float = 0.2, rho: float = 0.2):
        """
        alpha_{t+1} = (1 - lamb) * alpha_t + (1 - rho) * bar{s} + rho * tilde{s}
        beta_{t+1} = (1 - lamb) beta_t + (1 - rho) * bar{f} + rho * tilde{f}

        Args:
            n (int): number of tasks.
            m (int): repeat times per tasks.
            timeout (lamb): discount factor of historical estimation.
            rho (float): weight of pseudo counts.
        """
        self.n = n
        self.m = m
        self.lamb = lamb
        self.rho = rho
        self.alphas = np.ones(n, dtype=float)
        self.betas = np.ones(n, dtype=float)
        self.logger = get_logger("BetaPREstimator")
        self.logger.debug(
            f"{self.n=}, {self.m=}, {self.lamb=}, {self.rho=}, {self.alphas=}, {self.betas=}"
        )

    def set(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas

    def _update(self, s_bar, f_bar, p_tilde):
        self.alphas = (
            (1 - self.lamb) * self.alphas
            + self.lamb
            + (1 - self.rho) * s_bar
            + self.rho * p_tilde * self.m
        )
        self.betas = (
            (1 - self.lamb) * self.betas
            + self.lamb
            + (1 - self.rho) * f_bar
            + self.rho * (1 - p_tilde) * self.m
        )

    def update(self, ref_indices: List[int], ref_pass_rates: List[float]):
        raise NotImplementedError

    def predict_pr(self, rng=None, indices=None, do_sample=False):
        if rng is None:
            rng = np.random.default_rng()
        if indices is None:
            indices = np.arange(self.n)
        if not do_sample:
            return self.alphas[indices] / (self.alphas[indices] + self.betas[indices])
        else:
            return rng.beta(self.alphas[indices], self.betas[indices])

    def equivalent_count(self, indices=None):
        if indices is None:
            indices = np.arange(self.n)
        return self.alphas[indices] + self.betas[indices]


class InterpolationBetaPREstimator(BaseBetaPREstimator):
    def __init__(
        self,
        features: np.ndarray,
        m: int,
        lamb,
        rho,
        cap_coef_update_discount=0.9,
        adaptive_rho=False,
    ):
        super(InterpolationBetaPREstimator, self).__init__(len(features), m, lamb, rho)
        self.features = features  # [D, 2]
        self.cap_coef = None
        self.cap_coef_update_discount = cap_coef_update_discount
        self.adaptive_rho = adaptive_rho

    def update(self, ref_indices: List[int], ref_pass_rates: List[float]):
        ref_pass_rate = np.mean(ref_pass_rates)
        ref_anchor_pass_rates = np.mean(self.features[ref_indices], axis=0)
        cap_estimate = (ref_pass_rate - ref_anchor_pass_rates[0]) / (
            ref_anchor_pass_rates[1] - ref_anchor_pass_rates[0] + 1e-6
        )
        if self.cap_coef is None:
            self.cap_coef = cap_estimate
        else:
            self.cap_coef = (
                self.cap_coef_update_discount * self.cap_coef
                + (1 - self.cap_coef_update_discount) * cap_estimate
            )
        s_bar = np.zeros(self.n, dtype=float)
        s_bar[ref_indices] = np.array(ref_pass_rates) * self.m
        f_bar = np.zeros(self.n, dtype=float)
        f_bar[ref_indices] = (1 - np.array(ref_pass_rates)) * self.m
        p_tilde = np.clip(
            (self.features[:, 1] - self.features[:, 0]) * self.cap_coef + self.features[:, 0], 0, 1
        )

        predicted_pass_rates = p_tilde[ref_indices]
        mean_abs_error = np.mean(np.abs(np.array(predicted_pass_rates) - np.array(ref_pass_rates)))
        if self.adaptive_rho and mean_abs_error >= 0.25:
            self.rho = self.rho * 0.5
        self.logger.debug(f"{mean_abs_error=}, {self.rho=}")
        p_tilde[ref_indices] = np.array(ref_pass_rates)

        self._update(s_bar, f_bar, p_tilde)
