"""Common utils for algorithm module.

Modified from https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py
"""

import torch


def masked_loss(values, mask, loss_agg_mode="token-mean", normalizer=None):
    """
    Compute loss from values and mask with various aggregation modes.
    Modified from: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        values (torch.Tensor): Arbitrary shape tensor of values to aggregate.
        mask (torch.BoolTensor or torch.FloatTensor): Same shape as values, 1/True = include, 0 = ignore.
        loss_agg_mode (str): One of the following:
            - "token-mean": mean over all unmasked elements.
            - "seq-mean-token-sum": average over sequences, where each sequence's loss is sum of unmasked values.
            - "seq-mean-token-mean": average over sequences, where each sequence's loss is mean of unmasked values.
            - "seq-mean-token-sum-norm": total sum of unmasked values divided by a fixed normalizer (e.g., seq length).
        normalizer (float or None): Only used in 'seq-mean-token-sum-norm'. If None, uses mask.shape[-1].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if loss_agg_mode == "token-mean":
        return masked_mean(values, mask)

    elif loss_agg_mode == "seq-mean-token-sum":
        # Sum over last dimension (token-level), then take mean across batch (sequence-level)
        seq_losses = masked_sum(values, mask, axis=-1)  # [batch]
        return seq_losses.mean()

    elif loss_agg_mode == "seq-mean-token-mean":
        # Mean over tokens per sequence, then mean over sequences
        seq_losses = masked_mean(values, mask, axis=-1)  # [batch]
        return seq_losses.mean()

    elif loss_agg_mode == "seq-mean-token-sum-norm":
        total_token_sum = masked_sum(values, mask)  # scalar
        norm = normalizer if normalizer is not None else mask.shape[-1]
        return total_token_sum / (norm + 1e-8)

    else:
        raise ValueError(
            f"Invalid loss_agg_mode: {loss_agg_mode}. "
            f"Choose from ['token-mean', 'seq-mean-token-sum', 'seq-mean-token-mean', 'seq-mean-token-sum-norm']"
        )


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """
    Whiten `values` by normalizing with mean and variance computed over `mask`.

    Args:
        values (torch.Tensor): Input tensor.
        mask (torch.Tensor): Boolean tensor of same shape, selects elements for stats.
        shift_mean (bool): If True (default), output is zero-mean;
                           if False, the original mean is re-added after scaling.

    Returns:
        torch.Tensor: Whitened tensor of same shape as `values`.
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def prefix_metrics(src_metrics: dict, prefix: str, dst_metrics: dict = None) -> dict:
    if dst_metrics is None:
        dst_metrics = {}
    for k, v in src_metrics.items():
        dst_metrics[f"{prefix}/{k}"] = v

    return dst_metrics
