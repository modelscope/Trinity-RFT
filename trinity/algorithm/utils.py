"""Common utils for algorithm module.

Modified from https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py
"""


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)
