# -*- coding: utf-8 -*-
import numpy as np
import torch


def sample_base_of(
    lower_bound: int | float | np.ndarray | torch.Tensor,
    upper_bound: int | float | np.ndarray | torch.Tensor,
    base_matrix: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    if isinstance(lower_bound, np.ndarray):
        lower_bound = lower_bound.reshape(1, -1)
    elif isinstance(lower_bound, torch.Tensor):
        lower_bound = lower_bound.reshape(1, -1).to(base_matrix.device)
    if isinstance(upper_bound, np.ndarray):
        upper_bound = upper_bound.reshape(1, -1)
    elif isinstance(upper_bound, torch.Tensor):
        upper_bound = upper_bound.reshape(1, -1).to(base_matrix.device)
    return lower_bound + (upper_bound - lower_bound) * base_matrix


def lhs_np(
    n: int,
    d: int,
    lower_bound: int | float | np.ndarray,
    upper_bound: int | float | np.ndarray,
) -> np.ndarray:
    """Latin hypercude sampling

    Args:
        n: The number of the sample data
        d: The number of the decision variables
        lower_bound: A number or a vector, the lower bound of the decision variables
        upper_bound: A number or a vector, the upper_bound of the decision variables
    """
    if np.any(lower_bound > upper_bound):
        return None
    intervalSize = 1.0 / n
    # samplePoints[i] is the point that sampled from demension i
    samplePoints = np.empty([d, n])
    for i in range(n):
        samplePoints[:, i] = np.random.uniform(
            low=i * intervalSize, high=(i + 1) * intervalSize, size=d
        )
    for i in range(d):
        np.random.shuffle(samplePoints[i])
    return sample_base_of(lower_bound, upper_bound, samplePoints.T)


def lhs_torch(
    n: int,
    d: int,
    lower_bound: int | float | torch.Tensor,
    upper_bound: int | float | torch.Tensor,
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu"),
):
    if isinstance(lower_bound, torch.Tensor) or isinstance(upper_bound, torch.Tensor):
        if torch.any(lower_bound > upper_bound):
            return None
    elif lower_bound > upper_bound:
        return None
    intervalSize = 1.0 / n
    # samplePoints[i] is the point that sampled from demension i
    samplePoints = torch.empty([d, n], dtype=dtype, device=device)
    for i in range(n):
        low = i * intervalSize
        high = (i + 1) * intervalSize
        samplePoints[:, i] = low + (high - low) * torch.rand(
            d, dtype=dtype, device=device
        )
    for i in range(d):
        samplePoints[i] = samplePoints[i, torch.randperm(n, device=device)]
    return sample_base_of(lower_bound, upper_bound, samplePoints.T)


def rs_np(
    n: int,
    d: int,
    lower_bound: int | float | np.ndarray,
    upper_bound: int | float | np.ndarray,
) -> np.ndarray:
    """random sampling

    Args:
        n: The number of the sample data
        d: The number of the decision variables
        lower_bound: A number or a vector, the lower bound of the decision variables
        upper_bound: A number or a vector, the upper_bound of the decision variables
    """
    if np.any(lower_bound > upper_bound):
        return None
    return sample_base_of(lower_bound, upper_bound, np.random.rand(n, d))


def rs_torch(
    n: int,
    d: int,
    lower_bound: int | float | torch.Tensor,
    upper_bound: int | float | torch.Tensor,
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if isinstance(lower_bound, torch.Tensor) or isinstance(upper_bound, torch.Tensor):
        if torch.any(lower_bound > upper_bound):
            return None
    elif lower_bound > upper_bound:
        return None
    return sample_base_of(
        lower_bound, upper_bound, torch.rand([n, d], dtype=dtype, device=device)
    )
