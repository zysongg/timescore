"""Imputation point metrics: MSE, MAE, RMSE, MAPE, MRE, sMAPE, ND.

Input shapes:
  target:   (B, C, T) or (C, T)
  forecast: (B, C, T) or (C, T)  -- imputed values
  mask:     optional, broadcastable to (B, C, T). 1=valid, 0=masked.
            For imputation, typically mask=1 at observed positions where
            ground truth is available for evaluation.
"""

import torch
from ...utils import _prepare_point, masked_mean, masked_sum


def mse(target, forecast, mask=None):
    """Mean Squared Error."""
    t, f, m = _prepare_point(target, forecast, mask)
    return masked_mean((t - f) ** 2, m)


def mae(target, forecast, mask=None):
    """Mean Absolute Error."""
    t, f, m = _prepare_point(target, forecast, mask)
    return masked_mean(torch.abs(t - f), m)


def rmse(target, forecast, mask=None):
    """Root Mean Squared Error."""
    return torch.sqrt(mse(target, forecast, mask))


def mape(target, forecast, mask=None):
    """Mean Absolute Percentage Error."""
    t, f, m = _prepare_point(target, forecast, mask)
    denom = torch.abs(t)
    valid = (denom > 1e-8) & (m > 0.5)
    if valid.sum() < 1:
        return torch.tensor(float('inf'), device=t.device)
    return torch.mean(torch.abs(t - f)[valid] / denom[valid])


def mre(target, forecast, mask=None):
    """Mean Relative Error: mean(|t-f|) / mean(|t|)."""
    t, f, m = _prepare_point(target, forecast, mask)
    denom = masked_mean(torch.abs(t), m)
    if denom < 1e-8:
        return torch.tensor(float('inf'), device=t.device)
    return masked_mean(torch.abs(t - f), m) / denom


def smape(target, forecast, mask=None):
    """Symmetric Mean Absolute Percentage Error."""
    t, f, m = _prepare_point(target, forecast, mask)
    denom = torch.abs(t) + torch.abs(f)
    valid = (denom > 1e-8) & (m > 0.5)
    if valid.sum() < 1:
        return torch.tensor(0.0, device=t.device)
    return torch.mean(2.0 * torch.abs(t - f)[valid] / denom[valid])


def nd(target, forecast, mask=None):
    """Normalized Deviation: sum(|t-f|) / sum(|t|)."""
    t, f, m = _prepare_point(target, forecast, mask)
    denom = masked_sum(torch.abs(t), m)
    if denom < 1e-8:
        return torch.tensor(float('inf'), device=t.device)
    return masked_sum(torch.abs(t - f), m) / denom


POINT_METRICS = ["MSE", "RMSE", "MAE", "MAPE", "MRE", "sMAPE", "ND"]

POINT_METRIC_FUNCS = {
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "MAPE": mape,
    "MRE": mre,
    "sMAPE": smape,
    "ND": nd,
}
