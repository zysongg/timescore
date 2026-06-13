"""Imputation probabilistic metrics: CRPS, PICP, QICE, interval_width.

Input shapes:
  target:  (B, C, T) or (C, T)
  samples: (B, S, C, T) or (S, C, T)
  mask:    optional, broadcastable to (B, C, T). 1=valid, 0=masked.
"""

import torch
from ...utils import _prepare_prob, masked_mean, compute_quantiles_torch, DEFAULT_QUANTILE_LEVELS


def crps(target, samples, mask=None):
    """Continuous Ranked Probability Score (via quantile loss approximation)."""
    t, s, m = _prepare_prob(target, samples, mask)
    quantiles = compute_quantiles_torch(s, DEFAULT_QUANTILE_LEVELS)

    total_loss = torch.tensor(0.0, device=t.device, dtype=t.dtype)
    n_quantiles = 0
    for q, q_pred in quantiles.items():
        diff = t - q_pred
        loss = torch.max(q * diff, (q - 1.0) * diff)
        total_loss = total_loss + masked_mean(loss, m)
        n_quantiles += 1

    return 2.0 * total_loss / max(n_quantiles, 1)


def picp(target, samples, mask=None, alpha=0.1):
    """Prediction Interval Coverage Probability (90% by default)."""
    t, s, m = _prepare_prob(target, samples, mask)
    q_low = torch.quantile(s, alpha / 2, dim=1)
    q_high = torch.quantile(s, 1 - alpha / 2, dim=1)
    covered = ((t >= q_low) & (t <= q_high)).float()
    return masked_mean(covered, m)


def qice(target, samples, mask=None):
    """Quantile Interval Coverage Error."""
    t, s, m = _prepare_prob(target, samples, mask)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    errors = []
    for alpha in levels:
        q_low = torch.quantile(s, alpha / 2, dim=1)
        q_high = torch.quantile(s, 1 - alpha / 2, dim=1)
        covered = ((t >= q_low) & (t <= q_high)).float()
        observed = masked_mean(covered, m)
        errors.append(torch.abs(observed - (1 - alpha)))
    return torch.mean(torch.stack(errors))


def interval_width(target, samples, mask=None, alpha=0.1):
    """Average prediction interval width.

    alpha: significance level (e.g., 0.1 for 90% interval).
    """
    t, s, m = _prepare_prob(target, samples, mask)
    q_low = torch.quantile(s, alpha / 2, dim=1)
    q_high = torch.quantile(s, 1 - alpha / 2, dim=1)
    width = q_high - q_low
    return masked_mean(width, m)


PROB_METRICS = ["CRPS", "PICP", "QICE", "IntervalWidth"]

PROB_METRIC_FUNCS = {
    "CRPS": crps,
    "PICP": picp,
    "QICE": qice,
    "IntervalWidth": interval_width,
}
