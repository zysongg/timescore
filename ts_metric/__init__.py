"""ts_metric: Time series metric computation library.

Supports three tasks: prediction, imputation, generation.
Each task has point (regression) and probabilistic (distribution) metrics.

Quick start:
    import ts_metric as tm

    # Functional API
    mse_val = tm.prediction.mse(target, forecast)
    crps_val = tm.prediction.crps(target, samples)

    # Aggregator API
    calc = tm.MetricCalculator(task="prediction", mode="point")
    results = calc.compute(target, forecast)
"""

__version__ = "0.1.0"

from . import metrics, utils
from .metrics import prediction, imputation, generation
from .calculator import MetricCalculator, list_available_metrics

__all__ = [
    "prediction",
    "imputation",
    "generation",
    "MetricCalculator",
    "list_available_metrics",
]
