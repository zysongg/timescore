"""Tests verifying ts_metric against GluonTS for consistency."""

import torch
import numpy as np
import pytest
import ts_metric as tm


class TestGluonTSCompatibility:
    """Verify our metrics match GluonTS implementations."""

    def setup_method(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B, self.C, self.T = 2, 3, 24
        self.S = 50
        self.target = torch.randn(self.B, self.C, self.T)
        self.forecast = self.target + 0.1 * torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, self.S, self.C, self.T)

    def test_quantile_loss_gluonts(self):
        """Verify quantile_loss matches GluonTS formula: 2*sum(|(ŷ-y)*(𝟙[y≤ŷ]-q)|)."""
        from gluonts.evaluation.metrics import quantile_loss as gluonts_ql

        q = 0.5
        qhat = torch.quantile(self.samples, q, dim=1).numpy()
        target_np = self.target.numpy()

        our_ql = tm.prediction.quantile_loss(self.target, self.samples, quantile_levels=[q])
        gluonts_ql_val = gluonts_ql(target_np.flatten(), qhat.flatten(), q)

        assert abs(our_ql[q].item() - gluonts_ql_val) < 1e-3, \
            f"QuantileLoss mismatch: ours={our_ql[q].item():.6f}, gluonts={gluonts_ql_val:.6f}"

    def test_coverage_gluonts(self):
        """Verify coverage matches GluonTS: mean(y ≤ ŷ_q)."""
        from gluonts.evaluation.metrics import coverage as gluonts_cov

        q = 0.5
        qhat = torch.quantile(self.samples, q, dim=1).numpy()
        target_np = self.target.numpy()

        our_cov = tm.prediction.coverage(self.target, self.samples, quantile_levels=[q])
        gluonts_cov_val = gluonts_cov(target_np.flatten(), qhat.flatten())

        assert abs(our_cov[q].item() - gluonts_cov_val) < 1e-3, \
            f"Coverage mismatch: ours={our_cov[q].item():.6f}, gluonts={gluonts_cov_val:.6f}"

    def test_mse_gluonts(self):
        """Verify MSE matches GluonTS."""
        from gluonts.evaluation.metrics import mse as gluonts_mse

        target_np = self.target.numpy()
        forecast_np = self.forecast.numpy()

        our_mse = tm.prediction.mse(self.target, self.forecast).item()
        gluonts_mse_val = gluonts_mse(target_np.flatten(), forecast_np.flatten())

        assert abs(our_mse - gluonts_mse_val) < 1e-5, \
            f"MSE mismatch: ours={our_mse:.6f}, gluonts={gluonts_mse_val:.6f}"

    def test_mape_gluonts(self):
        """Verify MAPE matches GluonTS."""
        from gluonts.evaluation.metrics import mape as gluonts_mape

        target_np = self.target.numpy()
        forecast_np = self.forecast.numpy()

        our_mape = tm.prediction.mape(self.target, self.forecast).item()
        gluonts_mape_val = gluonts_mape(target_np.flatten(), forecast_np.flatten())

        assert abs(our_mape - gluonts_mape_val) < 1e-3, \
            f"MAPE mismatch: ours={our_mape:.6f}, gluonts={gluonts_mape_val:.6f}"

    def test_smape_gluonts(self):
        """Verify sMAPE matches GluonTS."""
        from gluonts.evaluation.metrics import smape as gluonts_smape

        target_np = self.target.numpy()
        forecast_np = self.forecast.numpy()

        our_smape = tm.prediction.smape(self.target, self.forecast).item()
        gluonts_smape_val = gluonts_smape(target_np.flatten(), forecast_np.flatten())

        assert abs(our_smape - gluonts_smape_val) < 1e-3, \
            f"sMAPE mismatch: ours={our_smape:.6f}, gluonts={gluonts_smape_val:.6f}"

    def test_msis_gluonts(self):
        """Verify MSIS matches GluonTS formula."""
        from gluonts.evaluation.metrics import msis as gluonts_msis

        alpha = 0.05
        lower = torch.quantile(self.samples, alpha / 2, dim=1).numpy()
        upper = torch.quantile(self.samples, 1 - alpha / 2, dim=1).numpy()
        target_np = self.target.numpy()

        seasonal_error = np.mean(np.abs(target_np[:, :, 1:] - target_np[:, :, :-1]))

        our_msis = tm.prediction.msis(
            self.target, self.samples, alpha=alpha, seasonal_error=seasonal_error
        ).item()
        gluonts_msis_val = gluonts_msis(
            target_np.flatten(), lower.flatten(), upper.flatten(),
            seasonal_error, alpha
        )

        assert abs(our_msis - gluonts_msis_val) / max(abs(gluonts_msis_val), 1e-6) < 0.01, \
            f"MSIS mismatch: ours={our_msis:.6f}, gluonts={gluonts_msis_val:.6f}"


class TestNewMetrics:
    """Test newly added GluonTS-style metrics."""

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 2, 3, 24
        self.S = 50
        self.target = torch.randn(self.B, self.C, self.T)
        self.forecast = self.target + 0.1 * torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, self.S, self.C, self.T)

    def test_nrmse(self):
        val = tm.prediction.nrmse(self.target, self.forecast)
        assert val.shape == ()
        assert val > 0
        rmse_val = tm.prediction.rmse(self.target, self.forecast)
        expected = rmse_val / torch.abs(self.target).mean()
        assert torch.allclose(val, expected, atol=1e-5)

    def test_quantile_loss_all_levels(self):
        result = tm.prediction.quantile_loss(self.target, self.samples)
        assert len(result) == 9
        for q, val in result.items():
            assert val.shape == ()
            assert val >= 0

    def test_w_quantile_loss(self):
        result = tm.prediction.w_quantile_loss(self.target, self.samples)
        assert len(result) == 9
        for q, val in result.items():
            assert val.shape == ()
            assert val >= 0

    def test_mean_w_quantile_loss(self):
        val = tm.prediction.mean_w_quantile_loss(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_mean_absolute_quantile_loss(self):
        val = tm.prediction.mean_absolute_quantile_loss(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_coverage_all_levels(self):
        result = tm.prediction.coverage(self.target, self.samples)
        assert len(result) == 9
        for q, val in result.items():
            assert 0 <= val <= 1

    def test_mae_coverage(self):
        val = tm.prediction.mae_coverage(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_msis(self):
        val = tm.prediction.msis(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_msis_custom_seasonal_error(self):
        val = tm.prediction.msis(self.target, self.samples, seasonal_error=1.0)
        assert val.shape == ()

    def test_calculator_new_metrics(self):
        calc = tm.MetricCalculator(
            task="prediction", mode="probabilistic",
            metrics=["mean_wQuantileLoss", "MAE_Coverage", "MSIS"]
        )
        results = calc.compute(self.target, self.samples)
        assert "mean_wQuantileLoss" in results
        assert "MAE_Coverage" in results
        assert "MSIS" in results
