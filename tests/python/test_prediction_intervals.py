"""Tests for prediction intervals (OLS, Ridge, Lasso, Elastic Net)."""

import pytest
import linreg_core


# Shared fixtures
Y_SIMPLE = [3.1, 4.9, 7.2, 8.8, 11.1]
X_SIMPLE = [[1.0, 2.0, 3.0, 4.0, 5.0]]
NEW_X_SIMPLE = [[6.0]]

Y_MULTI = [3.0, 5.5, 7.0, 9.5, 11.0, 13.5]
X_MULTI = [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [2.0, 4.0, 5.0, 6.0, 8.0, 9.0],
]
NEW_X_MULTI = [[7.0], [10.0]]

Y_REGULARIZED = [3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.0]
X_REGULARIZED = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]
NEW_X_REGULARIZED = [[8.0]]


class TestOLSPredictionIntervals:
    """Tests for OLS prediction intervals."""

    def test_basic_prediction_and_bounds(self):
        result = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE)
        assert len(result.predicted) == 1
        assert result.lower_bound[0] < result.predicted[0]
        assert result.upper_bound[0] > result.predicted[0]
        assert result.se_pred[0] > 0.0

    def test_result_attributes(self):
        result = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE)
        assert isinstance(result.predicted, list)
        assert isinstance(result.lower_bound, list)
        assert isinstance(result.upper_bound, list)
        assert isinstance(result.se_pred, list)
        assert isinstance(result.leverage, list)
        assert isinstance(result.alpha, float)
        assert isinstance(result.df_residuals, float)
        assert abs(result.alpha - 0.05) < 1e-10
        assert result.df_residuals > 0.0

    def test_multiple_new_observations(self):
        new_x = [[6.0, 7.0, 3.0]]
        result = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, new_x)
        assert len(result.predicted) == 3
        assert len(result.lower_bound) == 3
        assert len(result.upper_bound) == 3
        assert len(result.se_pred) == 3
        assert len(result.leverage) == 3
        for i in range(3):
            assert result.lower_bound[i] < result.predicted[i]
            assert result.upper_bound[i] > result.predicted[i]

    def test_multiple_predictors(self):
        result = linreg_core.ols_prediction_intervals(Y_MULTI, X_MULTI, NEW_X_MULTI)
        assert len(result.predicted) == 1
        assert result.lower_bound[0] < result.predicted[0]
        assert result.upper_bound[0] > result.predicted[0]

    def test_higher_confidence_gives_wider_interval(self):
        """99% PI (alpha=0.01) must be wider than 95% PI (alpha=0.05)."""
        r95 = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE, alpha=0.05)
        r99 = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE, alpha=0.01)
        width_95 = r95.upper_bound[0] - r95.lower_bound[0]
        width_99 = r99.upper_bound[0] - r99.lower_bound[0]
        assert width_99 > width_95

    def test_extrapolation_has_higher_leverage_and_wider_pi(self):
        """Points far from training data should have higher leverage and wider PI."""
        r_center = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, [[3.0]])
        r_extrap = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, [[20.0]])
        assert r_extrap.leverage[0] > r_center.leverage[0]
        assert r_extrap.se_pred[0] > r_center.se_pred[0]
        w_center = r_center.upper_bound[0] - r_center.lower_bound[0]
        w_extrap = r_extrap.upper_bound[0] - r_extrap.lower_bound[0]
        assert w_extrap > w_center

    def test_se_pred_at_least_sqrt_mse(self):
        """SE_pred = sqrt(MSE*(1+h)) must be >= sqrt(MSE) since h >= 0."""
        ols = linreg_core.ols_regression(Y_SIMPLE, X_SIMPLE, ["Intercept", "X1"])
        result = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE)
        sqrt_mse = ols.mse ** 0.5
        assert result.se_pred[0] >= sqrt_mse

    def test_invalid_alpha_raises(self):
        with pytest.raises(Exception):
            linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE, alpha=0.0)
        with pytest.raises(Exception):
            linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE, alpha=1.0)
        with pytest.raises(Exception):
            linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE, alpha=-0.1)

    def test_dimension_mismatch_raises(self):
        """new_x with wrong number of predictors should raise."""
        wrong_new_x = [[6.0], [7.0]]  # 2 predictors, model has 1
        with pytest.raises(Exception):
            linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, wrong_new_x)

    def test_summary_method(self):
        result = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Prediction" in summary
        assert "Alpha" in summary

    def test_to_dict_method(self):
        result = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE)
        d = result.to_dict()
        assert isinstance(d, dict)
        for key in ("predicted", "lower_bound", "upper_bound", "se_pred", "leverage", "alpha", "df_residuals"):
            assert key in d

    def test_default_alpha_is_0_05(self):
        result = linreg_core.ols_prediction_intervals(Y_SIMPLE, X_SIMPLE, NEW_X_SIMPLE)
        assert abs(result.alpha - 0.05) < 1e-10


class TestRidgePredictionIntervals:
    """Tests for Ridge regression prediction intervals."""

    def test_basic_prediction_and_bounds(self):
        result = linreg_core.ridge_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
            lambda_val=0.1, standardize=True,
        )
        assert len(result.predicted) == 1
        assert result.lower_bound[0] < result.predicted[0]
        assert result.upper_bound[0] > result.predicted[0]
        assert result.se_pred[0] > 0.0

    def test_predicted_value_reasonable(self):
        """y ≈ 2x + 1, so at x=8 expect ~17."""
        result = linreg_core.ridge_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
            lambda_val=0.01, standardize=True,
        )
        assert abs(result.predicted[0] - 17.0) < 2.0

    def test_higher_confidence_wider_interval(self):
        r95 = linreg_core.ridge_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED, alpha=0.05, lambda_val=0.1,
        )
        r99 = linreg_core.ridge_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED, alpha=0.01, lambda_val=0.1,
        )
        assert (r99.upper_bound[0] - r99.lower_bound[0]) > (r95.upper_bound[0] - r95.lower_bound[0])

    def test_extrapolation_wider(self):
        r_center = linreg_core.ridge_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, [[4.0]], lambda_val=0.1,
        )
        r_extrap = linreg_core.ridge_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, [[20.0]], lambda_val=0.1,
        )
        w_center = r_center.upper_bound[0] - r_center.lower_bound[0]
        w_extrap = r_extrap.upper_bound[0] - r_extrap.lower_bound[0]
        assert w_extrap > w_center

    def test_result_attributes(self):
        result = linreg_core.ridge_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
        )
        for attr in ("predicted", "lower_bound", "upper_bound", "se_pred", "leverage"):
            assert isinstance(getattr(result, attr), list)
        assert isinstance(result.alpha, float)
        assert isinstance(result.df_residuals, float)


class TestLassoPredictionIntervals:
    """Tests for Lasso regression prediction intervals."""

    def test_basic_prediction_and_bounds(self):
        result = linreg_core.lasso_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
            lambda_val=0.01, standardize=True,
        )
        assert len(result.predicted) == 1
        assert result.lower_bound[0] < result.predicted[0]
        assert result.upper_bound[0] > result.predicted[0]
        assert result.se_pred[0] > 0.0

    def test_higher_confidence_wider_interval(self):
        r95 = linreg_core.lasso_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED, alpha=0.05, lambda_val=0.01,
        )
        r99 = linreg_core.lasso_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED, alpha=0.01, lambda_val=0.01,
        )
        assert (r99.upper_bound[0] - r99.lower_bound[0]) > (r95.upper_bound[0] - r95.lower_bound[0])

    def test_result_attributes(self):
        result = linreg_core.lasso_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
        )
        for attr in ("predicted", "lower_bound", "upper_bound", "se_pred", "leverage"):
            assert isinstance(getattr(result, attr), list)
        assert result.df_residuals > 0.0


class TestElasticNetPredictionIntervals:
    """Tests for Elastic Net regression prediction intervals."""

    def test_basic_prediction_and_bounds(self):
        result = linreg_core.elastic_net_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
            lambda_val=0.01, enet_alpha=0.5, standardize=True,
        )
        assert len(result.predicted) == 1
        assert result.lower_bound[0] < result.predicted[0]
        assert result.upper_bound[0] > result.predicted[0]
        assert result.se_pred[0] > 0.0

    def test_alpha_zero_matches_ridge_closely(self):
        """alpha=0 is pure Ridge; results should be close to ridge_prediction_intervals."""
        r_enet = linreg_core.elastic_net_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
            lambda_val=0.1, enet_alpha=0.0,
        )
        r_ridge = linreg_core.ridge_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
            lambda_val=0.1,
        )
        assert abs(r_enet.predicted[0] - r_ridge.predicted[0]) < 0.5

    def test_higher_confidence_wider_interval(self):
        r95 = linreg_core.elastic_net_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED, alpha=0.05, lambda_val=0.01,
        )
        r99 = linreg_core.elastic_net_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED, alpha=0.01, lambda_val=0.01,
        )
        assert (r99.upper_bound[0] - r99.lower_bound[0]) > (r95.upper_bound[0] - r95.lower_bound[0])

    def test_result_attributes(self):
        result = linreg_core.elastic_net_prediction_intervals(
            Y_REGULARIZED, X_REGULARIZED, NEW_X_REGULARIZED,
        )
        for attr in ("predicted", "lower_bound", "upper_bound", "se_pred", "leverage"):
            assert isinstance(getattr(result, attr), list)
        assert result.df_residuals > 0.0
