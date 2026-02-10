"""Tests for WLS (Weighted Least Squares) regression."""

import pytest
import linreg_core


class TestWLSNative:
    """Tests for native Python type API for WLS regression."""

    def test_wls_regression_equal_weights_matches_ols(self):
        """Test WLS regression with equal weights (should match OLS)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Equal weights = OLS

        result = linreg_core.wls_regression(y, x, weights)

        # For perfect linear y = x, intercept should be ~0, slope ~1
        assert abs(result.coefficients[0]) < 1e-10, f"Intercept should be ~0, got {result.coefficients[0]}"
        assert abs(result.coefficients[1] - 1.0) < 1e-10, f"Slope should be ~1, got {result.coefficients[1]}"
        assert result.n_observations == 5
        assert result.n_predictors == 1

    def test_wls_result_object_attributes(self):
        """Test that all WlsResult attributes are accessible."""
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = linreg_core.wls_regression(y, x, weights)

        # Test numeric attributes
        assert isinstance(result.r_squared, float)
        assert isinstance(result.r_squared_adjusted, float)
        assert isinstance(result.f_statistic, float)
        assert isinstance(result.f_p_value, float)
        assert isinstance(result.residual_std_error, float)
        assert isinstance(result.mse, float)
        assert isinstance(result.rmse, float)
        assert isinstance(result.mae, float)

        # Test list attributes
        assert isinstance(result.coefficients, list)
        assert isinstance(result.standard_errors, list)
        assert isinstance(result.t_statistics, list)
        assert isinstance(result.p_values, list)
        assert isinstance(result.fitted_values, list)
        assert isinstance(result.residuals, list)

        # Test int attributes
        assert isinstance(result.n_observations, int)
        assert isinstance(result.n_predictors, int)
        assert isinstance(result.df_residuals, int)
        assert isinstance(result.df_model, int)

    def test_wls_with_weighted_outlier(self):
        """Test WLS with low weight on an outlier."""
        # Create data where one point is an outlier
        y = [2.0, 4.0, 6.0, 8.0, 100.0]  # Last point is outlier
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        # With low weight on the outlier, the fit should ignore it
        weights_low = [1.0, 1.0, 1.0, 1.0, 0.01]
        result_low = linreg_core.wls_regression(y, x, weights_low)

        # With high weight on the outlier, the fit should be pulled toward it
        weights_high = [1.0, 1.0, 1.0, 1.0, 10.0]
        result_high = linreg_core.wls_regression(y, x, weights_high)

        # The low-weight fit should have a smaller slope (closer to 2 from first 4 points)
        # The high-weight fit should have a much larger slope (pulled toward outlier)
        assert result_low.coefficients[1] < result_high.coefficients[1]

    def test_wls_negative_weight_error(self):
        """Test that negative weights raise an error."""
        y = [1.0, 2.0, 3.0]
        x = [[1.0, 2.0, 3.0]]
        weights = [1.0, -1.0, 1.0]  # Negative weight

        with pytest.raises(Exception) as exc:
            linreg_core.wls_regression(y, x, weights)

        assert "negative" in str(exc.value).lower() or "invalid" in str(exc.value).lower()

    def test_wls_mismatched_dimensions_error(self):
        """Test that mismatched dimensions raise an error."""
        y = [1.0, 2.0, 3.0]
        x = [[1.0, 2.0]]  # Only 2 elements
        weights = [1.0, 1.0, 1.0]

        with pytest.raises(Exception):
            linreg_core.wls_regression(y, x, weights)

    def test_wls_multiple_predictors(self):
        """Test WLS with multiple predictor variables."""
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [0.5, 1.0, 1.5, 2.0, 2.5]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = linreg_core.wls_regression(y, [x1, x2], weights)

        assert result.n_predictors == 2
        assert len(result.coefficients) == 3  # Intercept + 2 slopes
        assert len(result.fitted_values) == 5
        assert len(result.standard_errors) == 3
        assert len(result.t_statistics) == 3
        assert len(result.p_values) == 3

    def test_wls_insufficient_data_error(self):
        """Test that insufficient data raises an error."""
        y = [1.0, 2.0]
        x1 = [1.0, 2.0]
        x2 = [0.5, 1.0]
        weights = [1.0, 1.0]

        # n=2, k=2, need k+2=4 observations
        with pytest.raises(Exception):
            linreg_core.wls_regression(y, [x1, x2], weights)

    def test_wls_statistics_completeness(self):
        """Verify all statistics are computed correctly."""
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = linreg_core.wls_regression(y, x, weights)

        # Check all fields are populated
        assert len(result.coefficients) == 2
        assert len(result.standard_errors) == 2
        assert len(result.t_statistics) == 2
        assert len(result.p_values) == 2
        assert 0.0 <= result.r_squared <= 1.0
        assert 0.0 <= result.r_squared_adjusted <= 1.0
        assert result.f_statistic >= 0.0
        assert 0.0 <= result.f_p_value <= 1.0
        assert result.residual_std_error >= 0.0
        assert result.df_residuals == 3  # n=5, p=2, df=5-2=3
        assert result.df_model == 1
        assert len(result.fitted_values) == 5
        assert len(result.residuals) == 5
        assert result.mse >= 0.0
        assert result.rmse >= 0.0
        assert result.mae >= 0.0
        assert result.n_observations == 5
        assert result.n_predictors == 1

    def test_wls_summary_method(self):
        """Test the summary() method of WlsResult."""
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = linreg_core.wls_regression(y, x, weights)
        summary = result.summary()

        # Verify summary is a string with expected content
        assert isinstance(summary, str)
        assert "WLS" in summary
        assert "R-squared" in summary
        assert "F-statistic" in summary
        assert "Coefficients" in summary

    def test_wls_get_fitted_values(self):
        """Test get_fitted_values() method."""
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = linreg_core.wls_regression(y, x, weights)
        fitted = result.get_fitted_values()

        assert isinstance(fitted, list)
        assert len(fitted) == 5
        # Fitted values should be close to actual values for this perfect linear relationship
        for i, (actual, pred) in enumerate(zip(y, fitted)):
            assert abs(actual - pred) < 1e-10, f"Fitted value at {i} should be close to actual"

    def test_wls_get_residuals(self):
        """Test get_residuals() method."""
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = linreg_core.wls_regression(y, x, weights)
        residuals = result.get_residuals()

        assert isinstance(residuals, list)
        assert len(residuals) == 5
        # Residuals should be near zero for this perfect linear relationship
        for r in residuals:
            assert abs(r) < 1e-10, f"Residual should be near zero, got {r}"

    def test_wls_to_dict_method(self):
        """Test the to_dict() method of WlsResult."""
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = linreg_core.wls_regression(y, x, weights)
        d = result.to_dict()

        # Verify to_dict returns a proper dict
        assert isinstance(d, dict)
        assert 'coefficients' in d
        assert 'r_squared' in d
        assert 'f_statistic' in d
        assert 'n_observations' in d
