import pytest
import linreg_core


class TestOLSNative:
    """Tests for native Python type API (Phase 4)."""

    def test_ols_regression_native_lists(self):
        """Test OLS regression with Python lists."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        names = ["Intercept", "X1"]

        result = linreg_core.ols_regression(y, x, names)

        # Verify result is an OLSResult object
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'standard_errors')
        assert hasattr(result, 't_statistics')
        assert hasattr(result, 'p_values')
        assert len(result.coefficients) == 2

    def test_ols_result_object_attributes(self):
        """Test that all OLSResult attributes are accessible."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        names = ["Intercept", "X1"]

        result = linreg_core.ols_regression(y, x, names)

        # Test numeric attributes
        assert isinstance(result.r_squared, float)
        assert isinstance(result.r_squared_adjusted, float)
        assert isinstance(result.f_statistic, float)
        assert isinstance(result.f_p_value, float)
        assert isinstance(result.mse, float)
        assert isinstance(result.rmse, float)

        # Test list attributes
        assert isinstance(result.coefficients, list)
        assert isinstance(result.standard_errors, list)
        assert isinstance(result.t_statistics, list)
        assert isinstance(result.p_values, list)
        assert isinstance(result.residuals, list)
        assert isinstance(result.standardized_residuals, list)
        assert isinstance(result.leverage, list)
        assert isinstance(result.vif, list)

        # Test int attributes
        assert isinstance(result.n_observations, int)
        assert isinstance(result.n_predictors, int)
        assert isinstance(result.degrees_of_freedom, int)

    def test_ols_summary_method(self):
        """Test the summary() method of OLSResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        names = ["Intercept", "X1"]

        result = linreg_core.ols_regression(y, x, names)
        summary = result.summary()

        # Verify summary is a string with expected content
        assert isinstance(summary, str)
        assert "OLS Regression Results" in summary
        assert "R-squared" in summary
        assert "F-statistic" in summary
        assert "Observations" in summary

    def test_ols_to_dict_method(self):
        """Test the to_dict() method of OLSResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        names = ["Intercept", "X1"]

        result = linreg_core.ols_regression(y, x, names)
        d = result.to_dict()

        # Verify to_dict returns a proper dict
        assert isinstance(d, dict)
        assert "coefficients" in d
        assert "standard_errors" in d
        assert "r_squared" in d
        assert "mse" in d
        assert "rmse" in d

    def test_ols_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        names = ["Intercept", "X1"]

        result = linreg_core.ols_regression(y, x, names)

        # Test __str__
        str_result = str(result)
        assert "OLS Regression Results" in str_result

        # Test __repr__
        repr_result = repr(result)
        assert "OLSResult" in repr_result

    def test_ols_housing_regression_accuracy(self):
        """Verify housing regression accuracy matches expected values."""
        y = [245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1]
        x = [
            [1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0],
            [3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0]
        ]
        names = ["Intercept", "Square_Feet", "Bedrooms"]

        result = linreg_core.ols_regression(y, x, names)

        # Expected coefficients (verified via numpy.linalg.inv)
        expected_coeffs = [15.6480854, 0.1638012, 4.8496809]
        tolerance = 1e-5

        for i, (actual, expected) in enumerate(zip(result.coefficients, expected_coeffs)):
            assert abs(actual - expected) < tolerance, f"coeff[{i}] mismatch: {actual} vs {expected}"

    def test_ols_multiple_predictors(self):
        """Test OLS with multiple predictor variables."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3, 7.0]
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0, 2.0]
        names = ["Intercept", "X1", "X2"]

        result = linreg_core.ols_regression(y, [x1, x2], names)

        # Should have 3 coefficients (intercept + 2 predictors)
        assert len(result.coefficients) == 3
        assert result.n_observations == 6
        assert result.n_predictors == 2

    def test_ols_variable_names_as_list(self):
        """Test that variable names can be passed as a list."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        # Pass names as list
        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2


class TestOLSEdgeCases:
    """Edge case and error handling tests for OLS regression."""

    def test_empty_input_raises_error(self):
        """Test that empty input raises an error."""
        with pytest.raises(Exception):
            linreg_core.ols_regression([], [], ["Intercept"])

    def test_dimension_mismatch_raises_error(self):
        """Test that mismatched dimensions raise an error."""
        y = [1.0, 2.0, 3.0]
        x = [[1.0, 2.0]]  # Only 2 observations instead of 3

        with pytest.raises(Exception):
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data (n <= p + 1) raises an error."""
        y = [1.0, 2.0]
        x = [[1.0, 2.0], [2.0, 4.0]]  # 2 obs, 2 predictors (need n > p + 1)

        with pytest.raises(Exception):
            linreg_core.ols_regression(y, x, ["Intercept", "X1", "X2"])

    def test_single_observation_raises_error(self):
        """Test that a single observation raises an error (can't compute variance)."""
        y = [5.0]
        x = [[2.0]]

        with pytest.raises(Exception):
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

    def test_nan_in_input(self):
        """Test handling of NaN values in input."""
        y = [1.0, float('nan'), 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        # Should either raise an error or handle NaN gracefully
        with pytest.raises(Exception):
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

    def test_inf_in_input(self):
        """Test handling of infinite values in input."""
        y = [1.0, float('inf'), 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        with pytest.raises(Exception):
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

    def test_very_large_values(self):
        """Test that very large numeric values work correctly."""
        y = [1e10, 2e10, 3e10, 4e10, 5e10]
        x = [[1e10, 2e10, 3e10, 4e10, 5e10]]

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2
        # R-squared should be near 1 for perfect linear relationship
        assert result.r_squared > 0.99

    def test_very_small_values(self):
        """Test that very small numeric values work correctly."""
        y = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        x = [[1e-10, 2e-10, 3e-10, 4e-10, 5e-10]]

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2

    def test_constant_predictor_handled_gracefully(self):
        """Test that a constant predictor (zero variance) is handled gracefully."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 1.0, 1.0, 1.0, 1.0]]  # Constant predictor (collinear with intercept)

        # LINPACK QR handles rank-deficiency by dropping redundant columns
        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert result is not None
        assert len(result.coefficients) == 2
        # At least one coefficient should be NaN (dropped)
        import math
        assert any(math.isnan(c) for c in result.coefficients)

    def test_perfect_collinearity_handled_gracefully(self):
        """Test that perfectly collinear predictors are handled gracefully."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 6.0, 8.0, 10.0]  # Exactly 2 * x1

        # LINPACK QR handles rank-deficiency by dropping redundant columns
        result = linreg_core.ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        assert result is not None
        assert len(result.coefficients) == 3
        # At least one coefficient should be NaN (dropped)
        import math
        assert any(math.isnan(c) for c in result.coefficients)

    def test_negative_values(self):
        """Test that negative values are handled correctly."""
        y = [-5.0, -3.0, -1.0, 1.0, 3.0]
        x = [[-2.0, -1.0, 0.0, 1.0, 2.0]]

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2
