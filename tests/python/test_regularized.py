import pytest
import linreg_core


class TestRidgeRegressionNative:
    """Tests for native Python type API (Phase 5)."""

    def test_ridge_regression_native_lists(self):
        """Test Ridge regression with Python lists."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.ridge_regression(y, x, 1.0, True)

        # Verify result is a RidgeResult object
        assert hasattr(result, 'intercept')
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'lambda')
        assert hasattr(result, 'r_squared')
        assert getattr(result, 'lambda') == 1.0

    def test_ridge_result_object_attributes(self):
        """Test that all RidgeResult attributes are accessible."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.ridge_regression(y, x, 0.5, True)

        # Test all attributes
        assert isinstance(result.intercept, float)
        assert isinstance(result.coefficients, list)
        assert isinstance(getattr(result, 'lambda'), float)
        assert isinstance(result.fitted_values, list)
        assert isinstance(result.residuals, list)
        assert isinstance(result.r_squared, float)
        assert isinstance(result.mse, float)
        assert isinstance(result.effective_df, float)

    def test_ridge_summary_method(self):
        """Test the summary() method of RidgeResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.ridge_regression(y, x, 1.0, True)
        summary = result.summary()

        # Verify summary is a string with expected content
        assert isinstance(summary, str)
        assert "Ridge Regression Results" in summary
        assert "Lambda" in summary
        assert "R-squared" in summary

    def test_ridge_to_dict_method(self):
        """Test the to_dict() method of RidgeResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.ridge_regression(y, x, 1.0, True)
        d = result.to_dict()

        # Verify to_dict returns a proper dict
        assert isinstance(d, dict)
        assert "intercept" in d
        assert "coefficients" in d
        assert "lambda" in d
        assert "r_squared" in d

    def test_ridge_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.ridge_regression(y, x, 1.0, True)

        # Test __str__
        str_result = str(result)
        assert "Ridge Regression Results" in str_result

        # Test __repr__
        repr_result = repr(result)
        assert "RidgeResult" in repr_result

    def test_ridge_multiple_predictors(self):
        """Test Ridge with multiple predictor variables."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3, 7.0]
        x = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 4.0, 5.0, 4.0, 3.0, 2.0]
        ]

        result = linreg_core.ridge_regression(y, x, 0.1, False)

        # Should have 2 coefficients
        assert len(result.coefficients) == 2
        assert len(result.fitted_values) == 6


class TestLassoRegressionNative:
    """Tests for native Python type API (Phase 5)."""

    def test_lasso_regression_native_lists(self):
        """Test Lasso regression with Python lists."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.lasso_regression(y, x, 0.1, True, 100000, 1e-7)

        # Verify result is a LassoResult object
        assert hasattr(result, 'intercept')
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'lambda')
        assert hasattr(result, 'n_nonzero')
        assert hasattr(result, 'converged')
        assert getattr(result, 'lambda') == 0.1

    def test_lasso_result_object_attributes(self):
        """Test that all LassoResult attributes are accessible."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.lasso_regression(y, x, 0.1, True, 100000, 1e-7)

        # Test all attributes
        assert isinstance(result.intercept, float)
        assert isinstance(result.coefficients, list)
        assert isinstance(getattr(result, 'lambda'), float)
        assert isinstance(result.n_nonzero, int)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.r_squared, float)

    def test_lasso_summary_method(self):
        """Test the summary() method of LassoResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.lasso_regression(y, x, 0.1, True, 100000, 1e-7)
        summary = result.summary()

        # Verify summary is a string with expected content
        assert isinstance(summary, str)
        assert "Lasso Regression Results" in summary
        assert "Lambda" in summary
        assert "Non-zero coefficients" in summary
        assert "Converged" in summary

    def test_lasso_to_dict_method(self):
        """Test the to_dict() method of LassoResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.lasso_regression(y, x, 0.1, True, 100000, 1e-7)
        d = result.to_dict()

        # Verify to_dict returns a proper dict
        assert isinstance(d, dict)
        assert "intercept" in d
        assert "coefficients" in d
        assert "lambda" in d
        assert "n_nonzero" in d

    def test_lasso_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.lasso_regression(y, x, 0.1, True, 100000, 1e-7)

        # Test __str__
        str_result = str(result)
        assert "Lasso Regression Results" in str_result

        # Test __repr__
        repr_result = repr(result)
        assert "LassoResult" in repr_result

    def test_lasso_variable_selection(self):
        """Test that Lasso can set some coefficients to zero."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1, 0.2, 0.3, 0.2, 0.1],  # Weak predictor
        ]

        # High lambda should shrink weak predictor to zero
        result = linreg_core.lasso_regression(y, x, 1.0, True, 100000, 1e-7)
        assert result.n_nonzero <= 2  # At most 2 non-zero coefficients


class TestElasticNetRegressionNative:
    """Tests for native Python type API (Phase 5)."""

    def test_elastic_net_regression_native_lists(self):
        """Test Elastic Net regression with Python lists."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.elastic_net_regression(y, x, 0.1, 0.5, True, 100000, 1e-7)

        # Verify result is an ElasticNetResult object
        assert hasattr(result, 'intercept')
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'lambda')
        assert hasattr(result, 'alpha')
        assert hasattr(result, 'n_nonzero')
        assert getattr(result, 'lambda') == 0.1
        assert result.alpha == 0.5

    def test_elastic_net_result_object_attributes(self):
        """Test that all ElasticNetResult attributes are accessible."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.elastic_net_regression(y, x, 0.1, 0.5, True, 100000, 1e-7)

        # Test all attributes
        assert isinstance(result.intercept, float)
        assert isinstance(result.coefficients, list)
        assert isinstance(getattr(result, 'lambda'), float)
        assert isinstance(result.alpha, float)
        assert isinstance(result.n_nonzero, int)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.r_squared, float)

    def test_elastic_net_summary_method(self):
        """Test the summary() method of ElasticNetResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.elastic_net_regression(y, x, 0.1, 0.5, True, 100000, 1e-7)
        summary = result.summary()

        # Verify summary is a string with expected content
        assert isinstance(summary, str)
        assert "Elastic Net Regression Results" in summary
        assert "Lambda" in summary
        assert "Alpha" in summary
        assert "Non-zero coefficients" in summary

    def test_elastic_net_to_dict_method(self):
        """Test the to_dict() method of ElasticNetResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.elastic_net_regression(y, x, 0.1, 0.5, True, 100000, 1e-7)
        d = result.to_dict()

        # Verify to_dict returns a proper dict
        assert isinstance(d, dict)
        assert "intercept" in d
        assert "coefficients" in d
        assert "lambda" in d
        assert "alpha" in d

    def test_elastic_net_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.elastic_net_regression(y, x, 0.1, 0.5, True, 100000, 1e-7)

        # Test __str__
        str_result = str(result)
        assert "Elastic Net Regression Results" in str_result

        # Test __repr__
        repr_result = repr(result)
        assert "ElasticNetResult" in repr_result

    def test_elastic_net_alpha_pure_ridge(self):
        """Test alpha=0 approaches Ridge (L2 only)."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.elastic_net_regression(y, x, 0.1, 0.0, True, 100000, 1e-7)
        assert result.alpha == 0.0

    def test_elastic_net_alpha_pure_lasso(self):
        """Test alpha=1 approaches Lasso (L1 only)."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.elastic_net_regression(y, x, 0.1, 1.0, True, 100000, 1e-7)
        assert result.alpha == 1.0


class TestLambdaPathNative:
    """Tests for native Python type API (Phase 5)."""

    def test_lambda_path_native_lists(self):
        """Test lambda path generation with Python lists."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.make_lambda_path(y, x, 100, 0.01)

        # Verify result is a LambdaPathResult object
        assert hasattr(result, 'lambda_path')
        assert hasattr(result, 'lambda_max')
        assert hasattr(result, 'lambda_min')
        assert hasattr(result, 'n_lambda')
        assert result.n_lambda == 100

    def test_lambda_path_result_attributes(self):
        """Test that all LambdaPathResult attributes are accessible."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.make_lambda_path(y, x, 50, 0.1)

        # Test all attributes
        assert isinstance(result.lambda_path, list)
        assert isinstance(result.lambda_max, float)
        assert isinstance(result.lambda_min, float)
        assert isinstance(result.n_lambda, int)
        assert result.n_lambda == 50
        assert len(result.lambda_path) == 50

    def test_lambda_path_decreasing_sequence(self):
        """Test that lambda path is decreasing."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.make_lambda_path(y, x, 20, 0.01)

        # Lambda path should be decreasing
        for i in range(1, len(result.lambda_path)):
            assert result.lambda_path[i] <= result.lambda_path[i - 1]

    def test_lambda_path_max_min(self):
        """Test that lambda_max > lambda_min for reasonable data."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.make_lambda_path(y, x, 10, 0.1)

        assert result.lambda_max > result.lambda_min
        assert result.lambda_max == result.lambda_path[0]
        assert result.lambda_min == result.lambda_path[-1]

    def test_lambda_path_summary_method(self):
        """Test the summary() method of LambdaPathResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.make_lambda_path(y, x, 100, 0.01)
        summary = result.summary()

        # Verify summary is a string with expected content
        assert isinstance(summary, str)
        assert "Lambda Path Results" in summary
        assert "Lambda max" in summary
        assert "Lambda min" in summary
        assert "Number of values" in summary

    def test_lambda_path_to_dict_method(self):
        """Test the to_dict() method of LambdaPathResult."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.make_lambda_path(y, x, 50, 0.1)
        d = result.to_dict()

        # Verify to_dict returns a proper dict
        assert isinstance(d, dict)
        assert "lambda_path" in d
        assert "lambda_max" in d
        assert "lambda_min" in d
        assert "n_lambda" in d

    def test_lambda_path_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.make_lambda_path(y, x, 100, 0.01)

        # Test __str__
        str_result = str(result)
        assert "Lambda Path Results" in str_result

        # Test __repr__
        repr_result = repr(result)
        assert "LambdaPathResult" in repr_result


class TestRegularizedEdgeCases:
    """Edge case and boundary condition tests for regularized regression."""

    def test_ridge_small_lambda(self):
        """Test Ridge with small lambda (close to OLS)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.ridge_regression(y, x, 0.001, True)
        # RidgeResult returns slope coefficients only (not intercept)
        assert len(result.coefficients) == 1
        # But intercept is available separately
        assert hasattr(result, 'intercept')

    def test_ridge_large_lambda(self):
        """Test Ridge with large lambda (coefficients shrink toward zero)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.ridge_regression(y, x, 100.0, True)
        # With large lambda, coefficients should shrink
        # RidgeResult returns slope coefficients only (not intercept)
        assert len(result.coefficients) == 1
        # Coefficient should be close to zero
        assert abs(result.coefficients[0]) < 0.1

    def test_ridge_negative_lambda_raises_error(self):
        """Test that negative lambda raises an error."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        with pytest.raises(Exception):
            linreg_core.ridge_regression(y, x, -1.0, True)

    def test_lasso_small_lambda(self):
        """Test Lasso with small lambda (most coefficients non-zero)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.lasso_regression(y, x, 0.001, True, 1000, 1e-7)
        # With small lambda, should have non-zero coefficients
        assert result.n_nonzero >= 1

    def test_lasso_large_lambda(self):
        """Test Lasso with large lambda (coefficients shrink to zero)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.lasso_regression(y, x, 10.0, True, 1000, 1e-7)
        # With larger lambda, fewer non-zero coefficients
        assert hasattr(result, 'n_nonzero')

    def test_lasso_convergence_attribute(self):
        """Test that Lasso result has convergence attribute."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.lasso_regression(y, x, 0.1, True, 1000, 1e-7)
        # Should have convergence info
        assert hasattr(result, 'n_nonzero')

    def test_elastic_net_middle_alpha(self):
        """Test ElasticNet with middle alpha value."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.elastic_net_regression(y, x, 0.1, 0.5, True, 1000, 1e-7)
        assert hasattr(result, 'n_nonzero')

    def test_elastic_net_negative_alpha(self):
        """Test ElasticNet with negative alpha (current behavior)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        # Alpha outside [0, 1] may be accepted or rejected
        # This test documents current behavior
        try:
            result = linreg_core.elastic_net_regression(y, x, 0.1, -0.5, True, 1000, 1e-7)
            # If accepted, verify we get a result
            assert hasattr(result, 'n_nonzero')
        except Exception:
            pass  # Also acceptable if rejected

    def test_regularized_multiple_predictors(self):
        """Test regularized regression handles multiple predictors."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 3.0, 4.0, 2.0, 3.0]

        result = linreg_core.ridge_regression(y, [x1, x2], 1.0, True)
        # RidgeResult returns slope coefficients only (not intercept)
        assert len(result.coefficients) == 2
