"""Tests for polynomial regression Python bindings."""

import pytest
import numpy as np
import linreg_core


class TestPolynomialOLS:
    """Tests for OLS polynomial regression."""

    def test_polynomial_regression_basic(self, poly_x, poly_y_quadratic):
        """Test basic polynomial regression with degree 2."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2
        )

        assert hasattr(result, 'degree')
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'feature_names')

        assert result.degree == 2
        assert len(result.coefficients) == 3  # intercept + x + x²
        assert result.n_observations == 10
        assert result.r_squared > 0.9  # Should fit well

    def test_polynomial_regression_degree_3(self, poly_x, poly_y_cubic):
        """Test polynomial regression with degree 3."""
        result = linreg_core.polynomial_regression(
            poly_y_cubic, poly_x, degree=3
        )

        assert result.degree == 3
        assert len(result.coefficients) == 4  # intercept + x + x² + x³
        assert result.r_squared > 0.9

    def test_polynomial_regression_with_centering(self, poly_centered_x, poly_centered_y):
        """Test polynomial regression with centering enabled."""
        result = linreg_core.polynomial_regression(
            poly_centered_y, poly_centered_x, degree=2, center=True
        )

        assert result.centered == True
        assert result.x_mean != 0.0
        # The mean should be close to the mean of input x
        expected_mean = np.mean(poly_centered_x)
        assert abs(result.x_mean - expected_mean) < 0.01

    def test_polynomial_regression_with_standardization(self, poly_x, poly_y_quadratic):
        """Test polynomial regression with standardization."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2, standardize=True
        )

        assert result.standardized == True
        assert len(result.feature_means) == 2  # mean for x and x²
        assert len(result.feature_stds) == 2

    def test_polynomial_regression_feature_names(self, poly_x, poly_y_quadratic):
        """Test that feature names are generated correctly."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=3
        )

        expected_names = ["Intercept", "x", "x^2", "x^3"]
        assert result.feature_names == expected_names

    def test_polynomial_regression_no_intercept(self, poly_x, poly_y_quadratic):
        """Test polynomial regression without intercept."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2, intercept=False
        )

        # Without intercept, should still work but with different fit
        assert hasattr(result, 'coefficients')

    def test_polynomial_summary_method(self, poly_x, poly_y_quadratic):
        """Test the summary() method of PolynomialResult."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2
        )
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Polynomial Regression Results" in summary
        assert "degree=2" in summary
        assert "R-squared" in summary

    def test_polynomial_to_dict_method(self, poly_x, poly_y_quadratic):
        """Test the to_dict() method of PolynomialResult."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2
        )
        d = result.to_dict()

        assert isinstance(d, dict)
        assert 'degree' in d
        assert 'coefficients' in d
        assert 'r_squared' in d
        assert 'centered' in d
        assert 'standardized' in d

    def test_polynomial_repr(self, poly_x, poly_y_quadratic):
        """Test the __repr__ method of PolynomialResult."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2
        )
        repr_str = repr(result)

        assert "PolynomialResult" in repr_str
        assert "degree=2" in repr_str

    def test_polynomial_str(self, poly_x, poly_y_quadratic):
        """Test the __str__ method of PolynomialResult."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2
        )
        str_str = str(result)

        assert "Polynomial Regression Results" in str_str


class TestPolynomialPrediction:
    """Tests for polynomial prediction."""

    def test_polynomial_predict_method(self, poly_x, poly_y_quadratic, poly_x_new):
        """Test prediction using the result object's predict method."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2
        )
        predictions = result.predict(poly_x_new)

        assert isinstance(predictions, list)
        assert len(predictions) == len(poly_x_new)
        # Predictions should be increasing for quadratic with positive coeffs
        assert predictions[0] < predictions[1] < predictions[2] < predictions[3]

    def test_polynomial_predict_function(self, poly_x, poly_y_quadratic, poly_x_new):
        """Test prediction using the standalone polynomial_predict function."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2
        )
        predictions = linreg_core.polynomial_predict(result, poly_x_new)

        assert isinstance(predictions, list)
        assert len(predictions) == len(poly_x_new)

    def test_polynomial_predict_single_value(self, poly_small_x, poly_small_y):
        """Test prediction for a single new value."""
        result = linreg_core.polynomial_regression(
            poly_small_y, poly_small_x, degree=2
        )
        predictions = result.predict([6.0])

        assert len(predictions) == 1
        # y = 2 + x + 0.5x², at x=6: 2 + 6 + 18 = 26
        assert abs(predictions[0] - 26.0) < 1.0

    def test_polynomial_predict_with_centering(self, poly_centered_x, poly_centered_y):
        """Test prediction when model was fit with centering."""
        result = linreg_core.polynomial_regression(
            poly_centered_y, poly_centered_x, degree=2, center=True
        )
        # Predict at values within the training range
        predictions = result.predict([105.0, 109.0, 115.0])

        assert len(predictions) == 3
        # All predictions should be reasonable
        assert all(isinstance(p, float) for p in predictions)

    def test_polynomial_predict_with_standardization(self, poly_x, poly_y_quadratic):
        """Test prediction when model was fit with standardization."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=2, standardize=True
        )
        predictions = result.predict([11.0, 12.0])

        assert len(predictions) == 2

    def test_polynomial_predict_perfect_fit(self, poly_x, poly_y_perfect_quadratic):
        """Test prediction on a perfect quadratic relationship (no noise)."""
        result = linreg_core.polynomial_regression(
            poly_y_perfect_quadratic, poly_x, degree=2
        )

        # Should achieve near-perfect fit
        assert result.r_squared > 0.999

        # Predictions should be very accurate
        predictions = result.predict([11.0, 12.0])
        # y = 2 + x + 0.5x²
        # at x=11: 2 + 11 + 0.5*121 = 2 + 11 + 60.5 = 73.5
        # at x=12: 2 + 12 + 0.5*144 = 2 + 12 + 72 = 86
        assert abs(predictions[0] - 73.5) < 0.1
        assert abs(predictions[1] - 86.0) < 0.1


class TestPolynomialRegularized:
    """Tests for regularized polynomial regression."""

    def test_polynomial_ridge_basic(self, poly_x, poly_y_quadratic):
        """Test polynomial ridge regression."""
        result = linreg_core.polynomial_ridge(
            poly_y_quadratic, poly_x, degree=3, lambda_val=0.5
        )

        assert hasattr(result, 'intercept')
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'lambda')
        assert getattr(result, 'lambda') == 0.5

    def test_polynomial_ridge_with_centering(self, poly_centered_x, poly_centered_y):
        """Test polynomial ridge with centering."""
        result = linreg_core.polynomial_ridge(
            poly_centered_y, poly_centered_x, degree=3,
            lambda_val=1.0, center=True, standardize=True
        )

        assert hasattr(result, 'r_squared')

    def test_polynomial_lasso_basic(self, poly_x, poly_y_cubic):
        """Test polynomial lasso regression."""
        result = linreg_core.polynomial_lasso(
            poly_y_cubic, poly_x, degree=5, lambda_val=0.1
        )

        assert hasattr(result, 'intercept')
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'n_nonzero')
        assert hasattr(result, 'converged')
        assert result.converged == True

    def test_polynomial_lasso_variable_selection(self, poly_x, poly_y_quadratic):
        """Test that lasso can zero out higher-order terms."""
        result = linreg_core.polynomial_lasso(
            poly_y_quadratic, poly_x, degree=5, lambda_val=1.0
        )

        # With high lambda, some coefficients should be zero
        n_zero = sum(1 for c in result.coefficients if abs(c) < 1e-10)
        assert n_zero >= 1

    def test_polynomial_elastic_net_basic(self, poly_x, poly_y_cubic):
        """Test polynomial elastic net regression."""
        result = linreg_core.polynomial_elastic_net(
            poly_y_cubic, poly_x, degree=4,
            lambda_val=0.1, alpha=0.5
        )

        assert hasattr(result, 'intercept')
        assert hasattr(result, 'coefficients')
        assert hasattr(result, 'lambda')
        assert hasattr(result, 'alpha')
        assert result.alpha == 0.5

    def test_polynomial_elastic_net_ridge_like(self, poly_x, poly_y_quadratic):
        """Test elastic net with alpha=0 (pure ridge)."""
        result = linreg_core.polynomial_elastic_net(
            poly_y_quadratic, poly_x, degree=3,
            lambda_val=0.5, alpha=0.0
        )

        assert result.alpha == 0.0

    def test_polynomial_elastic_net_lasso_like(self, poly_x, poly_y_quadratic):
        """Test elastic net with alpha=1 (pure lasso)."""
        result = linreg_core.polynomial_elastic_net(
            poly_y_quadratic, poly_x, degree=3,
            lambda_val=0.1, alpha=1.0
        )

        assert result.alpha == 1.0


class TestPolynomialEdgeCases:
    """Tests for edge cases and error handling."""

    def test_polynomial_degree_1_linear(self, poly_x, poly_y_quadratic):
        """Test degree 1 (should be same as linear regression)."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x, degree=1
        )

        assert result.degree == 1
        assert len(result.coefficients) == 2

    def test_polynomial_small_dataset(self, poly_small_x, poly_small_y):
        """Test polynomial regression with minimal data."""
        result = linreg_core.polynomial_regression(
            poly_small_y, poly_small_x, degree=2
        )

        # Should fit perfectly with 5 points and degree 2
        assert result.r_squared > 0.99

    def test_polynomial_mismatched_lengths_raises(self, poly_x, poly_y_quadratic):
        """Test that mismatched y and x lengths raise an error."""
        with pytest.raises(Exception):  # Could be ValueError or PyValueError
            linreg_core.polynomial_regression(
                poly_y_quadratic[:5], poly_x, degree=2
            )


class TestPolynomialNumpyIntegration:
    """Tests for numpy array integration."""

    def test_polynomial_with_numpy_arrays(self, poly_x_np, poly_y_np):
        """Test polynomial regression with numpy arrays."""
        result = linreg_core.polynomial_regression(
            poly_y_np, poly_x_np, degree=2
        )

        assert result.degree == 2
        assert result.r_squared > 0.9

    def test_polynomial_predict_numpy(self, poly_x_np, poly_y_np):
        """Test prediction with numpy array input."""
        result = linreg_core.polynomial_regression(
            poly_y_np, poly_x_np, degree=2
        )
        x_new = np.array([11.0, 12.0, 13.0])
        predictions = result.predict(x_new)

        assert isinstance(predictions, list)
        assert len(predictions) == 3

    def test_polynomial_mixed_types(self, poly_x_np, poly_y_quadratic):
        """Test with numpy x and list y."""
        result = linreg_core.polynomial_regression(
            poly_y_quadratic, poly_x_np, degree=2
        )

        assert result.degree == 2
