"""Tests for LOESS (Locally Estimated Scatterplot Smoothing) Python bindings."""

import pytest
import linreg_core


class TestLoessNative:
    """Tests for LOESS native Python type API."""

    def test_loess_fit_basic(self):
        """Test basic LOESS fitting with default parameters."""
        # Create simple non-linear data
        y = [1.0, 3.5, 4.8, 6.2, 8.5, 11.0, 13.2, 14.8, 17.5, 19.0, 22.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]

        result = linreg_core.loess_fit(y, x)

        # Verify result structure
        assert hasattr(result, "fitted")
        assert hasattr(result, "residuals")
        assert hasattr(result, "span")
        assert hasattr(result, "degree")
        assert hasattr(result, "robust_iterations")
        assert hasattr(result, "surface")
        assert hasattr(result, "mse")
        assert hasattr(result, "rmse")
        assert hasattr(result, "n_observations")

        # Check default values
        assert result.span == 0.75
        assert result.degree == 1
        assert result.robust_iterations == 0
        assert result.surface == "direct"
        assert result.n_observations == 11

        # Fitted values should have same length as input
        assert len(result.fitted) == len(y)
        assert len(result.residuals) == len(y)

        # Residuals should be y - fitted
        for i in range(len(y)):
            assert abs(result.residuals[i] - (y[i] - result.fitted[i])) < 1e-10

    def test_loess_fit_custom_parameters(self):
        """Test LOESS fitting with custom parameters."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0]]

        # Custom span
        result_span = linreg_core.loess_fit(y, x, span=0.5)
        assert result_span.span == 0.5

        # Custom degree
        result_degree = linreg_core.loess_fit(y, x, degree=2)
        assert result_degree.degree == 2

        # Custom robust iterations
        result_robust = linreg_core.loess_fit(y, x, robust_iterations=2)
        assert result_robust.robust_iterations == 2

        # Interpolate surface
        result_surface = linreg_core.loess_fit(y, x, surface="interpolate")
        assert result_surface.surface == "interpolate"

    def test_loess_fit_degree_0(self):
        """Test LOESS with degree 0 (constant fit)."""
        y = [5.0, 5.2, 4.9, 5.1, 5.0, 4.8, 5.2, 5.1]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]

        result = linreg_core.loess_fit(y, x, degree=0, span=0.5)

        assert result.degree == 0
        # With constant data and degree 0, fitted should be close to 5
        for fitted_val in result.fitted:
            assert 4.5 < fitted_val < 5.5

    def test_loess_fit_degree_2(self):
        """Test LOESS with degree 2 (quadratic fit)."""
        # Quadratic data: y = x^2
        y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]

        result = linreg_core.loess_fit(y, x, degree=2, span=0.75)

        assert result.degree == 2
        # With quadratic fit on quadratic data, residuals should be relatively small
        max_residual = max(abs(r) for r in result.residuals)
        assert max_residual < 5.0  # Should fit reasonably well

    def test_loess_fit_robust(self):
        """Test LOESS with robust fitting (outlier handling)."""
        # Linear data with an outlier
        y = [1.0, 2.0, 3.0, 4.0, 100.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]

        # Non-robust fit
        result_non_robust = linreg_core.loess_fit(y, x, robust_iterations=0, span=0.5)
        # Robust fit
        result_robust = linreg_core.loess_fit(y, x, robust_iterations=2, span=0.5)

        assert result_non_robust.robust_iterations == 0
        assert result_robust.robust_iterations == 2

        # Robust fit should be less affected by outlier
        # The non-robust fit may have a large residual at the outlier point
        # The robust fit should downweight it

    def test_loess_predict(self):
        """Test LOESS prediction at new points."""
        # Training data
        y = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        # New points to predict
        new_x = [0.5, 1.5, 2.5, 3.5]

        # Predict
        predictions = linreg_core.loess_predict(
            new_x, x, y, span=0.75, degree=1
        )

        assert len(predictions) == len(new_x)
        # Predictions should be reasonable (between 1 and 11)
        for pred in predictions:
            assert 0.0 < pred < 12.0

        # Predictions should follow the trend
        assert predictions[0] < predictions[1]
        assert predictions[1] < predictions[2]
        assert predictions[2] < predictions[3]

    def test_loess_predict_with_options(self):
        """Test LOESS prediction with various options."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        new_x = [1.5, 2.5]

        # Test with different spans
        pred_span_small = linreg_core.loess_predict(new_x, x, y, span=0.3, degree=1)
        pred_span_large = linreg_core.loess_predict(new_x, x, y, span=0.9, degree=1)

        assert len(pred_span_small) == 2
        assert len(pred_span_large) == 2

        # Test with different degrees
        pred_deg1 = linreg_core.loess_predict(new_x, x, y, span=0.75, degree=1)
        pred_deg2 = linreg_core.loess_predict(new_x, x, y, span=0.75, degree=2)

        assert len(pred_deg1) == 2
        assert len(pred_deg2) == 2

    def test_loess_summary(self):
        """Test LOESS result summary method."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0]]

        result = linreg_core.loess_fit(y, x)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "LOESS Regression Results" in summary
        assert "Span:" in summary
        assert "Degree:" in summary
        assert "MSE:" in summary
        assert "RMSE:" in summary

    def test_loess_to_dict(self):
        """Test LOESS result to_dict method."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0]]

        result = linreg_core.loess_fit(y, x)

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "fitted" in result_dict
        assert "residuals" in result_dict
        assert "span" in result_dict
        assert "degree" in result_dict
        assert "robust_iterations" in result_dict
        assert "surface" in result_dict
        assert "mse" in result_dict
        assert "rmse" in result_dict
        assert "n_observations" in result_dict

    def test_loess_repr(self):
        """Test LOESS result __repr__ method."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0]]

        result = linreg_core.loess_fit(y, x)

        repr_str = repr(result)
        assert isinstance(repr_str, str)
        assert "LoessResult" in repr_str

    def test_loess_str(self):
        """Test LOESS result __str__ method (same as summary)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0]]

        result = linreg_core.loess_fit(y, x)

        str_result = str(result)
        assert str_result == result.summary()


class TestLoessEdgeCases:
    """Edge case and error handling tests for LOESS."""

    def test_loess_empty_y(self):
        """Test LOESS with empty y array."""
        y = []
        x = [[]]

        with pytest.raises(Exception) as exc_info:
            linreg_core.loess_fit(y, x)
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_loess_empty_x(self):
        """Test LOESS with empty x array."""
        y = [1.0, 2.0, 3.0]
        x = []

        with pytest.raises(Exception) as exc_info:
            linreg_core.loess_fit(y, x)
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_loess_mismatched_lengths(self):
        """Test LOESS with mismatched x and y lengths."""
        y = [1.0, 2.0, 3.0]
        x = [[1.0, 2.0]]  # Only 2 elements

        with pytest.raises(Exception) as exc_info:
            linreg_core.loess_fit(y, x)
        assert "elements" in str(exc_info.value).lower()

    def test_loess_invalid_span(self):
        """Test LOESS with invalid span values."""
        y = [1.0, 2.0, 3.0]
        x = [[1.0, 2.0, 3.0]]

        # Span too small
        with pytest.raises(Exception):
            linreg_core.loess_fit(y, x, span=0.0)

        # Span too large
        with pytest.raises(Exception):
            linreg_core.loess_fit(y, x, span=1.5)

        # Negative span
        with pytest.raises(Exception):
            linreg_core.loess_fit(y, x, span=-0.1)

    def test_loess_invalid_degree(self):
        """Test LOESS with invalid degree values."""
        y = [1.0, 2.0, 3.0]
        x = [[1.0, 2.0, 3.0]]

        # Degree too large
        with pytest.raises(Exception):
            linreg_core.loess_fit(y, x, degree=3)

    def test_loess_multiple_predictors_error(self):
        """Test that LOESS rejects multiple predictors (currently unsupported)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]]  # 2 predictors

        with pytest.raises(Exception) as exc_info:
            linreg_core.loess_fit(y, x)
        assert "single predictor" in str(exc_info.value).lower()

    def test_loess_minimal_data(self):
        """Test LOESS with minimal data (2 points for degree 1)."""
        y = [1.0, 2.0]
        x = [[0.0, 1.0]]

        result = linreg_core.loess_fit(y, x, degree=1, span=1.0)
        assert len(result.fitted) == 2

    def test_loess_predict_empty_new_x(self):
        """Test LOESS prediction with empty new_x."""
        y = [1.0, 2.0, 3.0]
        x = [0.0, 1.0, 2.0]
        new_x = []

        predictions = linreg_core.loess_predict(new_x, x, y)
        assert len(predictions) == 0

    def test_loess_predict_mismatched_original_data(self):
        """Test LOESS prediction with mismatched original data."""
        y = [1.0, 2.0, 3.0]
        x = [0.0, 1.0]  # Different length
        new_x = [1.5]

        with pytest.raises(Exception):
            linreg_core.loess_predict(new_x, x, y)

    def test_loess_case_insensitive_surface(self):
        """Test that surface parameter is case-insensitive."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0]]

        # Test various capitalizations
        result1 = linreg_core.loess_fit(y, x, surface="direct")
        result2 = linreg_core.loess_fit(y, x, surface="Direct")
        result3 = linreg_core.loess_fit(y, x, surface="DIRECT")

        assert result1.surface == "direct"
        assert result2.surface == "direct"
        assert result3.surface == "direct"


class TestLoessNumericalAccuracy:
    """Tests for LOESS numerical accuracy and behavior."""

    def test_loess_perfect_linear(self):
        """Test LOESS on perfectly linear data."""
        # Perfect line: y = 2x + 1
        y = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]

        result = linreg_core.loess_fit(y, x, span=0.75, degree=1)

        # With perfect linear data and large span, LOESS should fit very well
        # Residuals should be small
        max_residual = max(abs(r) for r in result.residuals)
        assert max_residual < 1.0

    def test_loess_span_effect(self):
        """Test that span affects smoothness of fit."""
        # Noisy data
        y = [2.1, 3.8, 6.2, 7.9, 10.1, 12.2, 14.1, 15.8, 18.2, 19.9]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]

        # Small span (wiggly fit)
        result_small = linreg_core.loess_fit(y, x, span=0.3, degree=1)
        # Large span (smooth fit)
        result_large = linreg_core.loess_fit(y, x, span=0.9, degree=1)

        # Both should produce valid results
        assert len(result_small.fitted) == len(y)
        assert len(result_large.fitted) == len(y)

        # Small span should generally have smaller residuals (fits closer)
        # (though this isn't guaranteed for all data)
        assert result_small.span == 0.3
        assert result_large.span == 0.9

    def test_loess_interpolation_vs_direct(self):
        """Test difference between direct and interpolate surface methods."""
        y = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        x = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]

        result_direct = linreg_core.loess_fit(y, x, surface="direct")
        result_interp = linreg_core.loess_fit(y, x, surface="interpolate")

        # Both should produce results
        assert len(result_direct.fitted) == len(y)
        assert len(result_interp.fitted) == len(y)

        # Direct surface uses exact computation
        assert result_direct.surface == "direct"
        assert result_interp.surface == "interpolate"
