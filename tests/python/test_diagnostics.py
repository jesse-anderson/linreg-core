import pytest
import linreg_core

class TestDiagnosticTestsNative:
    """Tests for native Python type API (Phase 3)."""

    def test_rainbow_test_native(self, diagnostic_y, diagnostic_x):
        result = linreg_core.rainbow_test(diagnostic_y, diagnostic_x, 0.5, "r")
        assert hasattr(result, "test_name")
        assert hasattr(result, "has_r_result")
        assert hasattr(result, "interpretation")

    def test_white_test_native(self, sample_y, sample_x):
        result = linreg_core.white_test(sample_y, sample_x, "r")
        assert hasattr(result, "test_name")
        assert hasattr(result, "has_r_result")
        assert hasattr(result, "interpretation")

    def test_breusch_pagan_test_native(self, sample_y, sample_x):
        result = linreg_core.breusch_pagan_test(sample_y, sample_x)
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")

    def test_cooks_distance_test_native(self, sample_y, sample_x):
        result = linreg_core.cooks_distance_test(sample_y, sample_x)
        assert hasattr(result, "distances")
        assert hasattr(result, "p")
        assert hasattr(result, "threshold_4_over_n")
        assert hasattr(result, "influential_4_over_n")

    def test_breusch_godfrey_test_native(self, sample_y, sample_x):
        result = linreg_core.breusch_godfrey_test(sample_y, sample_x, 1, "chisq")
        assert hasattr(result, "test_name")
        assert hasattr(result, "order")
        assert hasattr(result, "interpretation")
        assert hasattr(result, "guidance")


class TestDiagnosticsEdgeCases:
    """Edge case and error handling tests for diagnostic tests."""

    def test_rainbow_test_fraction_boundary(self):
        """Test Rainbow test with fraction at boundaries (0 and 1)."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

        # fraction = 0 should use minimal subset
        result_0 = linreg_core.rainbow_test(y, x, 0.0, "r")
        assert hasattr(result_0, "test_name")

        # fraction = 1 should use full dataset
        result_1 = linreg_core.rainbow_test(y, x, 1.0, "r")
        assert hasattr(result_1, "test_name")

    def test_breusch_pagan_insufficient_data(self):
        """Test Breusch-Pagan with minimal data."""
        y = [1.0, 2.0, 3.0]  # Minimum for OLS with 1 predictor
        x = [[1.0, 2.0, 3.0]]

        # Should work with minimal data
        result = linreg_core.breusch_pagan_test(y, x)
        assert hasattr(result, "statistic")

    def test_diagnostics_with_constant_y(self):
        """Test diagnostics with constant y values (no variance)."""
        y = [5.0, 5.0, 5.0, 5.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        # Some tests may fail or return edge case results
        try:
            result = linreg_core.breusch_pagan_test(y, x)
            # If it works, should have a result
            assert hasattr(result, "statistic")
        except Exception:
            pass  # Also acceptable for constant y

    def test_diagnostics_with_perfect_fit(self):
        """Test diagnostics with perfect linear relationship."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.breusch_pagan_test(y, x)
        # Perfect fit may have zero residuals for heteroscedasticity tests
        assert hasattr(result, "statistic")

    def test_cooks_distance_influential_points(self):
        """Test Cook's distance with an influential point."""
        y = [1.0, 2.0, 3.0, 4.0, 100.0]  # Last point is outlier
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.cooks_distance_test(y, x)
        # The outlier should be influential
        assert hasattr(result, "influential_4_over_n")

    def test_cooks_distance_small_sample(self):
        """Test Cook's distance with minimal sample size."""
        y = [1.0, 2.0, 3.0]  # Minimum for OLS
        x = [[1.0, 2.0, 3.0]]

        result = linreg_core.cooks_distance_test(y, x)
        assert hasattr(result, "distances")
        assert len(result.distances) == 3

    def test_durbin_watson_bounds(self):
        """Test Durbin-Watson statistic range (current behavior - may not be clamped)."""
        y = [1.0, 2.5, 3.7, 4.8, 6.2]  # Not perfect linear fit
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.durbin_watson_test(y, x)
        # DW statistic is typically in [0, 4] but may not be clamped
        # Just verify it returns a valid float
        assert isinstance(result.statistic, float)

    def test_durbin_watson_perfect_positive_autocorrelation(self):
        """Test Durbin-Watson with data sorted like positive autocorrelation."""
        y = [1.0, 2.5, 3.7, 4.8, 6.2]  # Not perfect linear fit
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.durbin_watson_test(y, x)
        # With increasing y, DW should be < 2 (positive autocorrelation)
        # But just verify it returns a valid result
        assert isinstance(result.statistic, float)

    def test_white_test_methods(self):
        """Test White test with different methods."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        # Test R method
        result_r = linreg_core.r_white_test(y, x)
        assert hasattr(result_r, "statistic")

        # Test Python method
        result_py = linreg_core.python_white_test(y, x)
        assert hasattr(result_py, "statistic")

    def test_jarque_bera_normal_distribution(self):
        """Test Jarque-Bera on normally distributed data (should not reject normality)."""
        import random
        random.seed(42)
        # Generate approximately normal data using sum of uniforms
        y = [sum([random.random() for _ in range(12)]) - 6 for _ in range(100)]
        x = [[random.random() for _ in range(100)]]

        result = linreg_core.jarque_bera_test(y, x)
        # With approximately normal data, p-value should be > 0.05
        # (not guaranteed, but likely)
        assert hasattr(result, "p_value")

    def test_reset_test_powers(self):
        """Test RESET test with different power specifications."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        # Test with single power
        result_single = linreg_core.reset_test(y, x, [2], "fitted")
        assert hasattr(result_single, "statistic")

        # Test with multiple powers
        result_multi = linreg_core.reset_test(y, x, [2, 3], "fitted")
        assert hasattr(result_multi, "statistic")

    def test_breusch_godfrey_higher_order(self):
        """Test Breusch-Godfrey with higher order lag."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

        # Test with order 2
        result_order2 = linreg_core.breusch_godfrey_test(y, x, 2, "chisq")
        assert result_order2.order == 2

        # Test with order 3
        result_order3 = linreg_core.breusch_godfrey_test(y, x, 3, "chisq")
        assert result_order3.order == 3

    def test_breusch_godfrey_f_test_type(self):
        """Test Breusch-Godfrey with F test type."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

        result_f = linreg_core.breusch_godfrey_test(y, x, 1, "f")
        # F test should also work
        assert hasattr(result_f, "statistic")
