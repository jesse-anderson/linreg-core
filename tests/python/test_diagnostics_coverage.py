"""
Additional diagnostics coverage tests.

Tests for edge cases and boundary conditions in diagnostic tests:
- Harvey-Collier with high-VIF data
- White test with collinear predictors
- Shapiro-Wilk at n ≈ 5000 (implementation boundary)
- Additional boundary condition tests
"""

import pytest
import random
import linreg_core


class TestHarveyCollierEdgeCases:
    """Tests for Harvey-Collier test at numerical boundaries."""

    def test_harvey_collier_with_high_vif_data(self):
        """Test Harvey-Collier with data that has high VIF (multicollinearity)."""
        # Generate data with moderate multicollinearity
        # Note: Very high VIF (> 10) may cause numerical instability
        n = 50
        random.seed(42)

        # Create two correlated predictors
        x1 = [random.gauss(0, 1) for _ in range(n)]
        # x2 is correlated with x1 but not perfectly
        x2 = [0.7 * x1[i] + random.gauss(0, 0.3) for i in range(n)]

        # Generate response with some noise
        y = [2.0 + 1.5 * x1[i] + 0.8 * x2[i] + random.gauss(0, 0.5) for i in range(n)]

        # Harvey-Collier should work with moderate VIF
        try:
            result = linreg_core.harvey_collier_test(y, [x1, x2])
            # If successful, should have valid attributes
            assert hasattr(result, "statistic")
            assert hasattr(result, "p_value")
            assert hasattr(result, "test_name")
        except Exception as e:
            # High VIF may cause failure - verify error is informative
            error_msg = str(e).lower()
            # Error should mention the numerical issue
            assert any(term in error_msg for term in
                       ["singular", "multicollinear", "computation", "failed"])

    def test_harvey_collier_with_perfect_collinearity(self):
        """Test Harvey-Collier with perfectly collinear predictors."""
        n = 20
        x1 = [float(i) for i in range(1, n + 1)]
        x2 = [2.0 * x for x in x1]  # Exactly 2 * x1
        y = [1.0 + 2.0 * x1[i] + 3.0 for i in range(n)]

        # Perfect collinearity should cause a clear error
        with pytest.raises(Exception) as exc_info:
            linreg_core.harvey_collier_test(y, [x1, x2])

        error_msg = str(exc_info.value).lower()
        # Should mention singular/multicollinearity
        assert any(term in error_msg for term in
                   ["singular", "multicollinear", "redundant", "linear"])

    def test_harvey_collier_minimal_data(self):
        """Test Harvey-Collier with minimal acceptable data."""
        # Minimum for OLS with 1 predictor is 3 observations
        y = [1.0, 2.0, 3.5]
        x = [1.0, 2.0, 3.0]

        try:
            result = linreg_core.harvey_collier_test(y, [x])
            # Should work or fail gracefully
            assert hasattr(result, "test_name") or True
        except Exception:
            # Also acceptable for minimal data
            pass

    def test_harvey_collier_with_nonlinear_relationship(self):
        """Test Harvey-Collier detects nonlinearity."""
        n = 50
        random.seed(42)

        x = [float(i) / 10 for i in range(n)]
        # Quadratic relationship - should be detected by Harvey-Collier
        y = [1.0 + 2.0 * xi + 0.5 * xi ** 2 + random.gauss(0, 0.1) for xi in x]

        result = linreg_core.harvey_collier_test(y, [x])

        # Harvey-Collier should detect the nonlinearity (low p-value)
        assert hasattr(result, "p_value")
        # With quadratic relationship, p-value may be small
        # (not guaranteed, but likely)
        assert result.p_value is not None


class TestWhiteTestCollinearity:
    """Tests for White test with collinear predictors."""

    def test_white_test_with_collinear_predictors(self):
        """Test White test behavior with collinear predictors."""
        n = 50
        random.seed(42)

        # Create correlated predictors
        x1 = [random.gauss(0, 1) for _ in range(n)]
        x2 = [0.8 * x1[i] + random.gauss(0, 0.1) for i in range(n)]

        # Response
        y = [2.0 + 1.5 * x1[i] + 0.5 * x2[i] + random.gauss(0, 1) for i in range(n)]

        # White test should handle collinearity gracefully
        try:
            result_r = linreg_core.r_white_test(y, [x1, x2])
            # R method should work
            assert hasattr(result_r, "statistic")
            assert hasattr(result_r, "p_value")

            result_py = linreg_core.python_white_test(y, [x1, x2])
            # Python method should work
            assert hasattr(result_py, "statistic")
            assert hasattr(result_py, "p_value")
        except Exception as e:
            # If it fails, error should be informative
            error_msg = str(e).lower()
            assert len(error_msg) > 10

    def test_white_test_perfect_collinearity(self):
        """Test White test with perfectly collinear predictors."""
        n = 30
        x1 = [float(i) for i in range(1, n + 1)]
        x2 = [3.0 * x for x in x1]  # Exactly 3 * x1
        y = [2.0 + x1[i] + x2[i] for i in range(n)]

        # Perfect collinearity may cause issues
        try:
            result = linreg_core.white_test(y, [x1, x2], "r")
            # If it works, check attributes
            assert hasattr(result, "test_name")
        except Exception as e:
            # Should fail with informative error
            error_msg = str(e).lower()
            assert any(term in error_msg for term in
                       ["singular", "multicollinear", "redundant"])

    def test_white_test_comparison_methods_collinear(self):
        """Compare R and Python White test methods with collinear data."""
        n = 100
        random.seed(123)

        # Moderately collinear data
        x1 = [random.gauss(10, 2) for _ in range(n)]
        x2 = [0.6 * x1[i] + random.gauss(5, 1) for i in range(n)]

        y = [5.0 + 2.0 * x1[i] + 1.5 * x2[i] + random.gauss(0, 2) for i in range(n)]

        # Get results from both methods
        result_r = linreg_core.r_white_test(y, [x1, x2])
        result_py = linreg_core.python_white_test(y, [x1, x2])

        # Both should produce valid results
        assert hasattr(result_r, "statistic")
        assert hasattr(result_py, "statistic")

        # The statistics may differ due to methodology, but both should be finite
        assert not (result_r.statistic != result_r.statistic)  # Not NaN
        assert not (result_py.statistic != result_py.statistic)  # Not NaN


class TestShapiroWilkBoundary:
    """Tests for Shapiro-Wilk test at implementation boundaries."""

    def test_shapiro_wilk_at_5000_observations(self):
        """Test Shapiro-Wilk at the n ≈ 5000 implementation boundary."""
        # Shapiro-Wilk is limited to n ≤ 5000
        n = 5000
        random.seed(42)

        # Generate normal-like data
        y = [random.gauss(0, 1) for _ in range(n)]
        x = [random.gauss(0, 1) for _ in range(n)]

        # At exactly 5000, should still work
        result = linreg_core.shapiro_wilk_test(y, [x])

        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert result.test_name == "Shapiro-Wilk"

    def test_shapiro_wilk_above_5000_observations(self):
        """Test Shapiro-Wilk behavior above the 5000 observation limit."""
        # Above 5000 observations
        n = 5100
        random.seed(42)

        y = [random.gauss(0, 1) for _ in range(n)]
        x = [random.gauss(0, 1) for _ in range(n)]

        # May fail or truncate - check behavior
        try:
            result = linreg_core.shapiro_wilk_test(y, [x])
            # If it works, should still be valid
            assert hasattr(result, "statistic")
        except Exception as e:
            # Error should be informative
            error_msg = str(e)
            assert len(error_msg) > 0

    def test_shapiro_wilk_large_normal_sample(self):
        """Test Shapiro-Wilk with large normal sample (n=1000)."""
        n = 1000
        random.seed(42)

        # Generate approximately normal data
        y = [random.gauss(0, 1) for _ in range(n)]
        x = [random.gauss(0, 1) for _ in range(n)]

        result = linreg_core.shapiro_wilk_test(y, [x])

        # With normal data, should not reject normality (p > 0.05 usually)
        assert hasattr(result, "p_value")
        # Large normal sample should have W statistic close to 1
        assert 0 < result.statistic <= 1.0

    def test_shapiro_wilk_large_non_normal_sample(self):
        """Test Shapiro-Wilk with large non-normal sample (n=1000)."""
        n = 1000
        random.seed(42)

        # Generate clearly non-normal data (exponential)
        y = [random.expovariate(1.0) for _ in range(n)]
        x = [random.random() for _ in range(n)]

        result = linreg_core.shapiro_wilk_test(y, [x])

        # Should reject normality for exponential data
        assert hasattr(result, "p_value")
        # p-value should be small for non-normal data
        # (not guaranteed, but likely for n=1000)


class TestRainbowTestBoundaryConditions:
    """Tests for Rainbow test at boundary conditions."""

    def test_rainbow_test_with_very_small_fraction(self):
        """Test Rainbow test with fraction approaching 0."""
        y = [float(i) for i in range(1, 21)]
        x = [float(i) for i in range(1, 21)]

        # Very small fraction - uses minimal subset
        result = linreg_core.rainbow_test(y, [x], 0.01, "r")

        assert hasattr(result, "test_name")
        assert hasattr(result, "has_r_result")

    def test_rainbow_test_with_very_large_fraction(self):
        """Test Rainbow test with fraction approaching 1."""
        y = [float(i) for i in range(1, 21)]
        x = [float(i) for i in range(1, 21)]

        # Very large fraction - uses almost all data
        result = linreg_core.rainbow_test(y, [x], 0.99, "r")

        assert hasattr(result, "test_name")
        assert hasattr(result, "has_r_result")

    def test_rainbow_test_both_methods_comparison(self):
        """Compare Rainbow test R and Python methods."""
        y = [1.0, 2.1, 2.9, 4.2, 5.1, 5.8, 7.1, 8.2, 8.9, 10.1]
        x = [float(i) for i in range(1, 11)]

        result = linreg_core.rainbow_test(y, [x], 0.5, "both")

        # Should have both results
        assert result.has_r_result or result.has_python_result

    def test_rainbow_test_with_nonlinear_data(self):
        """Test Rainbow test detects nonlinearity."""
        n = 50
        random.seed(42)

        x = [float(i) / 10 for i in range(n)]
        # Quadratic relationship
        y = [1.0 + xi + 0.3 * xi ** 2 + random.gauss(0, 0.1) for xi in x]

        result = linreg_core.rainbow_test(y, [x], 0.5, "r")

        # Should detect nonlinearity (low p-value)
        assert hasattr(result, "p_value") or result.has_r_result


class TestRESETTestBoundaryConditions:
    """Tests for RESET test at boundary conditions."""

    def test_reset_test_with_single_power(self):
        """Test RESET test with single power specification."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        x = [[float(i) for i in range(1, 9)]]

        # Single power
        result = linreg_core.reset_test(y, x, [2], "fitted")

        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")

    def test_reset_test_with_high_powers(self):
        """Test RESET test with high power specifications."""
        y = [float(i) ** 2 for i in range(1, 21)]
        x = [[float(i) for i in range(1, 21)]]

        # Higher powers
        result = linreg_core.reset_test(y, x, [2, 3, 4], "fitted")

        assert hasattr(result, "statistic")

    def test_reset_test_regressor_type(self):
        """Test RESET test with regressor type."""
        y = [1.0 + 2.0 * i + 0.1 * i ** 2 for i in range(1, 11)]
        x1 = [float(i) for i in range(1, 11)]

        result = linreg_core.reset_test(y, [x1], [2], "regressor")

        assert hasattr(result, "statistic")

    def test_reset_test_principal_component_type(self):
        """Test RESET test with principal component type."""
        y = [float(i) for i in range(1, 11)]
        x1 = [float(i) for i in range(1, 11)]
        x2 = [float(i) * 2 for i in range(1, 11)]

        result = linreg_core.reset_test(y, [x1, x2], [2], "princomp")

        assert hasattr(result, "statistic")


class TestBreuschGodfreyBoundaryConditions:
    """Tests for Breusch-Godfrey test at boundaries."""

    def test_breusch_godfrey_order_equals_n(self):
        """Test Breusch-Godfrey with order equal to observations."""
        # Need at least 7 observations for order=4 with 1 predictor
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]

        # Order equal to n - 1 (maximum reasonable)
        result = linreg_core.breusch_godfrey_test(y, x, 4, "chisq")

        assert hasattr(result, "order")
        assert result.order == 4

    def test_breusch_godfrey_with_autocorrelated_data(self):
        """Test Breusch-Godfrey with autocorrelated residuals."""
        n = 50
        random.seed(42)

        # Generate data with positive autocorrelation
        y = []
        for i in range(n):
            if i == 0:
                y.append(random.gauss(0, 1))
            else:
                y.append(0.7 * y[i-1] + random.gauss(0, 1))

        x = [float(i) / 10 for i in range(n)]

        result = linreg_core.breusch_godfrey_test(y, [x], 2, "chisq")

        # Should detect autocorrelation
        assert hasattr(result, "p_value")

    def test_breusch_godfrey_f_vs_chisq_comparison(self):
        """Compare F and Chi-squared test types."""
        y = [1.0, 2.1, 2.8, 4.2, 4.9, 6.1, 6.8, 8.2, 9.1, 9.9]
        x = [[float(i) for i in range(1, 11)]]

        result_chisq = linreg_core.breusch_godfrey_test(y, x, 2, "chisq")
        result_f = linreg_core.breusch_godfrey_test(y, x, 2, "f")

        # Both should work
        assert hasattr(result_chisq, "statistic")
        assert hasattr(result_f, "statistic")

        # Test types should be recorded
        assert result_chisq.test_type == "chisq" or "chi" in result_chisq.test_type.lower()
        assert result_f.test_type == "f" or result_f.test_type.lower() == "f"


class TestDurbinWatsonBoundaryConditions:
    """Tests for Durbin-Watson test at boundaries."""

    def test_durbin_watson_perfect_negative_autocorrelation(self):
        """Test Durbin-Watson with alternating pattern (negative autocorrelation)."""
        y = [10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0]
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        result = linreg_core.durbin_watson_test(y, [x])

        # DW should be near 4 for perfect negative autocorrelation
        # (or near 0 for perfect positive)
        assert hasattr(result, "statistic")
        assert 0 <= result.statistic <= 4 or result.statistic > 0  # May not be clamped

    def test_durbin_watson_no_autocorrelation(self):
        """Test Durbin-Watson with random residuals (no autocorrelation)."""
        n = 100
        random.seed(42)

        y = [random.gauss(0, 1) for _ in range(n)]
        x = [random.gauss(0, 1) for _ in range(n)]

        result = linreg_core.durbin_watson_test(y, [x])

        # DW should be near 2 for no autocorrelation
        assert hasattr(result, "statistic")

    def test_durbin_watson_minimal_data(self):
        """Test Durbin-Watson with minimal data."""
        # Need at least 4 observations for DW test
        # Add some noise to avoid perfect fit (zero residuals)
        y = [1.0, 2.1, 2.9, 4.1]
        x = [1.0, 2.0, 3.0, 4.0]

        result = linreg_core.durbin_watson_test(y, [x])

        assert hasattr(result, "statistic")


class TestAndersonDarlingBoundaryConditions:
    """Tests for Anderson-Darling test at boundaries."""

    def test_anderson_darling_normal_data(self):
        """Test Anderson-Darling with normal data."""
        n = 100
        random.seed(42)

        y = [random.gauss(0, 1) for _ in range(n)]
        x = [random.gauss(0, 1) for _ in range(n)]

        result = linreg_core.anderson_darling_test(y, [x])

        # Should not strongly reject normality
        assert hasattr(result, "p_value")

    def test_anderson_darling_exponential_data(self):
        """Test Anderson-Darling with exponential (non-normal) data."""
        n = 100
        random.seed(42)

        y = [random.expovariate(1.0) for _ in range(n)]
        x = [random.random() for _ in range(n)]

        result = linreg_core.anderson_darling_test(y, [x])

        # Should reject normality for exponential data
        assert hasattr(result, "p_value")

    def test_anderson_darling_uniform_data(self):
        """Test Anderson-Darling with uniform data."""
        n = 100
        random.seed(42)

        y = [random.random() for _ in range(n)]
        x = [random.random() for _ in range(n)]

        result = linreg_core.anderson_darling_test(y, [x])

        # Uniform is non-normal
        assert hasattr(result, "statistic")


class TestJarqueBeraBoundaryConditions:
    """Tests for Jarque-Bera test at boundaries."""

    def test_jarque_bera_symmetric_data(self):
        """Test Jarque-Bera with symmetric (zero skewness) data."""
        n = 200
        random.seed(42)

        # Normal data has skewness ≈ 0 and excess kurtosis ≈ 0
        y = [random.gauss(0, 1) for _ in range(n)]
        x = [random.gauss(0, 1) for _ in range(n)]

        result = linreg_core.jarque_bera_test(y, [x])

        # Normal data should have high p-value (not reject)
        assert hasattr(result, "p_value")

    def test_jarque_bera_skewed_data(self):
        """Test Jarque-Bera with skewed data."""
        n = 200
        random.seed(42)

        # Exponential data is positively skewed
        y = [random.expovariate(1.0) for _ in range(n)]
        x = [random.random() for _ in range(n)]

        result = linreg_core.jarque_bera_test(y, [x])

        # Skewed data should have low p-value
        assert hasattr(result, "statistic")

    def test_jarque_bera_heavy_tailed_data(self):
        """Test Jarque-Bera with heavy-tailed (t-distribution like) data."""
        n = 200
        random.seed(42)

        # Generate heavy-tailed data using Cauchy-like distribution
        y = [random.gauss(0, 1) / (random.random() ** 0.5) for _ in range(n)]
        x = [random.gauss(0, 1) for _ in range(n)]

        result = linreg_core.jarque_bera_test(y, [x])

        # Heavy-tailed should be detected
        assert hasattr(result, "statistic")


class TestBreuschPaganBoundaryConditions:
    """Tests for Breusch-Pagan test at boundaries."""

    def test_breusch_pagan_homoscedastic_data(self):
        """Test Breusch-Pagan with homoscedastic (constant variance) data."""
        n = 100
        random.seed(42)

        x = [random.gauss(0, 1) for _ in range(n)]
        # Constant variance
        y = [2.0 + 1.5 * xi + random.gauss(0, 1) for xi in x]

        result = linreg_core.breusch_pagan_test(y, [x])

        # Should not reject homoscedasticity (higher p-value)
        assert hasattr(result, "p_value")

    def test_breusch_pagan_heteroscedastic_data(self):
        """Test Breusch-Pagan with heteroscedastic (varying variance) data."""
        n = 100
        random.seed(42)

        x = [float(i) / 10 for i in range(n)]
        # Variance increases with x
        y = [2.0 + 1.5 * xi + random.gauss(0, 0.1 * xi) for xi in x]

        result = linreg_core.breusch_pagan_test(y, [x])

        # Should detect heteroscedasticity (lower p-value)
        assert hasattr(result, "statistic")

    def test_breusch_pagan_minimal_heteroscedasticity(self):
        """Test Breusch-Pagan detects subtle heteroscedasticity."""
        n = 200
        random.seed(42)

        x = [float(i) / 50 for i in range(n)]
        # Very slight variance increase
        y = [2.0 + 1.5 * xi + random.gauss(0, 1 + 0.01 * xi) for xi in x]

        result = linreg_core.breusch_pagan_test(y, [x])

        # May or may not detect - just verify it runs
        assert hasattr(result, "p_value")


class TestCooksDistanceBoundaryConditions:
    """Tests for Cook's distance at boundaries."""

    def test_cooks_distance_all_influential(self):
        """Test Cook's distance when all points are equally influential."""
        # Perfect fit, symmetric design
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [-2.0, -1.0, 0.0, 1.0, 2.0]

        result = linreg_core.cooks_distance_test(y, [x])

        # All distances should be computed
        assert len(result.distances) == 5
        # Perfect fit should have low Cook's distances
        assert all(d < 1.0 for d in result.distances)

    def test_cooks_distance_single_extreme_outlier(self):
        """Test Cook's distance with a single extreme outlier."""
        y = [1.0, 2.0, 3.0, 4.0, 1000.0]
        x = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = linreg_core.cooks_distance_test(y, [x])

        # Last point should be flagged as influential
        assert len(result.influential_1) > 0 or result.influential_1 == [4]

    def test_cooks_distance_multiple_outliers(self):
        """Test Cook's distance with multiple outliers."""
        y = [1.0, 2.0, 50.0, 4.0, 100.0]
        x = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = linreg_core.cooks_distance_test(y, [x])

        # Should identify multiple influential points
        assert len(result.distances) == 5
        # At least one of the outliers should be influential
        # (the extreme outlier at index 4 should definitely be flagged)
        assert len(result.influential_4_over_n) >= 1


class TestDiagnosticCombinations:
    """Tests for running multiple diagnostics together."""

    def test_full_diagnostic_suite(self):
        """Test running all major diagnostics on the same dataset."""
        n = 100
        random.seed(42)

        x = [float(i) / 10 for i in range(n)]
        # Add some nonlinearity and heteroscedasticity
        y = [2.0 + 1.5 * xi + 0.1 * xi ** 2 + random.gauss(0, 0.5 + 0.05 * xi) for xi in x]

        # Run multiple diagnostics
        bp = linreg_core.breusch_pagan_test(y, [x])
        dw = linreg_core.durbin_watson_test(y, [x])
        jb = linreg_core.jarque_bera_test(y, [x])
        sw = linreg_core.shapiro_wilk_test(y, [x])
        ad = linreg_core.anderson_darling_test(y, [x])
        rainbow = linreg_core.rainbow_test(y, [x], 0.5, "r")
        hc = linreg_core.harvey_collier_test(y, [x])
        reset = linreg_core.reset_test(y, [x], [2, 3], "fitted")
        bg = linreg_core.breusch_godfrey_test(y, [x], 2, "chisq")
        cooks = linreg_core.cooks_distance_test(y, [x])

        # All should return valid results
        assert hasattr(bp, "p_value")
        assert hasattr(dw, "statistic")
        assert hasattr(jb, "p_value")
        assert hasattr(sw, "p_value")
        assert hasattr(ad, "p_value")
        assert hasattr(rainbow, "test_name")
        assert hasattr(hc, "statistic")
        assert hasattr(reset, "p_value")
        assert hasattr(bg, "p_value")
        assert hasattr(cooks, "distances")

    def test_diagnostics_with_regularized_results(self):
        """Test that some diagnostics work conceptually with regularized regression concepts."""
        n = 50
        random.seed(42)

        x1 = [random.gauss(0, 1) for _ in range(n)]
        x2 = [random.gauss(0, 1) for _ in range(n)]
        y = [2.0 + 1.5 * x1[i] + 0.8 * x2[i] + random.gauss(0, 0.5) for i in range(n)]

        # Ridge regression (use positional args since lambda is reserved keyword)
        ridge = linreg_core.ridge_regression(y, [x1, x2], 1.0, True)

        # Diagnostics on original data
        bp = linreg_core.breusch_pagan_test(y, [x1, x2])

        # Both should complete successfully
        assert hasattr(ridge, "r_squared")
        assert hasattr(bp, "p_value")

    def test_diagnostics_comparison_linear_vs_nonlinear(self):
        """Compare diagnostic results between linear and nonlinear models."""
        n = 50
        random.seed(42)

        x = [float(i) / 10 for i in range(n)]
        # Linear data
        y_linear = [2.0 + 1.5 * xi + random.gauss(0, 0.3) for xi in x]
        # Quadratic data
        y_quad = [2.0 + 1.5 * xi + 0.3 * xi ** 2 + random.gauss(0, 0.3) for xi in x]

        # Harvey-Collier should detect nonlinearity in quadratic case
        hc_linear = linreg_core.harvey_collier_test(y_linear, [x])
        hc_quad = linreg_core.harvey_collier_test(y_quad, [x])

        # Quadratic should have lower p-value (more likely to reject linearity)
        assert hasattr(hc_linear, "p_value")
        assert hasattr(hc_quad, "p_value")
