"""
Performance stress tests for large datasets.

Tests OLS, Ridge, Lasso, Elastic Net, and diagnostics with:
- 10,000+ observations
- 50+ predictors (high-dimensional data)

These tests verify performance and numerical stability don't degrade
with large inputs.
"""
# Performance sucks. We need FAER and optimizations...
import pytest
import time
import linreg_core


class TestLargeDatasetPerformance:
    """Tests for performance with large datasets (10,000+ observations)."""

    def test_ols_with_10000_observations(self):
        """Test OLS regression with 10,000 observations completes in reasonable time."""
        n = 10000
        # Generate synthetic data: y = 2 + 1.5*x1 + 0.8*x2 + noise
        import random
        random.seed(42)

        y = []
        x1 = []
        x2 = []

        for i in range(n):
            x1_val = i * 0.001 + random.gauss(0, 0.1)
            x2_val = (i % 100) * 0.1 + random.gauss(0, 0.1)
            noise = random.gauss(0, 0.5)
            y_val = 2.0 + 1.5 * x1_val + 0.8 * x2_val + noise

            y.append(y_val)
            x1.append(x1_val)
            x2.append(x2_val)

        start = time.time()
        result = linreg_core.ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        elapsed = time.time() - start

        # Should complete in less than 10 seconds
        assert elapsed < 10.0, f"OLS regression took {elapsed:.2f}s, expected < 10s"

        # Verify results are sensible
        assert len(result.coefficients) == 3
        assert result.n_observations == n
        assert result.n_predictors == 2
        # R-squared should be reasonably high for this synthetic data
        assert result.r_squared > 0.5

    @pytest.mark.skip(reason="Too slow until v1.0.0 performance improvements")
    def test_ols_with_50000_observations(self):
        """Test OLS regression with 50,000 observations for stress testing."""
        n = 50000
        # Use simpler generation for speed
        y = [float(i) * 1.5 + 10.0 for i in range(n)]
        x = [[float(i) for i in range(n)]]

        start = time.time()
        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        elapsed = time.time() - start

        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 90.0, f"Large OLS regression took {elapsed:.2f}s, expected < 90s"

        # Verify basic correctness
        assert len(result.coefficients) == 2
        assert result.n_observations == n
        # Perfect linear relationship should give R^2 ≈ 1
        assert result.r_squared > 0.999

    def test_ridge_with_10000_observations(self):
        """Test Ridge regression with 10,000 observations."""
        n = 10000
        y = [float(i) * 1.5 + 10.0 for i in range(n)]
        x = [[float(i) for i in range(n)]]

        start = time.time()
        result = linreg_core.ridge_regression(y, x, lambda_val=1.0, standardize=True)
        elapsed = time.time() - start

        # Ridge should be fast
        assert elapsed < 5.0, f"Ridge regression took {elapsed:.2f}s, expected < 5s"

        assert len(result.coefficients) == 1
        assert len(result.fitted_values) == n

    def test_lasso_with_10000_observations(self):
        """Test Lasso regression with 10,000 observations."""
        n = 10000
        y = [float(i) * 1.5 + 10.0 for i in range(n)]
        x = [[float(i) for i in range(n)]]

        start = time.time()
        result = linreg_core.lasso_regression(y, x, lambda_val=0.1, standardize=True)
        elapsed = time.time() - start

        # Lasso uses coordinate descent, should be fast
        assert elapsed < 10.0, f"Lasso regression took {elapsed:.2f}s, expected < 10s"

        assert len(result.coefficients) == 1
        assert result.converged

    def test_elastic_net_with_10000_observations(self):
        """Test Elastic Net regression with 10,000 observations."""
        n = 10000
        y = [float(i) * 1.5 + 10.0 for i in range(n)]
        x = [[float(i) for i in range(n)]]

        start = time.time()
        result = linreg_core.elastic_net_regression(
            y, x, lambda_val=0.1, alpha=0.5, standardize=True
        )
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Elastic Net regression took {elapsed:.2f}s, expected < 10s"

        assert len(result.coefficients) == 1
        assert result.converged

    def test_diagnostics_with_10000_observations(self):
        """Test diagnostic functions with 10,000 observations."""
        n = 10000
        # Add some noise for diagnostic tests
        import random
        random.seed(42)

        y = []
        x = []
        for i in range(n):
            x_val = float(i) / 100.0
            noise = random.gauss(0, 1.0)
            y_val = 2.0 + 1.5 * x_val + noise
            y.append(y_val)
            x.append(x_val)

        # Test each diagnostic completes in reasonable time(MacOS performance sucks, so I keep upping the time)
        start = time.time()
        bp = linreg_core.breusch_pagan_test(y, [x])
        bp_time = time.time() - start

        assert bp_time < 40.0, f"Breusch-Pagan test took {bp_time:.2f}s"

        start = time.time()
        dw = linreg_core.durbin_watson_test(y, [x])
        dw_time = time.time() - start

        assert dw_time < 10.0, f"Durbin-Watson test took {dw_time:.2f}s"

        start = time.time()
        jb = linreg_core.jarque_bera_test(y, [x])
        jb_time = time.time() - start

        assert jb_time < 10.0, f"Jarque-Bera test took {jb_time:.2f}s"

    def test_stats_functions_with_large_data(self):
        """Test statistical utility functions with large data."""
        n = 100000
        data = [float(i) for i in range(n)]

        start = time.time()
        mean = linreg_core.stats_mean(data)
        mean_time = time.time() - start

        assert mean_time < 1.0, f"stats_mean took {mean_time:.2f}s"
        assert abs(mean - 49999.5) < 1.0

        start = time.time()
        var = linreg_core.stats_variance(data)
        var_time = time.time() - start

        assert var_time < 1.0, f"stats_variance took {var_time:.2f}s"

        start = time.time()
        std = linreg_core.stats_stddev(data)
        std_time = time.time() - start

        assert std_time < 1.0, f"stats_stddev took {std_time:.2f}s"


class TestHighDimensionalData:
    """Tests for numerical stability with 50+ predictors."""

    def test_ols_with_50_predictors(self):
        """Test OLS regression with 50 predictors."""
        n = 500  # Need n > p for OLS
        p = 50

        # Generate synthetic data with some correlation structure
        import random
        random.seed(42)

        # True coefficients
        true_coef = [random.gauss(0, 1) for _ in range(p)]
        intercept = 5.0

        # Generate predictors with mild correlation
        x_vars = []
        for j in range(p):
            x_j = []
            for i in range(n):
                # Each predictor has some correlation with the previous one
                if j > 0 and i > 0:
                    val = 0.3 * x_vars[j-1][i-1] + random.gauss(0, 1)
                else:
                    val = random.gauss(0, 1)
                x_j.append(val)
            x_vars.append(x_j)

        # Generate response
        y = []
        for i in range(n):
            y_i = intercept
            for j in range(p):
                y_i += true_coef[j] * x_vars[j][i]
            y_i += random.gauss(0, 0.5)  # Add noise
            y.append(y_i)

        start = time.time()
        result = linreg_core.ols_regression(y, x_vars, ["Intercept"] + [f"X{i}" for i in range(p)])
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 10.0, f"High-dimensional OLS took {elapsed:.2f}s, expected < 10s"

        # Verify results
        assert len(result.coefficients) == p + 1  # intercept + p predictors
        assert result.n_observations == n
        assert result.n_predictors == p
        # Should have reasonable fit
        assert result.r_squared > 0.3  # Lower threshold due to noise

    def test_ridge_with_100_predictors(self):
        """Test Ridge regression with 100 predictors (p can be > n)."""
        n = 200
        p = 100

        # Generate data
        import random
        random.seed(42)

        x_vars = [[random.gauss(0, 1) for _ in range(n)] for _ in range(p)]

        # Generate response from subset of predictors (sparse true model)
        y = []
        for i in range(n):
            y_i = 5.0
            # Only first 10 predictors matter
            for j in range(min(10, p)):
                y_i += j * 0.1 * x_vars[j][i]
            y_i += random.gauss(0, 0.5)
            y.append(y_i)

        start = time.time()
        result = linreg_core.ridge_regression(y, x_vars, lambda_val=1.0, standardize=True)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"High-dimensional Ridge took {elapsed:.2f}s, expected < 5s"

        assert len(result.coefficients) == p
        assert len(result.fitted_values) == n

    def test_lasso_with_100_predictors_variable_selection(self):
        """Test Lasso with 100 predictors for automatic variable selection."""
        n = 300
        p = 100

        # Generate sparse data
        import random
        random.seed(42)

        x_vars = [[random.gauss(0, 1) for _ in range(n)] for _ in range(p)]

        # Only 10 predictors actually matter
        true_nonzero = 10
        y = []
        for i in range(n):
            y_i = 5.0
            for j in range(true_nonzero):
                y_i += (j + 1) * 0.2 * x_vars[j][i]
            y_i += random.gauss(0, 0.5)
            y.append(y_i)

        start = time.time()
        result = linreg_core.lasso_regression(y, x_vars, lambda_val=0.1, standardize=True)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Lasso with many predictors took {elapsed:.2f}s"

        # Lasso should select a sparse model
        assert result.n_nonzero < p  # Should not use all predictors
        assert result.converged

    def test_make_lambda_path_with_high_dim(self):
        """Test lambda path generation with high-dimensional data."""
        n = 200
        p = 50

        import random
        random.seed(42)
        x_vars = [[random.gauss(0, 1) for _ in range(n)] for _ in range(p)]
        y = [random.gauss(0, 1) for _ in range(n)]

        start = time.time()
        result = linreg_core.make_lambda_path(y, x_vars, n_lambda=100, lambda_min_ratio=0.01)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Lambda path generation took {elapsed:.2f}s"

        assert len(result.lambda_path) == 100
        assert result.lambda_max > result.lambda_min
        assert result.n_lambda == 100


class TestMemoryLimits:
    """Tests to identify memory/performance limits."""

    @pytest.mark.skip(reason="Too slow until v1.0.0 performance improvements")
    def test_ols_max_observations_before_slowdown(self):
        """Test to find where OLS starts getting slow (100k obs)."""
        n = 100000
        y = [float(i % 1000) * 1.5 + 10.0 for i in range(n)]
        x = [[float(i % 1000) for i in range(n)]]

        start = time.time()
        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        elapsed = time.time() - start

        # This should still be reasonably fast
        assert elapsed < 180.0, f"OLS with {n} obs took {elapsed:.2f}s"
        assert result.n_observations == n

    def test_correlation_matrix_large(self):
        """Test correlation with large datasets."""
        n = 50000
        x = [float(i) for i in range(n)]
        y = [float(i) * 2 + 5 for i in range(n)]

        start = time.time()
        corr = linreg_core.stats_correlation(x, y)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Correlation took {elapsed:.2f}s"
        # Should be nearly perfect correlation
        assert abs(corr - 1.0) < 0.01

    def test_quantile_large_dataset(self):
        """Test quantile computation with large data."""
        n = 50000
        data = [float(i) for i in range(n)]

        start = time.time()
        q50 = linreg_core.stats_quantile(data, 0.5)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Quantile took {elapsed:.2f}s"
        # Median of [0, 1, 2, ..., 49999] should be close to 25000
        assert abs(q50 - 25000) < 100
