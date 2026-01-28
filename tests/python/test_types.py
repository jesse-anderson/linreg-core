"""Type variant tests for linreg-core Python bindings.

Tests that the Python API works with different input types beyond just lists:
- NumPy arrays (if NumPy is installed)
- pandas Series (if pandas is installed)
- Tuples
- Mixed int/float inputs
"""

import pytest
import linreg_core

# Check if NumPy is available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Check if pandas is available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
class TestNumPyArrayInputs:
    """Test that functions accept NumPy arrays as input."""

    def test_ols_with_numpy_arrays(self):
        """Test OLS regression with numpy array inputs."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        # Verify result is same as with lists
        assert isinstance(result.r_squared, float)
        assert len(result.coefficients) == 2
        assert abs(result.r_squared - 1.0) < 0.01  # Perfect fit

    def test_ridge_with_numpy_arrays(self):
        """Test Ridge regression with numpy array inputs."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        result = linreg_core.ridge_regression(y, x, 1.0, True)

        assert hasattr(result, 'intercept')
        assert hasattr(result, 'coefficients')

    def test_lasso_with_numpy_arrays(self):
        """Test Lasso regression with numpy array inputs."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        result = linreg_core.lasso_regression(y, x, 0.1, True, 1000, 1e-7)

        assert hasattr(result, 'n_nonzero')

    def test_diagnostics_with_numpy_arrays(self):
        """Test diagnostic tests with numpy array inputs."""
        y = np.array([2.1, 4.3, 5.8, 8.2, 9.7, 12.1, 13.8, 16.2, 17.9, 20.3])
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])

        result = linreg_core.breusch_pagan_test(y, x)

        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')

    def test_stats_with_numpy_arrays(self):
        """Test statistical functions with numpy array inputs."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mean = linreg_core.stats_mean(data)
        assert abs(mean - 3.0) < 1e-10

        variance = linreg_core.stats_variance(data)
        assert abs(variance - 2.5) < 1e-10

    def test_numpy_2d_array_single_column(self):
        """Test numpy array with shape (n, 1) for x variables."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # 2D array with single column - transpose to get row vector [[1, 2, 3, 4, 5]]
        x_col = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        x = x_col.T  # Transpose to get shape (1, 5) representing one predictor

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2

    def test_numpy_different_dtypes(self):
        """Test numpy arrays with different dtypes."""
        # float64 (default and supported)
        y_f64 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        x_f64 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)

        result_f64 = linreg_core.ols_regression(y_f64, x_f64, ["Intercept", "X1"])
        assert abs(result_f64.r_squared - 1.0) < 0.01

        # int64 (supported, auto-converted to float64)
        y_i64 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        x_i64 = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        result_i64 = linreg_core.ols_regression(y_i64, x_i64, ["Intercept", "X1"])
        assert abs(result_i64.r_squared - 1.0) < 0.01


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
class TestNumericCoercionWithNumPy:
    """Test numeric type coercion with NumPy arrays."""

    def test_numpy_int_to_float_coercion(self):
        """Test that numpy int arrays are coerced to float."""
        y = np.array([1, 2, 3, 4, 5])
        x = np.array([[1, 2, 3, 4, 5]])

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2
        # Results should be floats
        assert all(isinstance(c, float) for c in result.coefficients)


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
class TestPandasSeriesInputs:
    """Test that functions accept pandas Series as input."""

    def test_ols_with_pandas_series(self):
        """Test OLS regression with pandas Series inputs."""
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        x1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = linreg_core.ols_regression(y.tolist(), [x1.tolist()], ["Intercept", "X1"])
        assert len(result.coefficients) == 2

    def test_ols_with_pandas_dataframe_columns(self):
        """Test OLS regression with pandas DataFrame columns."""
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        result = linreg_core.ols_regression(df['y'].tolist(), [df['x1'].tolist()], ["Intercept", "x1"])
        assert len(result.coefficients) == 2

    def test_stats_with_pandas_series(self):
        """Test statistical functions with pandas Series inputs."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        mean = linreg_core.stats_mean(data.tolist())
        assert abs(mean - 3.0) < 1e-10

    def test_diagnostics_with_pandas_series(self):
        """Test diagnostic tests with pandas Series inputs."""
        y = pd.Series([2.1, 4.3, 5.8, 8.2, 9.7, 12.1, 13.8, 16.2, 17.9, 20.3])
        x = pd.Series([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])

        result = linreg_core.breusch_pagan_test(y.tolist(), x.tolist())
        assert hasattr(result, 'statistic')

    def test_csv_parse_to_pandas_dataframe_workflow(self):
        """Test CSV parsing → pandas DataFrame → regression workflow."""
        csv_content = """y,x1,x2
1.0,2.0,3.0
2.0,4.0,5.0
3.0,6.0,8.0
4.0,8.0,10.0
5.0,10.0,12.0"""

        csv_result = linreg_core.parse_csv(csv_content)

        # Convert to pandas DataFrame
        df = pd.DataFrame(csv_result.data)

        # Extract numeric columns for regression
        y = df['y'].tolist()
        x = [df['x1'].tolist(), df['x2'].tolist()]

        result = linreg_core.ols_regression(y, x, ["Intercept", "x1", "x2"])

        assert len(result.coefficients) == 3


class TestTupleInputs:
    """Test that functions accept tuples as input (not just lists)."""

    def test_ols_with_tuples(self):
        """Test OLS regression with tuple inputs."""
        y = (1.0, 2.0, 3.0, 4.0, 5.0)
        x = ((1.0, 2.0, 3.0, 4.0, 5.0),)
        names = ["Intercept", "X1"]

        result = linreg_core.ols_regression(y, x, names)
        assert len(result.coefficients) == 2

    def test_stats_with_tuples(self):
        """Test statistical functions with tuple inputs."""
        data = (1.0, 2.0, 3.0, 4.0, 5.0)

        mean = linreg_core.stats_mean(data)
        assert abs(mean - 3.0) < 1e-10

        variance = linreg_core.stats_variance(data)
        assert abs(variance - 2.5) < 1e-10

        stddev = linreg_core.stats_stddev(data)
        assert abs(stddev - 1.58113883) < 1e-6

        median = linreg_core.stats_median(data)
        assert median == 3.0

    def test_correlation_with_tuples(self):
        """Test correlation with tuple inputs."""
        x = (1, 2, 3, 4, 5)
        y = (2, 4, 6, 8, 10)  # y = 2*x (perfect correlation)

        result = linreg_core.stats_correlation(x, y)
        assert abs(result - 1.0) < 1e-10


class TestMixedIntFloatInputs:
    """Test that functions accept integers where floats are expected."""

    def test_ols_with_integers(self):
        """Test OLS regression with integer inputs (should work)."""
        y = [1, 2, 3, 4, 5]
        x = [[1, 2, 3, 4, 5]]

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2
        # Perfect linear relationship with integers
        assert abs(result.r_squared - 1.0) < 0.01

    def test_ridge_with_integers(self):
        """Test Ridge regression with integer inputs."""
        y = [1, 2, 3, 4, 5]
        x = [[1, 2, 3, 4, 5]]

        result = linreg_core.ridge_regression(y, x, 1.0, True)
        assert hasattr(result, 'intercept')

    def test_lasso_with_integers(self):
        """Test Lasso regression with integer inputs."""
        y = [1, 2, 3, 4, 5]
        x = [[1, 2, 3, 4, 5]]

        result = linreg_core.lasso_regression(y, x, 0.1, True, 1000, 1e-7)
        assert hasattr(result, 'n_nonzero')

    def test_stats_with_integers(self):
        """Test statistical functions with integer inputs."""
        data = [1, 2, 3, 4, 5]

        mean = linreg_core.stats_mean(data)
        assert abs(mean - 3.0) < 1e-10

        variance = linreg_core.stats_variance(data)
        assert abs(variance - 2.5) < 1e-10

        stddev = linreg_core.stats_stddev(data)
        assert abs(stddev - 1.58113883) < 1e-6

    def test_correlation_with_integers(self):
        """Test correlation with integer inputs."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # y = 2*x (perfect correlation)

        result = linreg_core.stats_correlation(x, y)
        assert abs(result - 1.0) < 1e-10


class TestNumericTypeCoercion:
    """Test that numeric type coercion works correctly."""

    def test_ols_with_mixed_int_float_y(self):
        """Test OLS with mixed int/float in y."""
        y = [1, 2.5, 3, 4.5, 5]  # mixed ints and floats
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2

    def test_ols_with_mixed_int_float_x(self):
        """Test OLS with mixed int/float in x."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1, 2, 3, 4, 5]]  # ints

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2


class TestSequenceProtocolInputs:
    """Test that other sequence types work."""

    def test_ols_with_range_list(self):
        """Test OLS with range objects converted to list."""
        y = list(range(1, 6))  # [1, 2, 3, 4, 5]
        x = [list(range(1, 6))]  # [[1, 2, 3, 4, 5]]

        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        assert len(result.coefficients) == 2

    def test_stats_with_range_list(self):
        """Test stats functions with range objects converted to list."""
        data = list(range(1, 11))  # [1, 2, ..., 10]

        mean = linreg_core.stats_mean(data)
        assert abs(mean - 5.5) < 1e-10
