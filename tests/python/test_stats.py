import pytest
import linreg_core

class TestStatisticalUtilities:
    def test_get_version(self):
        version = linreg_core.get_version()
        assert isinstance(version, str)
        assert len(version.split(".")) >= 2  # Major.Minor at minimum

    def test_get_t_cdf(self):
        result = linreg_core.get_t_cdf(1.96, 20)
        assert isinstance(result, float)
        assert 0.5 < result < 1.0  # t=1.96 should be > 0.5

    def test_get_t_critical(self):
        result = linreg_core.get_t_critical(0.05, 20)
        assert isinstance(result, float)
        # For alpha=0.05, two-tailed, t_crit should be around 2.09
        assert 2.0 < result < 2.2

    def test_get_normal_inverse(self):
        result = linreg_core.get_normal_inverse(0.975)
        assert isinstance(result, float)
        # 97.5th percentile of standard normal is ~1.96
        assert 1.95 < result < 1.97


class TestStatisticsEdgeCases:
    """Edge case tests for statistical utility functions."""

    def test_stats_mean_empty_list(self):
        """Test mean of empty list raises exception."""
        # Empty input should raise an exception
        with pytest.raises(Exception) as exc_info:
            linreg_core.stats_mean([])
        assert "empty" in str(exc_info.value).lower()

    def test_stats_mean_single_value(self):
        """Test mean of a single value."""
        result = linreg_core.stats_mean([42.0])
        assert result == 42.0

    def test_stats_mean_all_zeros(self):
        """Test mean of all zeros."""
        result = linreg_core.stats_mean([0.0, 0.0, 0.0])
        assert result == 0.0

    def test_stats_variance_empty_list(self):
        """Test variance of empty list (current behavior)."""
        result = linreg_core.stats_variance([])
        # Returns NaN for empty input (variance undefined)
        assert result != result  # NaN check

    def test_stats_variance_single_value(self):
        """Test variance of a single value."""
        result = linreg_core.stats_variance([42.0])
        # Returns NaN for single value (variance undefined)
        assert result != result  # NaN check

    def test_stats_stddev_empty_list(self):
        """Test stddev of empty list (current behavior)."""
        result = linreg_core.stats_stddev([])
        # Returns NaN for empty input
        assert result != result

    def test_stats_median_empty_list(self):
        """Test median of empty list (current behavior)."""
        result = linreg_core.stats_median([])
        # Returns NaN for empty input
        assert result != result

    def test_stats_median_single_value(self):
        """Test median of a single value."""
        result = linreg_core.stats_median([42.0])
        assert result == 42.0

    def test_stats_median_even_length(self):
        """Test median of even-length list (average of two middle values)."""
        result = linreg_core.stats_median([1.0, 2.0, 3.0, 4.0])
        assert result == 2.5

    def test_stats_quantile_extremes(self):
        """Test quantile at extremes (0 and 1)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        q0 = linreg_core.stats_quantile(data, 0.0)
        q1 = linreg_core.stats_quantile(data, 1.0)

        assert q0 == 1.0
        assert q1 == 5.0

    def test_stats_correlation_empty_lists(self):
        """Test correlation of empty lists raises exception."""
        # Empty input should raise an exception
        with pytest.raises(Exception) as exc_info:
            linreg_core.stats_correlation([], [])
        assert "observation" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()

    def test_stats_correlation_single_element(self):
        """Test correlation with single element raises exception."""
        # Single element should raise an exception
        with pytest.raises(Exception) as exc_info:
            linreg_core.stats_correlation([1.0], [2.0])
        assert "observation" in str(exc_info.value).lower()

    def test_stats_correlation_perfect_positive(self):
        """Test correlation with perfect positive relationship."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2*x

        result = linreg_core.stats_correlation(x, y)
        assert abs(result - 1.0) < 1e-10

    def test_stats_correlation_perfect_negative(self):
        """Test correlation with perfect negative relationship."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]  # y = 6 - x (perfect negative)

        result = linreg_core.stats_correlation(x, y)
        assert abs(result - (-1.0)) < 1e-10

    def test_stats_with_nan_values(self):
        """Test statistics functions with NaN values (current behavior)."""
        data_with_nan = [1.0, float('nan'), 3.0, 4.0, 5.0]

        # Mean with NaN returns NaN (propagates through)
        result = linreg_core.stats_mean(data_with_nan)
        assert result != result  # NaN check

    def test_stats_with_inf_values(self):
        """Test statistics functions with infinite values (current behavior)."""
        data_with_inf = [1.0, float('inf'), 3.0, 4.0, 5.0]

        # Mean with inf returns inf
        result = linreg_core.stats_mean(data_with_inf)
        assert result == float('inf')

    def test_get_t_cdf_extreme_values(self):
        """Test t CDF with extreme values."""
        # Very large t-value should approach 1.0
        result_large = linreg_core.get_t_cdf(100.0, 10)
        assert result_large > 0.999

        # Very negative t-value should approach 0.0
        result_small = linreg_core.get_t_cdf(-100.0, 10)
        assert result_small < 0.001

    def test_get_normal_inverse_extremes(self):
        """Test normal inverse at extreme probabilities."""
        # Near 0 should give large negative z-score
        z_001 = linreg_core.get_normal_inverse(0.001)
        assert z_001 < -3.0

        # Near 1 should give large positive z-score
        z_999 = linreg_core.get_normal_inverse(0.999)
        assert z_999 > 3.0
