"""
Tests for feature importance Python bindings.

These tests verify that the feature importance functions work correctly
through the Python interface.
"""

import pytest
import numpy as np
from linreg_core import (
    ols_regression,
    ridge_regression,
    lasso_regression,
    elastic_net_regression,
    polynomial_regression,
    # Feature importance functions (exported with py_ prefix in Rust, but exposed without it)
    py_standardized_coefficients as standardized_coefficients,
    py_shap_values_linear as shap_values_linear,
    py_shap_values_ridge as shap_values_ridge,
    py_shap_values_lasso as shap_values_lasso,
    py_shap_values_elastic_net as shap_values_elastic_net,
    py_shap_values_polynomial as shap_values_polynomial,
    py_vif_ranking as vif_ranking,
    py_vif_ranking_from_values as vif_ranking_from_values,
    py_permutation_importance_ols as permutation_importance_ols,
    py_permutation_importance_ridge as permutation_importance_ridge,
    py_permutation_importance_lasso as permutation_importance_lasso,
    py_permutation_importance_elastic_net as permutation_importance_elastic_net,
    py_permutation_importance_loess as permutation_importance_loess,
    py_feature_importance_ols as feature_importance_ols,
)


class TestStandardizedCoefficients:
    """Tests for standardized_coefficients function."""

    def test_basic(self):
        """Test basic standardized coefficients calculation."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [10.0, 20.0, 30.0, 40.0, 50.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        names = ["Intercept", "X1", "X2"]

        fit = ols_regression(y, [x1, x2], names)
        result = standardized_coefficients(fit.coefficients, [x1, x2])

        assert result.variable_names == ["X1", "X2"]
        assert len(result.standardized_coefficients) == 2
        assert result.y_std > 0

    def test_with_custom_names(self):
        """Test with custom variable names."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ols_regression(y, [x1], ["Intercept", "X1"])
        result = standardized_coefficients(
            fit.coefficients,
            [x1],
            variable_names=["Temperature"]
        )

        assert result.variable_names == ["Temperature"]

    def test_with_custom_y_std(self):
        """Test with custom y_std value."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ols_regression(y, [x1], ["Intercept", "X1"])
        result = standardized_coefficients(
            fit.coefficients,
            [x1],
            y_std=2.5
        )

        assert result.y_std == 2.5

    def test_ranking(self):
        """Test ranking method."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [3.0, 1.0, 5.0, 2.0, 4.0]  # Not correlated with x1
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2*x (approximately)

        fit = ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        result = standardized_coefficients(fit.coefficients, [x1, x2])

        ranking = result.ranking()
        assert len(ranking) == 2
        # Should be sorted by absolute value (descending)
        assert ranking[0][1] >= ranking[1][1]

    def test_summary(self):
        """Test summary method."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ols_regression(y, [x1], ["Intercept", "X1"])
        result = standardized_coefficients(fit.coefficients, [x1])

        summary = result.summary()
        assert "Standardized Coefficients" in summary
        assert "X1" in summary


class TestShapValues:
    """Tests for SHAP values functions."""

    def test_linear_shap(self):
        """Test linear SHAP values."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        names = ["Intercept", "X1", "X2"]

        fit = ols_regression(y, [x1, x2], names)
        result = shap_values_linear(fit.coefficients, [x1, x2])

        assert result.variable_names == ["X1", "X2"]
        assert len(result.shap_values) == 5  # 5 observations
        assert len(result.shap_values[0]) == 2  # 2 features
        assert len(result.mean_abs_shap) == 2

    def test_shap_with_custom_names(self):
        """Test SHAP with custom names."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ols_regression(y, [x1], ["Intercept", "X1"])
        result = shap_values_linear(
            fit.coefficients,
            [x1],
            variable_names=["Temperature"]
        )

        assert result.variable_names == ["Temperature"]

    def test_local_accuracy(self):
        """Test that SHAP values are computed correctly."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ols_regression(y, [x1], ["Intercept", "X1"])
        shap = shap_values_linear(fit.coefficients, [x1])

        # SHAP value formula: coef * (x - mean(x))
        # First observation: x[0] = 1.0, mean(x) = 3.0
        # So SHAP = coef * (1.0 - 3.0) = coef * (-2.0)
        # The base_value is just the intercept

        # Verify SHAP values are computed correctly
        x_mean = sum(x1) / len(x1)
        expected_shap = fit.coefficients[1] * (x1[0] - x_mean)
        assert abs(shap.shap_values[0][0] - expected_shap) < 1e-10

    def test_observation_contribution(self):
        """Test observation_contribution method."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        shap = shap_values_linear(fit.coefficients, [x1, x2])

        contrib = shap.observation_contribution(0)
        assert len(contrib) == 2
        assert contrib[0][0] == "X1"
        assert contrib[1][0] == "X2"

    def test_shap_ranking(self):
        """Test ranking method."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [3.0, 1.0, 5.0, 2.0, 4.0]  # Not correlated with x1
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2*x (approximately)

        fit = ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        shap = shap_values_linear(fit.coefficients, [x1, x2])

        ranking = shap.ranking()
        assert len(ranking) == 2
        # Should be sorted by mean absolute SHAP (descending)
        assert ranking[0][1] >= ranking[1][1]


class TestShapRegularized:
    """Tests for SHAP values with regularized models."""

    def test_ridge_shap(self):
        """Test SHAP values for Ridge regression."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ridge_regression(y, [x1, x2], lambda_val=1.0)
        result = shap_values_ridge([x1, x2], fit)

        assert result.variable_names == ["X1", "X2"]
        assert len(result.shap_values) == 5
        assert len(result.shap_values[0]) == 2

    def test_lasso_shap(self):
        """Test SHAP values for Lasso regression."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = lasso_regression(y, [x1, x2], lambda_val=0.1)
        result = shap_values_lasso([x1, x2], fit)

        assert result.variable_names == ["X1", "X2"]
        assert len(result.shap_values) == 5

    def test_elastic_net_shap(self):
        """Test SHAP values for Elastic Net regression."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = elastic_net_regression(y, [x1, x2], lambda_val=0.1, alpha=0.5)
        result = shap_values_elastic_net([x1, x2], fit)

        assert result.variable_names == ["X1", "X2"]
        assert len(result.shap_values) == 5

    def test_polynomial_shap(self):
        """Test SHAP values for polynomial regression."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 4.0, 9.0, 16.0, 25.0]  # y = x^2

        fit = polynomial_regression(y, x, degree=2)
        result = shap_values_polynomial(x, fit)

        assert len(result.variable_names) == 2  # x, x²
        assert len(result.shap_values) == 5
        assert len(result.shap_values[0]) == 2


class TestVifRanking:
    """Tests for VIF ranking functions."""

    def test_vif_from_ols_result(self):
        """Test VIF ranking from OLS result."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [1.0, 2.0, 3.0, 4.0, 5.0]  # Highly correlated with x1
        x3 = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfectly correlated with x1
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ols_regression(y, [x1, x2, x3], ["Intercept", "X1", "X2", "X3"])
        result = vif_ranking(fit)

        assert len(result.variable_names) == 3
        assert len(result.vif_values) == 3
        # x3 should have high VIF due to perfect correlation with x1
        assert result.vif_values[2] > 100

    def test_vif_from_values(self):
        """Test VIF ranking from values directly."""
        vif_values = [1.2, 3.5, 8.5]
        names = ["Low", "Medium", "High"]

        result = vif_ranking_from_values(vif_values, variable_names=names)

        assert result.variable_names == names
        assert result.vif_values == vif_values

    def test_vif_default_names(self):
        """Test VIF ranking with default names."""
        vif_values = [1.2, 3.5]

        result = vif_ranking_from_values(vif_values)

        assert result.variable_names == ["X1", "X2"]

    def test_vif_interpretation(self):
        """Test VIF interpretation helper."""
        result = vif_ranking_from_values([2.0, 7.0, 15.0])

        interpretations = result.interpretations()
        # Low multicollinearity
        assert "Low" in interpretations[0]
        # Moderate multicollinearity
        assert "Moderate" in interpretations[1]
        # High multicollinearity
        assert "High" in interpretations[2]

    def test_vif_ranking_ascending(self):
        """Test that VIF ranking sorts by ascending VIF (best first)."""
        vif_values = [8.5, 1.2, 3.5]

        result = vif_ranking_from_values(vif_values)
        ranking = result.ranking()

        # Should be sorted ascending (lowest VIF = best = first)
        assert ranking[0][1] == 1.2
        assert ranking[1][1] == 3.5
        assert ranking[2][1] == 8.5


class TestPermutationImportance:
    """Tests for permutation importance functions."""

    def test_ols_permutation_importance(self):
        """Test permutation importance for OLS."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [0.5, 1.0, 1.5, 2.0, 2.5]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        fit = ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        result = permutation_importance_ols(y, [x1, x2], fit, n_permutations=10, seed=42)

        assert len(result.variable_names) == 2
        assert len(result.importance) == 2
        assert result.baseline_score > 0.8  # High R² for this data
        assert result.n_permutations == 10

    def test_ols_permutation_with_names(self):
        """Test with custom variable names."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        fit = ols_regression(y, [x1], ["Intercept", "X1"])
        result = permutation_importance_ols(
            y, [x1], fit,
            variable_names=["Temperature"],
            n_permutations=10,
            seed=42
        )

        assert result.variable_names == ["Temperature"]

    def test_ridge_permutation_importance(self):
        """Test permutation importance for Ridge."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ridge_regression(y, [x1, x2], lambda_val=1.0)
        result = permutation_importance_ridge(
            y, [x1, x2], fit,
            n_permutations=10,
            seed=42
        )

        assert len(result.variable_names) == 2
        assert len(result.importance) == 2

    def test_lasso_permutation_importance(self):
        """Test permutation importance for Lasso."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = lasso_regression(y, [x1, x2], lambda_val=0.1)
        result = permutation_importance_lasso(
            y, [x1, x2], fit,
            n_permutations=10,
            seed=42
        )

        assert len(result.variable_names) == 2
        assert len(result.importance) == 2

    def test_elastic_net_permutation_importance(self):
        """Test permutation importance for Elastic Net."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = elastic_net_regression(y, [x1, x2], lambda_val=0.1, alpha=0.5)
        result = permutation_importance_elastic_net(
            y, [x1, x2], fit,
            n_permutations=10,
            seed=42
        )

        assert len(result.variable_names) == 2
        assert len(result.importance) == 2

    def test_loess_permutation_importance(self):
        """Test permutation importance for LOESS."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        result = permutation_importance_loess(
            y, [x],
            span=0.75,
            degree=1,
            n_permutations=5,  # Fewer for LOESS since it's slow
            seed=42
        )

        assert len(result.variable_names) == 1
        assert len(result.importance) == 1

    def test_permutation_ranking(self):
        """Test permutation importance ranking."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [0.1, 0.2, 0.3, 0.4, 0.5]  # Less important
        y = [1.0, 2.0, 3.0, 4.0, 5.0]

        fit = ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        result = permutation_importance_ols(
            y, [x1, x2], fit,
            n_permutations=20,
            seed=42
        )

        ranking = result.ranking()
        assert len(ranking) == 2
        # X1 should be ranked higher (more important)
        assert ranking[0][1] >= ranking[1][1]


class TestCombinedFeatureImportance:
    """Tests for combined feature_importance_ols function."""

    def test_combined_function(self):
        """Test the combined feature importance function."""
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        names = ["X1", "X2"]

        result = feature_importance_ols(y, [x1, x2], names, n_permutations=10, seed=42)

        assert "standardized_coefficients" in result
        assert "shap_values" in result
        assert "vif_ranking" in result
        assert "permutation_importance" in result

        # Check standardized coefficients
        std_coefs = result["standardized_coefficients"]
        assert len(std_coefs.standardized_coefficients) == 2

        # Check SHAP values
        shap = result["shap_values"]
        assert len(shap.shap_values) == 5

        # Check VIF ranking
        vif = result["vif_ranking"]
        assert len(vif.vif_values) == 2

        # Check permutation importance
        perm = result["permutation_importance"]
        assert len(perm.importance) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_predictor(self):
        """Test with only one predictor."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.5, 3.7, 4.2, 5.1, 6.3]

        fit = ols_regression(y, [x], ["Intercept", "X"])
        shap = shap_values_linear(fit.coefficients, [x])

        assert len(shap.shap_values[0]) == 1
