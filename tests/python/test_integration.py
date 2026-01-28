"""
Integration tests for end-to-end workflows.

These tests verify typical user workflows work end-to-end:
- CSV parsing → OLS regression → Diagnostics
- Result chaining and transformation
- Multi-step analysis pipelines
"""

import pytest
import linreg_core


class TestCSVToOLSPipeline:
    """Tests for CSV → OLS → Diagnostics pipeline."""

    def test_csv_to_ols_basic_workflow(self):
        """Test complete workflow: parse CSV → OLS regression."""
        csv_content = """square_feet,bedrooms,price
1200,3,245.5
1800,4,312.8
950,2,198.4
2400,4,425.6
1450,3,278.9
2000,4,356.2
1100,2,189.5
2800,5,512.3
1350,3,234.7
1650,3,298.1"""

        # Step 1: Parse CSV
        csv_result = linreg_core.parse_csv(csv_content)
        assert csv_result.n_rows == 10
        assert "square_feet" in csv_result.numeric_columns
        assert "bedrooms" in csv_result.numeric_columns
        assert "price" in csv_result.numeric_columns

        # Step 2: Extract data for regression
        data = csv_result.data
        y = [row["price"] for row in data]
        x1 = [row["square_feet"] for row in data]
        x2 = [row["bedrooms"] for row in data]

        # Step 3: Run OLS regression
        ols_result = linreg_core.ols_regression(
            y, [x1, x2],
            ["Intercept", "Square_Feet", "Bedrooms"]
        )

        # Verify regression results
        assert len(ols_result.coefficients) == 3
        assert ols_result.n_observations == 10
        assert ols_result.n_predictors == 2
        assert ols_result.r_squared > 0.8  # Should fit this data well

    def test_csv_to_ols_to_diagnostics_full_pipeline(self):
        """Test complete workflow: CSV → OLS → Diagnostics."""
        # Generate synthetic CSV data with some noise for diagnostics
        csv_content = """x1,x2,y
1.2,2.3,12.5
2.1,3.1,15.8
3.5,4.2,19.2
4.8,5.1,23.1
5.2,6.3,26.7
6.5,7.2,30.1
7.8,8.1,33.5
8.9,9.2,37.2
9.5,10.1,40.8
10.1,11.3,44.2
11.5,12.1,47.8
12.8,13.2,51.5
13.2,14.1,54.9
14.5,15.3,58.2
15.1,16.2,61.8"""

        # Step 1: Parse CSV
        csv_result = linreg_core.parse_csv(csv_content)

        # Step 2: Extract and prepare data
        data = csv_result.data
        y = [row["y"] for row in data]
        x1 = [row["x1"] for row in data]
        x2 = [row["x2"] for row in data]

        # Step 3: Run OLS regression
        ols_result = linreg_core.ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])

        # Step 4: Run diagnostic tests
        bp_result = linreg_core.breusch_pagan_test(y, [x1, x2])
        dw_result = linreg_core.durbin_watson_test(y, [x1, x2])
        jb_result = linreg_core.jarque_bera_test(y, [x1, x2])

        # Verify all steps completed successfully
        assert ols_result.r_squared > 0.9  # Strong linear relationship
        assert bp_result.p_value is not None
        assert dw_result.statistic is not None
        assert jb_result.p_value is not None

    def test_csv_to_regularized_regression_pipeline(self):
        """Test CSV → Ridge/Lasso/ElasticNet pipeline."""
        csv_content = """x1,x2,x3,x4,x5,y
1.2,2.3,3.1,4.5,5.2,25.5
2.1,3.1,4.2,5.1,6.3,35.8
3.5,4.2,5.1,6.5,7.1,45.2
4.8,5.1,6.2,7.8,8.5,58.1
5.2,6.3,7.1,8.9,9.2,68.7
6.5,7.2,8.5,9.1,10.5,78.3
7.8,8.1,9.2,10.5,11.8,92.5
8.9,9.2,10.1,11.2,12.5,105.2
9.5,10.1,11.5,12.8,13.1,118.8
10.1,11.3,12.1,13.5,14.2,132.5"""

        csv_result = linreg_core.parse_csv(csv_content)
        data = csv_result.data

        y = [row["y"] for row in data]
        x_vars = [
            [row["x1"] for row in data],
            [row["x2"] for row in data],
            [row["x3"] for row in data],
            [row["x4"] for row in data],
            [row["x5"] for row in data],
        ]

        # Test Ridge
        ridge_result = linreg_core.ridge_regression(y, x_vars, lambda_val=1.0)
        assert len(ridge_result.coefficients) == 5
        assert ridge_result.r_squared > 0.9

        # Test Lasso (increase max_iter to ensure convergence)
        lasso_result = linreg_core.lasso_regression(y, x_vars, lambda_val=0.1, standardize=True, max_iter=10000, tol=1e-7)
        assert len(lasso_result.coefficients) == 5
        assert lasso_result.converged

        # Test Elastic Net (increase max_iter to ensure convergence)
        enet_result = linreg_core.elastic_net_regression(
            y, x_vars, lambda_val=0.1, alpha=0.5, standardize=True, max_iter=10000, tol=1e-7
        )
        assert len(enet_result.coefficients) == 5
        assert enet_result.converged


class TestResultChaining:
    """Tests for chaining results between operations."""

    def test_ols_residuals_to_diagnostics(self):
        """Test using OLS residuals for further analysis."""
        y = [2.1, 4.3, 5.8, 8.2, 9.7, 12.1, 13.8, 16.2, 17.9, 20.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]

        # Run OLS
        ols_result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        # Extract residuals for diagnostic checks
        residuals = ols_result.residuals
        assert len(residuals) == 10

        # Run diagnostics on original data
        bp_result = linreg_core.breusch_pagan_test(y, x)
        assert bp_result.p_value is not None

        # Durbin-Watson should use residuals internally
        dw_result = linreg_core.durbin_watson_test(y, x)
        assert 0 <= dw_result.statistic <= 4

    def test_lambda_path_to_lasso_selection(self):
        """Test lambda path generation followed by Lasso with selected lambda."""
        n = 200
        import random
        random.seed(42)

        # Generate sparse data
        x_vars = [[random.gauss(0, 1) for _ in range(n)] for _ in range(10)]
        y = [sum(x_vars[j][i] * (j + 1) * 0.1 for j in range(5)) +
             random.gauss(0, 0.1) for i in range(n)]

        # Step 1: Generate lambda path
        lambda_path_result = linreg_core.make_lambda_path(y, x_vars, 100, 0.01)
        assert len(lambda_path_result.lambda_path) == 100

        # Step 2: Select a lambda from the middle of the path
        selected_lambda = lambda_path_result.lambda_path[50]

        # Step 3: Fit Lasso with selected lambda
        lasso_result = linreg_core.lasso_regression(y, x_vars, lambda_val=selected_lambda)

        # Verify results
        assert lasso_result.converged
        assert lasso_result.n_nonzero <= 10  # Should select sparse model

    def test_fitted_values_to_residual_analysis(self):
        """Test computing custom residuals from fitted values."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        ols_result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        # For Ridge, we have explicit fitted_values
        ridge_result = linreg_core.ridge_regression(y, x, lambda_val=0.1)
        fitted = ridge_result.fitted_values
        residuals = ridge_result.residuals

        # Verify residuals = y - fitted
        for i in range(len(y)):
            assert abs(residuals[i] - (y[i] - fitted[i])) < 1e-10


class TestMultiStepAnalysis:
    """Tests for multi-step analysis workflows."""

    def test_model_selection_workflow(self):
        """Test comparing OLS, Ridge, Lasso, and selecting best model."""
        # Generate data
        import random
        random.seed(42)
        n = 100
        p = 8

        x_vars = [[random.gauss(0, 1) for _ in range(n)] for _ in range(p)]
        y = [sum(x_vars[j][i] * 0.5 for j in range(p)) +
             random.gauss(0, 0.5) for i in range(n)]

        # Fit all models
        ols_result = linreg_core.ols_regression(y, x_vars, ["Intercept"] + [f"X{i}" for i in range(p)])
        ridge_result = linreg_core.ridge_regression(y, x_vars, lambda_val=1.0)
        lasso_result = linreg_core.lasso_regression(y, x_vars, lambda_val=0.1)

        # Collect R-squared values
        r2_values = {
            "OLS": ols_result.r_squared,
            "Ridge": ridge_result.r_squared,
            "Lasso": lasso_result.r_squared,
        }

        # All should have reasonable fit
        for model, r2 in r2_values.items():
            assert r2 > 0, f"{model} has negative R-squared"
            assert r2 <= 1.0, f"{model} has R-squared > 1"

    def test_cross_validation_style_split(self):
        """Test train/test split workflow."""
        import random
        random.seed(42)

        # Generate data
        n = 100
        x = [float(i) / 10 for i in range(n)]
        y = [xi * 2 + 5 + random.gauss(0, 0.5) for xi in x]

        # Split: 80% train, 20% test
        split_idx = int(0.8 * n)
        x_train = x[:split_idx]
        y_train = y[:split_idx]
        x_test = x[split_idx:]
        y_test = y[split_idx:]

        # Train on training data
        train_result = linreg_core.ols_regression(
            y_train, [x_train],
            ["Intercept", "X1"]
        )

        # Make predictions on test data manually
        # y_pred = intercept + coef * x_test
        intercept = train_result.coefficients[0]
        coef = train_result.coefficients[1]
        y_pred = [intercept + coef * xi for xi in x_test]

        # Compute test MSE manually
        test_mse = sum((yt - yp) ** 2 for yt, yp in zip(y_test, y_pred)) / len(y_test)

        # Test MSE should be reasonable
        assert test_mse < 10.0, f"Test MSE too high: {test_mse}"

    def test_stepwise_feature_selection_simulation(self):
        """Test adding features sequentially and tracking R-squared."""
        import random
        random.seed(42)

        n = 100
        # Generate 3 predictors, only first 2 matter
        x1 = [random.gauss(0, 1) for _ in range(n)]
        x2 = [random.gauss(0, 1) for _ in range(n)]
        x3 = [random.gauss(0, 1) for _ in range(n)]  # Noise

        y = [2 * x1[i] + 1.5 * x2[i] + random.gauss(0, 0.5) for i in range(n)]

        # Model 1: Only x1
        result1 = linreg_core.ols_regression(y, [x1], ["Intercept", "X1"])
        r2_1 = result1.r_squared

        # Model 2: x1 and x2
        result2 = linreg_core.ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        r2_2 = result2.r_squared

        # Model 3: All three
        result3 = linreg_core.ols_regression(y, [x1, x2, x3], ["Intercept", "X1", "X2", "X3"])
        r2_3 = result3.r_squared

        # R-squared should generally increase with more predictors
        assert r2_1 < r2_2 or abs(r2_1 - r2_2) < 0.1  # x2 adds information
        # Adding x3 (noise) might not help much


class TestErrorRecoveryInPipelines:
    """Tests for error handling in multi-step workflows."""

    def test_csv_with_missing_values_to_regression(self):
        """Test handling CSV with missing/non-numeric values."""
        csv_content = """x,y
1,10
2,20
NA,30
4,40
5,50"""

        csv_result = linreg_core.parse_csv(csv_content)

        # Only x rows with valid numbers should be in numeric columns
        # "NA" should not be numeric
        assert "y" in csv_result.numeric_columns

        # Extract valid rows where x is numeric
        data = csv_result.data
        valid_rows = [row for row in data if isinstance(row.get("x"), (int, float))]

        # Should have at least some valid rows
        assert len(valid_rows) >= 2

        # Can run regression on valid data
        if len(valid_rows) >= 3:
            y = [row["y"] for row in valid_rows]
            x = [row["x"] for row in valid_rows]

            result = linreg_core.ols_regression(y, [x], ["Intercept", "X"])
            assert result.r_squared > 0.5

    def test_insufficient_data_recovery(self):
        """Test recovering from insufficient data error."""
        # Too few observations
        y = [1.0, 2.0]
        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 3 predictors, 2 obs

        with pytest.raises(Exception):
            linreg_core.ols_regression(y, x, ["Intercept", "X1", "X2", "X3"])

        # But with sufficient data it should work
        y_more = [1.0, 2.0, 3.0, 4.0, 5.0]
        x_more = [[1.0, 2.0, 3.0, 4.0, 5.0],
                  [2.0, 3.5, 6.0, 8.5, 10.0]]  # Fixed: not perfectly collinear

        result = linreg_core.ols_regression(y_more, x_more, ["Intercept", "X1", "X2"])
        assert result.n_observations == 5


class TestSummaryAndReporting:
    """Tests for summary generation and reporting workflows."""

    def test_generate_model_comparison_report(self):
        """Test generating a comparison report across models."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 8.1, 9.2]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]

        # Fit multiple models
        ols = linreg_core.ols_regression(y, x, ["Intercept", "X1"])
        ridge = linreg_core.ridge_regression(y, x, lambda_val=0.1)

        # Generate summaries
        ols_summary = ols.summary()
        ridge_summary = ridge.summary()

        # Both should be strings
        assert isinstance(ols_summary, str)
        assert isinstance(ridge_summary, str)
        assert "OLS" in ols_summary
        assert "Ridge" in ridge_summary

    def test_to_dict_workflow(self):
        """Test converting results to dict for serialization."""
        y = [2.5, 3.7, 4.2, 5.1, 6.3]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        ols_result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        # Convert to dict
        result_dict = ols_result.to_dict()

        # Verify dict structure
        assert isinstance(result_dict, dict)
        assert "coefficients" in result_dict
        assert "r_squared" in result_dict
        assert "mse" in result_dict

        # Verify values can be accessed
        assert len(result_dict["coefficients"]) == 2

    def test_full_diagnostic_workflow(self):
        """Test running multiple diagnostics and collecting results."""
        y = [2.1, 4.3, 5.8, 8.2, 9.7, 12.1, 13.8, 16.2, 17.9, 20.3,
              22.1, 24.3, 25.8, 28.2, 29.7]
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
              11.0, 12.0, 13.0, 14.0, 15.0]
        x2 = [2.0, 4.0, 5.0, 4.0, 3.0, 2.0, 4.0, 5.0, 6.0, 5.0,
              4.0, 3.0, 2.0, 4.0, 5.0]

        # Run OLS
        ols = linreg_core.ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])

        # Run all diagnostic tests
        diagnostics = {
            "breusch_pagan": linreg_core.breusch_pagan_test(y, [x1, x2]),
            "durbin_watson": linreg_core.durbin_watson_test(y, [x1, x2]),
            "jarque_bera": linreg_core.jarque_bera_test(y, [x1, x2]),
            "shapiro_wilk": linreg_core.shapiro_wilk_test(y, [x1, x2]),
            "anderson_darling": linreg_core.anderson_darling_test(y, [x1, x2]),
        }

        # Verify all diagnostics completed
        for name, result in diagnostics.items():
            assert result is not None
            assert hasattr(result, "p_value") or hasattr(result, "statistic")
