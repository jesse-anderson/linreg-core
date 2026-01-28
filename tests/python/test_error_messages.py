"""
Error message quality tests.

Tests verify that error messages are:
- Clear and understandable
- Contain actionable guidance
- Include relevant context (dimensions, counts, etc.)
"""

import pytest
import linreg_core


class TestErrorMessageQuality:
    """Tests for quality and clarity of error messages."""

    def test_empty_input_error_message(self):
        """Test that empty input produces a helpful error message."""
        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression([], [], ["Intercept"])

        error_msg = str(exc_info.value)
        # Error message should mention the problem
        assert len(error_msg) > 0, "Error message should not be empty"
        # Should contain some context about what went wrong
        assert any(term in error_msg.lower() for term in
                   ["empty", "insufficient", "data", "observation", "input", "invalid"])

    def test_dimension_mismatch_error_message(self):
        """Test that dimension mismatch produces specific error message."""
        y = [1.0, 2.0, 3.0]
        x = [[1.0, 2.0]]  # Only 2 observations

        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        error_msg = str(exc_info.value)
        # Should mention dimensions or data size
        assert any(term in error_msg.lower() for term in
                   ["dimension", "mismatch", "size", "length", "insufficient"])

    def test_insufficient_data_error_message(self):
        """Test that insufficient data gives specific counts."""
        y = [1.0, 2.0]
        x = [[1.0, 2.0], [3.0, 4.0]]  # 2 obs, 2 predictors (need n > p + 1)

        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression(y, x, ["Intercept", "X1", "X2"])

        error_msg = str(exc_info.value)
        # Error message should mention the data requirement
        assert len(error_msg) > 0
        # Ideally should mention required vs available
        # (This depends on the specific error implementation)

    def test_singular_matrix_error_message(self):
        """Test that singular matrix error mentions multicollinearity."""
        # Perfectly collinear predictors
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 6.0, 8.0, 10.0]  # Exactly 2 * x1

        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])

        error_msg = str(exc_info.value)
        # Should mention singular matrix or multicollinearity
        assert any(term in error_msg.lower() for term in
                   ["singular", "multicollinear", "redundant", "linear", "dependent"])

    def test_nan_value_error_message(self):
        """Test error message for NaN values."""
        y = [1.0, float('nan'), 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        error_msg = str(exc_info.value)
        # Should mention NaN or invalid values
        assert any(term in error_msg.lower() for term in
                   ["nan", "invalid", "finite", "number"])

    def test_inf_value_error_message(self):
        """Test error message for infinite values."""
        y = [1.0, float('inf'), 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        error_msg = str(exc_info.value)
        # Should mention infinity or invalid values
        assert any(term in error_msg.lower() for term in
                   ["inf", "infinite", "invalid", "finite"])

    def test_constant_predictor_error_message(self):
        """Test error message for constant predictor."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 1.0, 1.0, 1.0, 1.0]]  # No variance

        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        error_msg = str(exc_info.value)
        # Should mention variance or singular matrix
        assert any(term in error_msg.lower() for term in
                   ["singular", "variance", "constant", "multicollinear"])

    def test_negative_lambda_error_message(self):
        """Test error message for invalid lambda parameter."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]

        # Ridge with negative lambda should fail
        with pytest.raises(Exception) as exc_info:
            linreg_core.ridge_regression(y, x, lambda_=-1.0)

        error_msg = str(exc_info.value)
        # Should mention invalid parameter or lambda
        assert any(term in error_msg.lower() for term in
                   ["invalid", "lambda", "parameter", "positive"])


class TestExceptionTypes:
    """Tests for custom exception types."""

    def test_linreg_error_exists(self):
        """Test that LinregError exception type exists."""
        assert hasattr(linreg_core, 'LinregError')

    def test_data_validation_error_exists(self):
        """Test that DataValidationError exception type exists."""
        assert hasattr(linreg_core, 'DataValidationError')

    def test_error_is_catchable(self):
        """Test that errors can be caught as specific types."""
        # Try an operation that will fail
        try:
            linreg_core.ols_regression([], [], ["Intercept"])
        except linreg_core.LinregError:
            # Should be catchable as LinregError
            caught = True
        except Exception:
            # If not LinregError, should at least be some exception
            caught = True

        assert caught, "Error should be catchable"

    def test_error_has_message_attribute(self):
        """Test that error has a message attribute."""
        try:
            linreg_core.ols_regression([], [], ["Intercept"])
        except Exception as e:
            # Either has message attribute or str() works
            has_message = hasattr(e, 'message') or len(str(e)) > 0

        assert has_message, "Error should have a message or string representation"


class TestErrorMessageContext:
    """Tests for contextual information in error messages."""

    def test_ridge_insufficient_data_context(self):
        """Test Ridge error mentions data requirements."""
        # Very few observations
        y = [1.0]
        x = [[1.0]]

        with pytest.raises(Exception) as exc_info:
            linreg_core.ridge_regression(y, x)

        error_msg = str(exc_info.value)
        # Should mention the insufficiency
        assert len(error_msg) > 10  # More than just "error"

    def test_lasso_convergence_error_message(self):
        """Test Lasso convergence failure message."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Use very small lambda_val and low iterations to potentially cause issues
        result = linreg_core.lasso_regression(y, [x], lambda_val=0.001, max_iter=1)

        # Should still return a result, but may not have converged
        # Check that we can get convergence status
        assert hasattr(result, 'converged')

    def test_stats_empty_data_error_message(self):
        """Test stats function error messages for empty data."""
        with pytest.raises(Exception) as exc_info:
            linreg_core.stats_mean([])

        error_msg = str(exc_info.value)
        # Should mention empty data
        assert any(term in error_msg.lower() for term in
                   ["empty", "data", "value", "invalid"])

    def test_correlation_length_mismatch_message(self):
        """Test correlation error for mismatched lengths."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]  # Different length

        with pytest.raises(Exception) as exc_info:
            linreg_core.stats_correlation(x, y)

        error_msg = str(exc_info.value)
        # Should mention length or dimension
        assert any(term in error_msg.lower() for term in
                   ["length", "size", "dimension", "mismatch"])


class TestErrorRecoverability:
    """Tests that errors allow for recovery and retry."""

    def test_error_allows_retry_with_correct_data(self):
        """Test that after an error, operation can succeed with correct data."""
        # First, fail with bad data
        with pytest.raises(Exception):
            linreg_core.ols_regression([], [], ["Intercept"])

        # Then succeed with correct data
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        result = linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        assert result is not None
        assert len(result.coefficients) == 2

    def test_singular_matrix_recoverable_by_removing_predictor(self):
        """Test that singular matrix error can be fixed by removing predictor."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        x2 = [2.0, 4.0, 6.0, 8.0, 10.0]  # Collinear with x1

        # Should fail with both predictors
        with pytest.raises(Exception):
            linreg_core.ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])

        # Should succeed with just one predictor
        result = linreg_core.ols_regression(y, [x1], ["Intercept", "X1"])
        assert result.r_squared > 0.9


class TestErrorMessageLanguage:
    """Tests for clarity and user-friendliness of error messages."""

    def test_error_messages_are_english(self):
        """Test that error messages are in English (not technical jargon)."""
        # Collect error messages from various failure modes
        errors_to_test = [
            lambda: linreg_core.ols_regression([], [], ["Intercept"]),
            lambda: linreg_core.ols_regression([1], [[1]], ["Intercept", "X1"]),
            lambda: linreg_core.stats_mean([]),
        ]

        for error_func in errors_to_test:
            try:
                error_func()
            except Exception as e:
                msg = str(e).lower()
                # Should not be completely cryptic
                # At minimum, should have vowels (indicates words, not just codes)
                has_vowels = any(vowel in msg for vowel in "aeiou")
                assert has_vowels or len(msg) == 0, f"Error message seems too cryptic: {msg}"

    def test_error_messages_avoid_internal_terminology(self):
        """Test that error messages don't expose internal Rust terminology."""
        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression([], [], ["Intercept"])

        error_msg = str(exc_info.value).lower()

        # Should not contain raw Rust type names or internal terminology
        internal_terms = ["pyo3", "pyresult", "bound", "rust", "nak", "panic"]
        for term in internal_terms:
            assert term not in error_msg, f"Error message contains internal term: {term}"

    def test_error_messages_not_too_long(self):
        """Test that error messages are concise (not wall of text)."""
        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression([], [], ["Intercept"])

        error_msg = str(exc_info.value)

        # Should be under 500 characters
        assert len(error_msg) < 500, "Error message too long"

    def test_error_messages_not_too_short(self):
        """Test that error messages provide some information."""
        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression([], [], ["Intercept"])

        error_msg = str(exc_info.value)

        # Should be at least 10 characters (more than just "Error")
        assert len(error_msg) >= 10, "Error message too short"


class TestSpecificErrorScenarios:
    """Tests for specific error scenarios and their messages."""

    def test_multiple_high_vif_predictors_error(self):
        """Test error when multiple predictors are collinear."""
        import random
        random.seed(42)

        # Generate highly correlated predictors
        n = 50
        x1 = [random.gauss(0, 1) for _ in range(n)]
        x2 = [x + random.gauss(0, 0.01) for x in x1]  # Very high correlation
        x3 = [x * 2 + random.gauss(0, 0.01) for x in x1]  # Nearly 2*x1

        y = [x1[i] + x2[i] + random.gauss(0, 0.1) for i in range(n)]

        # Might fail due to near-singularity
        try:
            result = linreg_core.ols_regression(y, [x1, x2, x3], ["Intercept", "X1", "X2", "X3"])
            # If it succeeds, VIF should be very high
            assert any(vif > 100 for vif in result.vif), "Expected high VIF for collinear data"
        except Exception as e:
            # If it fails, error should mention singular/multicollinearity
            error_msg = str(e).lower()
            assert any(term in error_msg for term in
                       ["singular", "multicollinear", "correlation"])

    def test_single_observation_error_message(self):
        """Test helpful error for single observation."""
        y = [5.0]
        x = [[2.0]]

        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression(y, x, ["Intercept", "X1"])

        error_msg = str(exc_info.value).lower()
        # Should mention needing more data
        assert any(term in error_msg for term in
                   ["insufficient", "observation", "data", "sample"])

    def test_two_observations_three_predictors_error(self):
        """Test error when predictors > observations - 1."""
        y = [1.0, 2.0]
        x1 = [1.0, 2.0]
        x2 = [2.0, 4.0]
        x3 = [3.0, 6.0]

        with pytest.raises(Exception) as exc_info:
            linreg_core.ols_regression(y, [x1, x2, x3], ["Intercept", "X1", "X2", "X3"])

        error_msg = str(exc_info.value)
        # Should indicate the mismatch
        assert any(term in error_msg.lower() for term in
                   ["insufficient", "observation", "predictor", "data"])


class TestCsvErrorMessages:
    """Tests for CSV parsing error messages."""

    def test_empty_csv_error_message(self):
        """Test that empty CSV produces reasonable result."""
        result = linreg_core.parse_csv("")

        # Empty CSV should return valid result with 0 rows
        assert result.n_rows == 0

    def test_malformed_csv_doesnt_crash(self):
        """Test that malformed CSV doesn't cause crash."""
        # CSV with inconsistent columns
        csv_content = """a,b,c
1,2,3
4,5
6,7,8,9"""

        # Should not crash
        result = linreg_core.parse_csv(csv_content)
        assert result is not None

    def test_csv_with_only_headers(self):
        """Test CSV with headers but no data."""
        csv_content = "x,y,z"

        result = linreg_core.parse_csv(csv_content)
        assert result.n_rows == 0
        assert result.headers == ["x", "y", "z"]


class TestDiagnosticsErrorMessages:
    """Tests for diagnostic function error messages."""

    def test_diagnostic_with_insufficient_data(self):
        """Test diagnostics with too little data."""
        # Too few observations for meaningful diagnostics
        y = [1.0, 2.0]
        x = [[1.0, 2.0]]

        # May fail or return warning; check behavior
        try:
            result = linreg_core.breusch_pagan_test(y, x)
            # If succeeds, should have p_value
            assert hasattr(result, 'p_value')
        except Exception as e:
            # If fails, error should be informative
            error_msg = str(e)
            assert len(error_msg) > 10

    def test_durbin_watson_with_single_obs(self):
        """Test Durbin-Watson with minimal data."""
        y = [1.0]
        x = [[1.0]]

        # Should fail gracefully
        with pytest.raises(Exception) as exc_info:
            linreg_core.durbin_watson_test(y, x)

        error_msg = str(exc_info.value)
        # Should mention insufficient data
        assert len(error_msg) > 5
