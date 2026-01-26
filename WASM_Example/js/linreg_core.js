/* @ts-self-types="./linreg_core.d.ts" */

/**
 * Performs the Anderson-Darling test for normality via WASM.
 *
 * The Anderson-Darling test checks whether the residuals are normally distributed
 * by comparing the empirical distribution to the expected normal distribution.
 * This test is particularly sensitive to deviations in the tails of the distribution.
 * A significant p-value suggests that the residuals deviate from normality.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the A² statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function anderson_darling_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.anderson_darling_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Performs the Breusch-Godfrey test for higher-order serial correlation via WASM.
 *
 * Unlike the Durbin-Watson test which only detects first-order autocorrelation,
 * the Breusch-Godfrey test can detect serial correlation at any lag order.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `order` - Maximum order of serial correlation to test (default: 1)
 * * `test_type` - Type of test statistic: "chisq" or "f" (default: "chisq")
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, degrees of freedom, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {number} order
 * @param {string} test_type
 * @returns {string}
 */
export function breusch_godfrey_test(y_json, x_vars_json, order, test_type) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(test_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.breusch_godfrey_test(ptr0, len0, ptr1, len1, order, ptr2, len2);
        deferred4_0 = ret[0];
        deferred4_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Performs the Breusch-Pagan test for heteroscedasticity via WASM.
 *
 * The Breusch-Pagan test checks whether the variance of residuals is constant
 * across the range of predicted values (homoscedasticity assumption).
 * A significant p-value suggests heteroscedasticity.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function breusch_pagan_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.breusch_pagan_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Computes Cook's distance for identifying influential observations via WASM.
 *
 * Cook's distance measures how much each observation influences the regression
 * model by comparing coefficient estimates with and without that observation.
 * Unlike hypothesis tests, this is an influence measure - not a test with p-values.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing:
 * - Vector of Cook's distances (one per observation)
 * - Thresholds for identifying influential observations
 * - Indices of potentially influential observations
 * - Interpretation and guidance
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function cooks_distance_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.cooks_distance_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Performs the Durbin-Watson test for autocorrelation via WASM.
 *
 * The Durbin-Watson test checks for autocorrelation in the residuals.
 * Values near 2 indicate no autocorrelation, values near 0 suggest positive
 * autocorrelation, and values near 4 suggest negative autocorrelation.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the DW statistic and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function durbin_watson_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.durbin_watson_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Performs Elastic Net regression via WASM.
 *
 * Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `variable_names` - JSON array of variable names
 * * `lambda` - Regularization strength (>= 0)
 * * `alpha` - Elastic net mixing parameter (0 = Ridge, 1 = Lasso)
 * * `standardize` - Whether to standardize predictors (recommended: true)
 * * `max_iter` - Maximum coordinate descent iterations
 * * `tol` - Convergence tolerance
 *
 * # Returns
 *
 * JSON string containing regression results (same fields as Lasso).
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, parameters are invalid,
 * or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {string} _variable_names
 * @param {number} lambda
 * @param {number} alpha
 * @param {boolean} standardize
 * @param {number} max_iter
 * @param {number} tol
 * @returns {string}
 */
export function elastic_net_regression(y_json, x_vars_json, _variable_names, lambda, alpha, standardize, max_iter, tol) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(_variable_names, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.elastic_net_regression(ptr0, len0, ptr1, len1, ptr2, len2, lambda, alpha, standardize, max_iter, tol);
        deferred4_0 = ret[0];
        deferred4_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Computes the inverse of the standard normal CDF (probit function).
 *
 * Returns the z-score such that P(Z ≤ z) = p for a standard normal distribution.
 *
 * # Arguments
 *
 * * `p` - Probability (0 < p < 1)
 *
 * # Returns
 *
 * The z-score, or `NaN` if domain check fails.
 * @param {number} p
 * @returns {number}
 */
export function get_normal_inverse(p) {
    const ret = wasm.get_normal_inverse(p);
    return ret;
}

/**
 * Computes the Student's t-distribution cumulative distribution function.
 *
 * Returns P(T ≤ t) for a t-distribution with the given degrees of freedom.
 *
 * # Arguments
 *
 * * `t` - t-statistic value
 * * `df` - Degrees of freedom
 *
 * # Returns
 *
 * The CDF value, or `NaN` if domain check fails.
 * @param {number} t
 * @param {number} df
 * @returns {number}
 */
export function get_t_cdf(t, df) {
    const ret = wasm.get_t_cdf(t, df);
    return ret;
}

/**
 * Computes the critical t-value for a given significance level.
 *
 * Returns the t-value such that the area under the t-distribution curve
 * to the right equals alpha/2 (two-tailed test).
 *
 * # Arguments
 *
 * * `alpha` - Significance level (typically 0.05 for 95% confidence)
 * * `df` - Degrees of freedom
 *
 * # Returns
 *
 * The critical t-value, or `NaN` if domain check fails.
 * @param {number} alpha
 * @param {number} df
 * @returns {number}
 */
export function get_t_critical(alpha, df) {
    const ret = wasm.get_t_critical(alpha, df);
    return ret;
}

/**
 * Returns the current version of the library.
 *
 * Returns the Cargo package version as a string (e.g., "0.1.0").
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 * @returns {string}
 */
export function get_version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.get_version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Performs the Harvey-Collier test for linearity via WASM.
 *
 * The Harvey-Collier test checks whether the residuals exhibit a linear trend,
 * which would indicate that the model's functional form is misspecified.
 * A significant p-value suggests non-linearity.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function harvey_collier_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.harvey_collier_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Performs the Jarque-Bera test for normality via WASM.
 *
 * The Jarque-Bera test checks whether the residuals are normally distributed
 * by examining skewness and kurtosis. A significant p-value suggests that
 * the residuals deviate from normality.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function jarque_bera_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.jarque_bera_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Performs Lasso regression via WASM.
 *
 * Lasso regression adds an L1 penalty to the coefficients, which performs
 * automatic variable selection by shrinking some coefficients to exactly zero.
 * The intercept is never penalized.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `variable_names` - JSON array of variable names
 * * `lambda` - Regularization strength (>= 0, typical range 0.01 to 10)
 * * `standardize` - Whether to standardize predictors (recommended: true)
 * * `max_iter` - Maximum coordinate descent iterations (default: 100000)
 * * `tol` - Convergence tolerance (default: 1e-7)
 *
 * # Returns
 *
 * JSON string containing:
 * - `lambda` - Lambda value used
 * - `intercept` - Intercept coefficient
 * - `coefficients` - Slope coefficients (some may be exactly zero)
 * - `fitted_values` - Predictions on training data
 * - `residuals` - Residuals (y - fitted_values)
 * - `n_nonzero` - Number of non-zero coefficients (excluding intercept)
 * - `iterations` - Number of coordinate descent iterations
 * - `converged` - Whether the algorithm converged
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, lambda is negative,
 * or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {string} _variable_names
 * @param {number} lambda
 * @param {boolean} standardize
 * @param {number} max_iter
 * @param {number} tol
 * @returns {string}
 */
export function lasso_regression(y_json, x_vars_json, _variable_names, lambda, standardize, max_iter, tol) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(_variable_names, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.lasso_regression(ptr0, len0, ptr1, len1, ptr2, len2, lambda, standardize, max_iter, tol);
        deferred4_0 = ret[0];
        deferred4_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Generates a lambda path for regularized regression via WASM.
 *
 * Creates a logarithmically-spaced sequence of lambda values from lambda_max
 * (where all penalized coefficients are zero) down to lambda_min. This is
 * useful for fitting regularization paths and selecting optimal lambda via
 * cross-validation.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `n_lambda` - Number of lambda values to generate (default: 100)
 * * `lambda_min_ratio` - Ratio for smallest lambda (default: 0.0001 if n >= p, else 0.01)
 *
 * # Returns
 *
 * JSON string containing:
 * - `lambda_path` - Array of lambda values in decreasing order
 * - `lambda_max` - Maximum lambda value
 * - `lambda_min` - Minimum lambda value
 * - `n_lambda` - Number of lambda values
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {number} n_lambda
 * @param {number} lambda_min_ratio
 * @returns {string}
 */
export function make_lambda_path(y_json, x_vars_json, n_lambda, lambda_min_ratio) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.make_lambda_path(ptr0, len0, ptr1, len1, n_lambda, lambda_min_ratio);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Performs OLS regression via WASM.
 *
 * All parameters and return values are JSON-encoded strings for JavaScript
 * interoperability. Returns regression output including coefficients,
 * standard errors, diagnostic statistics, and VIF analysis.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values: `[1.0, 2.0, 3.0]`
 * * `x_vars_json` - JSON array of predictor arrays: `[[1.0, 2.0], [0.5, 1.0]]`
 * * `variable_names` - JSON array of variable names: `["Intercept", "X1", "X2"]`
 *
 * # Returns
 *
 * JSON string containing the complete regression output with coefficients,
 * standard errors, t-statistics, p-values, R², F-statistic, residuals, leverage, VIF, etc.
 *
 * # Errors
 *
 * Returns a JSON error object if:
 * - JSON parsing fails
 * - Insufficient data (n ≤ k + 1)
 * - Matrix is singular
 * - Domain check fails
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {string} variable_names
 * @returns {string}
 */
export function ols_regression(y_json, x_vars_json, variable_names) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(variable_names, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.ols_regression(ptr0, len0, ptr1, len1, ptr2, len2);
        deferred4_0 = ret[0];
        deferred4_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Parses CSV data and returns it as a JSON string.
 *
 * Parses the CSV content and identifies numeric columns. Returns a JSON object
 * with headers, data rows, and a list of numeric column names.
 *
 * # Arguments
 *
 * * `content` - CSV content as a string
 *
 * # Returns
 *
 * JSON string with structure:
 * ```json
 * {
 *   "headers": ["col1", "col2", ...],
 *   "data": [{"col1": 1.0, "col2": "text"}, ...],
 *   "numeric_columns": ["col1", ...]
 * }
 * ```
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} content
 * @returns {string}
 */
export function parse_csv(content) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(content, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.parse_csv(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Performs the Python method White test for heteroscedasticity via WASM.
 *
 * This implementation matches Python's `statsmodels.stats.diagnostic.het_white()` function.
 * Uses the LINPACK QR decomposition with column pivoting and the Python-specific
 * auxiliary matrix structure (intercept, X, X², and cross-products).
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays (each array is a column)
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function python_white_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.python_white_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Performs the R method White test for heteroscedasticity via WASM.
 *
 * This implementation matches R's `skedastic::white()` function behavior.
 * Uses the standard QR decomposition and the R-specific auxiliary matrix
 * structure (intercept, X, X² only - no cross-products).
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays (each array is a column)
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function r_white_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.r_white_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Performs the Rainbow test for linearity via WASM.
 *
 * The Rainbow test checks whether the relationship between predictors and response
 * is linear. A significant p-value suggests non-linearity.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `fraction` - Fraction of data to use in the central subset (0.0 to 1.0, typically 0.5)
 * * `method` - Method to use: "r", "python", or "both" (case-insensitive, defaults to "r")
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {number} fraction
 * @param {string} method
 * @returns {string}
 */
export function rainbow_test(y_json, x_vars_json, fraction, method) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(method, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.rainbow_test(ptr0, len0, ptr1, len1, fraction, ptr2, len2);
        deferred4_0 = ret[0];
        deferred4_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Performs the RESET test for model specification error via WASM.
 *
 * The RESET (Regression Specification Error Test) test checks whether the model
 * is correctly specified by testing if additional terms (powers of fitted values,
 * regressors, or first principal component) significantly improve the model fit.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `powers_json` - JSON array of powers to use (e.g., [2, 3] for ŷ², ŷ³)
 * * `type_` - Type of terms to add: "fitted", "regressor", or "princomp"
 *
 * # Returns
 *
 * JSON string containing the F-statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {string} powers_json
 * @param {string} type_
 * @returns {string}
 */
export function reset_test(y_json, x_vars_json, powers_json, type_) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(powers_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(type_, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.reset_test(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
        deferred5_0 = ret[0];
        deferred5_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}

/**
 * Performs Ridge regression via WASM.
 *
 * Ridge regression adds an L2 penalty to the coefficients, which helps with
 * multicollinearity and overfitting. The intercept is never penalized.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `variable_names` - JSON array of variable names
 * * `lambda` - Regularization strength (>= 0, typical range 0.01 to 100)
 * * `standardize` - Whether to standardize predictors (recommended: true)
 *
 * # Returns
 *
 * JSON string containing:
 * - `lambda` - Lambda value used
 * - `intercept` - Intercept coefficient
 * - `coefficients` - Slope coefficients
 * - `fitted_values` - Predictions on training data
 * - `residuals` - Residuals (y - fitted_values)
 * - `df` - Effective degrees of freedom
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails, lambda is negative,
 * or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {string} _variable_names
 * @param {number} lambda
 * @param {boolean} standardize
 * @returns {string}
 */
export function ridge_regression(y_json, x_vars_json, _variable_names, lambda, standardize) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(_variable_names, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.ridge_regression(ptr0, len0, ptr1, len1, ptr2, len2, lambda, standardize);
        deferred4_0 = ret[0];
        deferred4_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

/**
 * Performs the Shapiro-Wilk test for normality via WASM.
 *
 * The Shapiro-Wilk test is a powerful test for normality,
 * especially for small to moderate sample sizes (3 ≤ n ≤ 5000). It tests
 * the null hypothesis that the residuals are normally distributed.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 *
 * # Returns
 *
 * JSON string containing the W statistic (ranges from 0 to 1), p-value,
 * and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @returns {string}
 */
export function shapiro_wilk_test(y_json, x_vars_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.shapiro_wilk_test(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Computes the correlation coefficient between two JSON arrays of f64 values.
 *
 * # Arguments
 *
 * * `x_json` - JSON string representing the first array of f64 values
 * * `y_json` - JSON string representing the second array of f64 values
 *
 * # Returns
 *
 * JSON string with the correlation coefficient, or "null" if input is invalid
 * @param {string} x_json
 * @param {string} y_json
 * @returns {string}
 */
export function stats_correlation(x_json, y_json) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(x_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.stats_correlation(ptr0, len0, ptr1, len1);
        deferred3_0 = ret[0];
        deferred3_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Computes the arithmetic mean of a JSON array of f64 values.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 *
 * # Returns
 *
 * JSON string with the mean, or "null" if input is invalid/empty
 * @param {string} data_json
 * @returns {string}
 */
export function stats_mean(data_json) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(data_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.stats_mean(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Computes the median of a JSON array of f64 values.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 *
 * # Returns
 *
 * JSON string with the median, or "null" if input is invalid/empty
 * @param {string} data_json
 * @returns {string}
 */
export function stats_median(data_json) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(data_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.stats_median(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Computes a quantile of a JSON array of f64 values.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 * * `q` - Quantile to calculate (0.0 to 1.0)
 *
 * # Returns
 *
 * JSON string with the quantile value, or "null" if input is invalid
 * @param {string} data_json
 * @param {number} q
 * @returns {string}
 */
export function stats_quantile(data_json, q) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(data_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.stats_quantile(ptr0, len0, q);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Computes the sample standard deviation of a JSON array of f64 values.
 *
 * Uses the (n-1) denominator for unbiased estimation.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 *
 * # Returns
 *
 * JSON string with the standard deviation, or "null" if input is invalid
 * @param {string} data_json
 * @returns {string}
 */
export function stats_stddev(data_json) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(data_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.stats_stddev(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Computes the sample variance of a JSON array of f64 values.
 *
 * Uses the (n-1) denominator for unbiased estimation.
 *
 * # Arguments
 *
 * * `data_json` - JSON string representing an array of f64 values
 *
 * # Returns
 *
 * JSON string with the variance, or "null" if input is invalid
 * @param {string} data_json
 * @returns {string}
 */
export function stats_variance(data_json) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(data_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.stats_variance(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Simple test function to verify WASM is working.
 *
 * Returns a success message confirming the WASM module loaded correctly.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 * @returns {string}
 */
export function test() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.test();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Test function for confidence interval computation.
 *
 * Returns JSON with the computed confidence interval for a coefficient.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 * @param {number} coef
 * @param {number} se
 * @param {number} df
 * @param {number} alpha
 * @returns {string}
 */
export function test_ci(coef, se, df, alpha) {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.test_ci(coef, se, df, alpha);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Test function for regression validation against R reference values.
 *
 * Runs a regression on a housing dataset and compares results against R's lm() output.
 * Returns JSON with status "PASS" or "FAIL" with details.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 * @returns {string}
 */
export function test_housing_regression() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.test_housing_regression();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Test function for R accuracy validation.
 *
 * Returns JSON comparing our statistical functions against R reference values.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 * @returns {string}
 */
export function test_r_accuracy() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.test_r_accuracy();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Test function for t-critical value computation.
 *
 * Returns JSON with the computed t-critical value for the given parameters.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 * @param {number} df
 * @param {number} alpha
 * @returns {string}
 */
export function test_t_critical(df, alpha) {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.test_t_critical(df, alpha);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Performs the White test for heteroscedasticity via WASM.
 *
 * The White test is a more general test for heteroscedasticity that does not
 * assume a specific form of heteroscedasticity. A significant p-value suggests
 * that the error variance is not constant.
 *
 * # Arguments
 *
 * * `y_json` - JSON array of response variable values
 * * `x_vars_json` - JSON array of predictor arrays
 * * `method` - Method to use: "r", "python", or "both" (case-insensitive, defaults to "r")
 *
 * # Returns
 *
 * JSON string containing test statistic, p-value, and interpretation.
 *
 * # Errors
 *
 * Returns a JSON error object if parsing fails or domain check fails.
 * @param {string} y_json
 * @param {string} x_vars_json
 * @param {string} method
 * @returns {string}
 */
export function white_test(y_json, x_vars_json, method) {
    let deferred4_0;
    let deferred4_1;
    try {
        const ptr0 = passStringToWasm0(y_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_vars_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(method, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.white_test(ptr0, len0, ptr1, len1, ptr2, len2);
        deferred4_0 = ret[0];
        deferred4_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
    }
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./linreg_core_bg.js": import0,
    };
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('linreg_core_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
