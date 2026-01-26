/* tslint:disable */
/* eslint-disable */

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
 */
export function anderson_darling_test(y_json: string, x_vars_json: string): string;

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
 */
export function breusch_godfrey_test(y_json: string, x_vars_json: string, order: number, test_type: string): string;

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
 */
export function breusch_pagan_test(y_json: string, x_vars_json: string): string;

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
 */
export function cooks_distance_test(y_json: string, x_vars_json: string): string;

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
 */
export function durbin_watson_test(y_json: string, x_vars_json: string): string;

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
 */
export function elastic_net_regression(y_json: string, x_vars_json: string, _variable_names: string, lambda: number, alpha: number, standardize: boolean, max_iter: number, tol: number): string;

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
 */
export function get_normal_inverse(p: number): number;

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
 */
export function get_t_cdf(t: number, df: number): number;

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
 */
export function get_t_critical(alpha: number, df: number): number;

/**
 * Returns the current version of the library.
 *
 * Returns the Cargo package version as a string (e.g., "0.1.0").
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function get_version(): string;

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
 */
export function harvey_collier_test(y_json: string, x_vars_json: string): string;

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
 */
export function jarque_bera_test(y_json: string, x_vars_json: string): string;

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
 */
export function lasso_regression(y_json: string, x_vars_json: string, _variable_names: string, lambda: number, standardize: boolean, max_iter: number, tol: number): string;

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
 */
export function make_lambda_path(y_json: string, x_vars_json: string, n_lambda: number, lambda_min_ratio: number): string;

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
 */
export function ols_regression(y_json: string, x_vars_json: string, variable_names: string): string;

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
 */
export function parse_csv(content: string): string;

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
 */
export function python_white_test(y_json: string, x_vars_json: string): string;

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
 */
export function r_white_test(y_json: string, x_vars_json: string): string;

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
 */
export function rainbow_test(y_json: string, x_vars_json: string, fraction: number, method: string): string;

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
 */
export function reset_test(y_json: string, x_vars_json: string, powers_json: string, type_: string): string;

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
 */
export function ridge_regression(y_json: string, x_vars_json: string, _variable_names: string, lambda: number, standardize: boolean): string;

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
 */
export function shapiro_wilk_test(y_json: string, x_vars_json: string): string;

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
 */
export function stats_correlation(x_json: string, y_json: string): string;

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
 */
export function stats_mean(data_json: string): string;

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
 */
export function stats_median(data_json: string): string;

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
 */
export function stats_quantile(data_json: string, q: number): string;

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
 */
export function stats_stddev(data_json: string): string;

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
 */
export function stats_variance(data_json: string): string;

/**
 * Simple test function to verify WASM is working.
 *
 * Returns a success message confirming the WASM module loaded correctly.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test(): string;

/**
 * Test function for confidence interval computation.
 *
 * Returns JSON with the computed confidence interval for a coefficient.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test_ci(coef: number, se: number, df: number, alpha: number): string;

/**
 * Test function for regression validation against R reference values.
 *
 * Runs a regression on a housing dataset and compares results against R's lm() output.
 * Returns JSON with status "PASS" or "FAIL" with details.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test_housing_regression(): string;

/**
 * Test function for R accuracy validation.
 *
 * Returns JSON comparing our statistical functions against R reference values.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test_r_accuracy(): string;

/**
 * Test function for t-critical value computation.
 *
 * Returns JSON with the computed t-critical value for the given parameters.
 *
 * # Errors
 *
 * Returns a JSON error object if domain check fails.
 */
export function test_t_critical(df: number, alpha: number): string;

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
 */
export function white_test(y_json: string, x_vars_json: string, method: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly anderson_darling_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly breusch_godfrey_test: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly breusch_pagan_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly cooks_distance_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly durbin_watson_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly elastic_net_regression: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number];
    readonly get_version: () => [number, number];
    readonly harvey_collier_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly jarque_bera_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly lasso_regression: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number];
    readonly make_lambda_path: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly ols_regression: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly parse_csv: (a: number, b: number) => [number, number];
    readonly python_white_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly r_white_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly rainbow_test: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly reset_test: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly ridge_regression: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly shapiro_wilk_test: (a: number, b: number, c: number, d: number) => [number, number];
    readonly stats_correlation: (a: number, b: number, c: number, d: number) => [number, number];
    readonly stats_mean: (a: number, b: number) => [number, number];
    readonly stats_median: (a: number, b: number) => [number, number];
    readonly stats_quantile: (a: number, b: number, c: number) => [number, number];
    readonly stats_stddev: (a: number, b: number) => [number, number];
    readonly stats_variance: (a: number, b: number) => [number, number];
    readonly test: () => [number, number];
    readonly test_ci: (a: number, b: number, c: number, d: number) => [number, number];
    readonly test_housing_regression: () => [number, number];
    readonly test_r_accuracy: () => [number, number];
    readonly test_t_critical: (a: number, b: number) => [number, number];
    readonly white_test: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly get_t_cdf: (a: number, b: number) => number;
    readonly get_t_critical: (a: number, b: number) => number;
    readonly get_normal_inverse: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
