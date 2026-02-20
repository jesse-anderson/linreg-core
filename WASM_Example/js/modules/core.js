// ============================================================================
// core.js - Core WASM integration and regression functions
// ============================================================================

import init, {
    ols_regression,
    ridge_regression,
    lasso_regression,
    elastic_net_regression,
    wls_regression,
    make_lambda_path,
    get_t_critical,
    get_normal_inverse,
    parse_csv,
    rainbow_test,
    white_test,
    harvey_collier_test,
    breusch_pagan_test,
    jarque_bera_test,
    durbin_watson_test,
    shapiro_wilk_test,
    anderson_darling_test,
    cooks_distance_test,
    dfbetas_test,
    dffits_test,
    reset_test,
    breusch_godfrey_test,
    vif_test,
    get_version,
    stats_mean,
    stats_stddev,
    stats_variance,
    stats_median,
    stats_quantile,
    stats_correlation,
    loess_fit,
    loess_predict,
    serialize_model,
    deserialize_model,
    get_model_metadata,
    elastic_net_path_wasm,
    kfold_cv_ols,
    kfold_cv_ridge,
    kfold_cv_lasso,
    kfold_cv_elastic_net,
    ols_prediction_intervals,
    ridge_prediction_intervals,
    lasso_prediction_intervals,
    elastic_net_prediction_intervals,
    polynomial_regression_wasm,
    polynomial_predict_wasm,
    polynomial_ridge_wasm,
    polynomial_lasso_wasm,
    polynomial_elastic_net_wasm,
    standardized_coefficients,
    shap_values_linear,
    permutation_importance_ols,
    vif_ranking,
    feature_importance_ols
} from '../linreg_core.js';

import { STATE, showToast } from './utils.js';

// ============================================================================
// WASM STATE
// ============================================================================

let wasmReady = false;
let wasmError = null;
let wasmVersion = null;

/**
 * Initialize the WASM module
 * @returns {Promise<void>}
 */
export async function initWasm() {
    try {
        await init();
        wasmReady = true;
        wasmVersion = get_version();
        console.log(`%c[linreg-core] v${wasmVersion}`, 'color: #16a34a; font-weight: bold;');
        console.log('[linreg-core] Rust WASM engine loaded successfully');
    } catch (e) {
        wasmError = e;
        console.error('[linreg-core] Failed to load WASM:', e);
    }
}

/**
 * Check if WASM is ready
 * @returns {boolean}
 */
export function isWasmReady() {
    return wasmReady;
}

/**
 * Get WASM error if any
 * @returns {Error|null}
 */
export function getWasmError() {
    return wasmError;
}

/**
 * Get WASM version
 * @returns {string|null}
 */
export function getWasmVersion() {
    return wasmVersion;
}

// ============================================================================
// WASM REGRESSION API WRAPPER
// ============================================================================

export const WasmRegression = {
    isReady: () => wasmReady,
    getError: () => wasmError,
    getVersion: () => wasmVersion,

    // OLS Regression
    ols: (y, xVars, names) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const namesJson = JSON.stringify(names);
        return ols_regression(yJson, xJson, namesJson);
    },

    // Ridge Regression
    ridge: (y, xVars, names, lambda, standardize = true) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const namesJson = JSON.stringify(names);
        return ridge_regression(yJson, xJson, namesJson, lambda, standardize);
    },

    // Lasso Regression
    lasso: (y, xVars, names, lambda, standardize = true, maxIter = 1000, tol = 1e-7) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const namesJson = JSON.stringify(names);
        return lasso_regression(yJson, xJson, namesJson, lambda, standardize, maxIter, tol);
    },

    // Elastic Net Regression
    elasticNet: (y, xVars, names, lambda, alpha, standardize = true, maxIter = 1000, tol = 1e-7) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const namesJson = JSON.stringify(names);
        return elastic_net_regression(yJson, xJson, namesJson, lambda, alpha, standardize, maxIter, tol);
    },

    // WLS (Weighted Least Squares) Regression
    wls: (y, xVars, weights) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const weightsJson = JSON.stringify(weights);
        return wls_regression(yJson, xJson, weightsJson);
    },

    // Polynomial OLS Regression
    polynomialOls: (y, x, degree, center, standardize) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(x);
        return polynomial_regression_wasm(yJson, xJson, degree, center, standardize);
    },

    // Polynomial Predictions
    polynomialPredict: (fitJson, xNew) => {
        const xNewJson = JSON.stringify(xNew);
        return polynomial_predict_wasm(fitJson, xNewJson);
    },

    // Polynomial Ridge Regression
    polynomialRidge: (y, x, degree, lambda, center, standardize) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(x);
        return polynomial_ridge_wasm(yJson, xJson, degree, lambda, center, standardize);
    },

    // Polynomial Lasso Regression
    polynomialLasso: (y, x, degree, lambda, center, standardize) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(x);
        return polynomial_lasso_wasm(yJson, xJson, degree, lambda, center, standardize);
    },

    // Polynomial Elastic Net Regression
    polynomialElasticNet: (y, x, degree, lambda, alpha, center, standardize) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(x);
        return polynomial_elastic_net_wasm(yJson, xJson, degree, lambda, alpha, center, standardize);
    },

    // Lambda Path Generation
    lambdaPath: (y, xVars, nLambda = 100, lambdaMinRatio = 0.0001) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return make_lambda_path(yJson, xJson, nLambda, lambdaMinRatio);
    },

    // CSV Parsing
    parseCsv: (content) => parse_csv(content),

    // Diagnostic Tests
    rainbowTest: (y, xVars, fraction = 0.5, method = 'r') => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return rainbow_test(yJson, xJson, fraction, method);
    },

    whiteTest: (y, xVars, method = 'r') => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return white_test(yJson, xJson, method);
    },

    harveyCollierTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return harvey_collier_test(yJson, xJson);
    },

    breuschPaganTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return breusch_pagan_test(yJson, xJson);
    },

    jarqueBeraTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return jarque_bera_test(yJson, xJson);
    },

    durbinWatsonTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return durbin_watson_test(yJson, xJson);
    },

    shapiroWilkTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return shapiro_wilk_test(yJson, xJson);
    },

    andersonDarlingTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return anderson_darling_test(yJson, xJson);
    },

    cooksDistanceTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return cooks_distance_test(yJson, xJson);
    },

    // New Diagnostic Tests
    resetTest: (y, xVars, powers = [2, 3], type_ = 'fitted') => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const powersJson = JSON.stringify(powers);
        return reset_test(yJson, xJson, powersJson, type_);
    },

    breuschGodfreyTest: (y, xVars, order = 1, testType = 'chisq') => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return breusch_godfrey_test(yJson, xJson, order, testType);
    },

    // Influence Diagnostic Tests
    dfbetasTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return dfbetas_test(yJson, xJson);
    },

    dffitsTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return dffits_test(yJson, xJson);
    },

    vifTest: (y, xVars) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return vif_test(yJson, xJson);
    },

    // LOESS Regression
    loess: (y, xVars, span = 0.75, degree = 1, robustIterations = 0, surface = 'direct') => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return loess_fit(yJson, xJson, span, degree, robustIterations, surface);
    },

    // LOESS Predictions at new points
    loessPredict: (newX, originalX, originalY, span, degree, robustIterations, surface) => {
        const newXJson = JSON.stringify(newX);
        const origXJson = JSON.stringify(originalX);
        const origYJson = JSON.stringify(originalY);
        return loess_predict(newXJson, origXJson, origYJson, span, degree, robustIterations, surface);
    },

    // Model Serialization
    serializeModel: async (modelData, modelType, name = null) => {
        await initWasm();
        if (typeof serialize_model !== 'function') {
            throw new Error('WASM serialize_model not available');
        }
        const modelJson = JSON.stringify(modelData);
        return serialize_model(modelJson, modelType, name);
    },

    deserializeModel: async (jsonString) => {
        await initWasm();
        if (typeof deserialize_model !== 'function') {
            throw new Error('WASM deserialize_model not available');
        }
        const resultJson = deserialize_model(jsonString);
        return JSON.parse(resultJson);
    },

    getModelMetadata: async (jsonString) => {
        await initWasm();
        if (typeof get_model_metadata !== 'function') {
            throw new Error('WASM get_model_metadata not available');
        }
        const resultJson = get_model_metadata(jsonString);
        return JSON.parse(resultJson);
    },

    // K-Fold Cross-Validation
    kfoldCvOls: (y, xVars, names, nFolds, shuffle, seed) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const namesJson = JSON.stringify(names);
        const shuffleJson = JSON.stringify(shuffle);
        const seedJson = seed === null ? 'null' : JSON.stringify(seed);
        return kfold_cv_ols(yJson, xJson, namesJson, nFolds, shuffleJson, seedJson);
    },

    kfoldCvRidge: (y, xVars, lambda, standardize, nFolds, shuffle, seed) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const shuffleJson = JSON.stringify(shuffle);
        const seedJson = seed === null ? 'null' : JSON.stringify(seed);
        return kfold_cv_ridge(yJson, xJson, lambda, standardize, nFolds, shuffleJson, seedJson);
    },

    kfoldCvLasso: (y, xVars, lambda, standardize, nFolds, shuffle, seed) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const shuffleJson = JSON.stringify(shuffle);
        const seedJson = seed === null ? 'null' : JSON.stringify(seed);
        return kfold_cv_lasso(yJson, xJson, lambda, standardize, nFolds, shuffleJson, seedJson);
    },

    kfoldCvElasticNet: (y, xVars, lambda, alpha, standardize, nFolds, shuffle, seed) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const shuffleJson = JSON.stringify(shuffle);
        const seedJson = seed === null ? 'null' : JSON.stringify(seed);
        return kfold_cv_elastic_net(yJson, xJson, lambda, alpha, standardize, nFolds, shuffleJson, seedJson);
    },

    // Trace Path (Coefficient Path)
    tracePath: (y, xVars, nLambda, lambdaMinRatio, alpha, standardize, maxIter, tol) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        return elastic_net_path_wasm(yJson, xJson, nLambda, lambdaMinRatio, alpha, standardize, maxIter, tol);
    },

    // ========================================================================
    // Prediction Intervals
    // ========================================================================

    /**
     * Compute OLS prediction intervals
     * @param {number[]} y - Response variable
     * @param {number[][]} xVars - Predictor variables (training data)
     * @param {number[][]} newX - New predictor values
     * @param {number} alpha - Significance level (e.g., 0.05 for 95% PI)
     * @returns {string} JSON string with predicted, lower_bound, upper_bound, se_pred, leverage
     */
    olsPredictionIntervals: (y, xVars, newX, alpha = 0.05) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const newXJson = JSON.stringify(newX);
        return ols_prediction_intervals(yJson, xJson, newXJson, alpha);
    },

    /**
     * Compute approximate Ridge prediction intervals
     * @param {number[]} y - Response variable
     * @param {number[][]} xVars - Predictor variables (training data)
     * @param {number[][]} newX - New predictor values
     * @param {number} alpha - Significance level
     * @param {number} lambda - Regularization strength
     * @param {boolean} standardize - Whether to standardize predictors
     * @returns {string} JSON string with prediction interval results
     */
    ridgePredictionIntervals: (y, xVars, newX, alpha = 0.05, lambda = 1.0, standardize = true) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const newXJson = JSON.stringify(newX);
        return ridge_prediction_intervals(yJson, xJson, newXJson, alpha, lambda, standardize);
    },

    /**
     * Compute approximate Lasso prediction intervals
     * @param {number[]} y - Response variable
     * @param {number[][]} xVars - Predictor variables (training data)
     * @param {number[][]} newX - New predictor values
     * @param {number} alpha - Significance level
     * @param {number} lambda - Regularization strength
     * @param {boolean} standardize - Whether to standardize predictors
     * @param {number} maxIter - Maximum iterations
     * @param {number} tol - Convergence tolerance
     * @returns {string} JSON string with prediction interval results
     */
    lassoPredictionIntervals: (y, xVars, newX, alpha = 0.05, lambda = 1.0, standardize = true, maxIter = 100000, tol = 1e-7) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const newXJson = JSON.stringify(newX);
        return lasso_prediction_intervals(yJson, xJson, newXJson, alpha, lambda, standardize, maxIter, tol);
    },

    /**
     * Compute approximate Elastic Net prediction intervals
     * @param {number[]} y - Response variable
     * @param {number[][]} xVars - Predictor variables (training data)
     * @param {number[][]} newX - New predictor values
     * @param {number} alpha - Significance level
     * @param {number} lambda - Regularization strength
     * @param {number} enetAlpha - Elastic net mixing parameter (0=Ridge, 1=Lasso)
     * @param {boolean} standardize - Whether to standardize predictors
     * @param {number} maxIter - Maximum iterations
     * @param {number} tol - Convergence tolerance
     * @returns {string} JSON string with prediction interval results
     */
    elasticNetPredictionIntervals: (y, xVars, newX, alpha = 0.05, lambda = 1.0, enetAlpha = 0.5, standardize = true, maxIter = 100000, tol = 1e-7) => {
        const yJson = JSON.stringify(y);
        const xJson = JSON.stringify(xVars);
        const newXJson = JSON.stringify(newX);
        return elastic_net_prediction_intervals(yJson, xJson, newXJson, alpha, lambda, enetAlpha, standardize, maxIter, tol);
    }
};

// ============================================================================
// REGRESSION FUNCTIONS
// ============================================================================

/**
 * Calculate OLS regression
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @returns {Promise<Object>} Regression results
 */
export async function calculateRegression(yVar, xVars) {
    const n = STATE.rawData.length;
    const k = xVars.length;

    // Validate minimum sample size
    if (n <= k + 1) {
        throw new Error(`Need at least ${k + 2} data points for ${k} predictor(s). You have ${n}.`);
    }

    // Validate that all variables exist in the data
    const allVariables = [yVar, ...xVars];
    for (const v of allVariables) {
        if (typeof v !== 'string') {
            throw new Error(`Invalid variable name: ${v}`);
        }
        if (STATE.rawData.length > 0 && !(v in STATE.rawData[0])) {
            throw new Error(`Variable "${v}" not found in data. Available columns: ${Object.keys(STATE.rawData[0]).join(', ')}`);
        }
    }

    // Check if WASM is ready
    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    // Prepare data for WASM
    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = xVars.map(v => STATE.rawData.map(row => row[v]));
    const names = ['Intercept', ...xVars];

    // Call WASM OLS regression
    const resultJson = WasmRegression.ols(yData, xData, names);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] OLS regression: n=${n}, k=${k}, R²=${result.r_squared?.toFixed(4) || 'N/A'}`);

    // Check for error
    if (result.error) {
        throw new Error(result.error);
    }

    // Validate required arrays exist
    if (!result.coefficients || !result.std_errors || !result.conf_int_lower || !result.conf_int_upper) {
        console.error('WASM result:', result);
        throw new Error('Invalid WASM response: missing required fields');
    }

    // Check for NaN values
    if (result.conf_int_lower.some(v => v === null || v === undefined || isNaN(v))) {
        console.error('WASM conf_int_lower contains NaN:', result.conf_int_lower);
        throw new Error('WASM calculation error: confidence intervals contain NaN values');
    }

    // Transform WASM output
    return {
        coefficients: result.coefficients,
        stdErrors: result.std_errors,
        tStats: result.t_stats,
        pValues: result.p_values,
        rSquared: result.r_squared,
        adjRSquared: result.adj_r_squared,
        mse: result.mse,
        rmse: result.rmse,
        mae: result.mae,
        stdError: result.std_error,
        fStat: result.f_statistic,
        fPValue: result.f_p_value,
        predictions: result.predictions,
        residuals: result.residuals,
        standardizedResiduals: result.standardized_residuals,
        leverage: result.leverage,
        confidenceIntervals: result.conf_int_lower.map((lower, i) => [lower, result.conf_int_upper[i]]),
        confIntLower: result.conf_int_lower,
        confIntUpper: result.conf_int_upper,
        vif: result.vif,
        n: result.n,
        k: result.k,
        df: result.df,
        variableNames: result.variable_names,
        logLikelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
        method: 'ols'
    };
}

/**
 * Calculate Ridge regression
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @param {number} lambda - Regularization strength
 * @param {boolean} standardize - Whether to standardize predictors
 * @returns {Promise<Object>} Regression results
 */
export async function calculateRidgeRegression(yVar, xVars, lambda = 1.0, standardize = true) {
    const n = STATE.rawData.length;
    const k = xVars.length;

    if (n <= k + 1) {
        throw new Error(`Need at least ${k + 2} data points for ${k} predictor(s). You have ${n}.`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = xVars.map(v => STATE.rawData.map(row => row[v]));
    const names = ['Intercept', ...xVars];

    const resultJson = WasmRegression.ridge(yData, xData, names, lambda, standardize);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] Ridge regression: n=${n}, k=${k}, λ=${lambda}, R²=${result.r_squared?.toFixed(4) || 'N/A'}`);

    if (result.error) {
        throw new Error(result.error);
    }

    // Standardized residuals
    const standardizedResiduals = result.residuals.map(r => r / result.rmse);

    return {
        coefficients: [result.intercept, ...result.coefficients],
        lambda: result.lambda,
        rSquared: result.r_squared,
        adjRSquared: result.adj_r_squared,
        mse: result.mse,
        stdError: result.rmse,
        rmse: result.rmse,
        mae: result.mae,
        predictions: result.fitted_values,
        residuals: result.residuals,
        standardizedResiduals: standardizedResiduals,
        df: result.df,
        n: n,
        k: k,
        variableNames: names,
        logLikelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
        method: 'ridge'
    };
}

/**
 * Calculate Lasso regression
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @param {number} lambda - Regularization strength
 * @param {boolean} standardize - Whether to standardize predictors
 * @param {number} maxIter - Maximum iterations
 * @param {number} tol - Convergence tolerance
 * @returns {Promise<Object>} Regression results
 */
export async function calculateLassoRegression(yVar, xVars, lambda = 1.0, standardize = true, maxIter = 1000, tol = 1e-7) {
    const n = STATE.rawData.length;
    const k = xVars.length;

    if (n <= k + 1) {
        throw new Error(`Need at least ${k + 2} data points for ${k} predictor(s). You have ${n}.`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = xVars.map(v => STATE.rawData.map(row => row[v]));
    const names = ['Intercept', ...xVars];

    const resultJson = WasmRegression.lasso(yData, xData, names, lambda, standardize, maxIter, tol);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] Lasso regression: n=${n}, k=${k}, λ=${lambda}, R²=${result.r_squared?.toFixed(4) || 'N/A'}, nonzero=${result.n_nonzero || 0}, iter=${result.iterations || 'N/A'}`);

    if (result.error) {
        throw new Error(result.error);
    }

    const standardizedResiduals = result.residuals.map(r => r / result.rmse);

    return {
        coefficients: [result.intercept, ...result.coefficients],
        lambda: result.lambda,
        rSquared: result.r_squared,
        adjRSquared: result.adj_r_squared,
        mse: result.mse,
        stdError: result.rmse,
        rmse: result.rmse,
        mae: result.mae,
        predictions: result.fitted_values,
        residuals: result.residuals,
        standardizedResiduals: standardizedResiduals,
        nNonzero: result.n_nonzero,
        iterations: result.iterations,
        converged: result.converged,
        n: n,
        k: k,
        variableNames: names,
        logLikelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
        method: 'lasso'
    };
}

/**
 * Calculate Elastic Net regression
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @param {number} lambda - Regularization strength
 * @param {number} alpha - Mixing parameter (0=Ridge, 1=Lasso)
 * @param {boolean} standardize - Whether to standardize predictors
 * @param {number} maxIter - Maximum iterations
 * @param {number} tol - Convergence tolerance
 * @returns {Promise<Object>} Regression results
 */
export async function calculateElasticNetRegression(yVar, xVars, lambda = 1.0, alpha = 0.5, standardize = true, maxIter = 1000, tol = 1e-7) {
    const n = STATE.rawData.length;
    const k = xVars.length;

    if (n <= k + 1) {
        throw new Error(`Need at least ${k + 2} data points for ${k} predictor(s). You have ${n}.`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = xVars.map(v => STATE.rawData.map(row => row[v]));
    const names = ['Intercept', ...xVars];

    const resultJson = WasmRegression.elasticNet(yData, xData, names, lambda, alpha, standardize, maxIter, tol);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] Elastic Net regression: n=${n}, k=${k}, λ=${lambda}, α=${alpha}, R²=${result.r_squared?.toFixed(4) || 'N/A'}, nonzero=${result.n_nonzero || 0}, iter=${result.iterations || 'N/A'}`);

    if (result.error) {
        throw new Error(result.error);
    }

    const standardizedResiduals = result.residuals.map(r => r / result.rmse);

    return {
        coefficients: [result.intercept, ...result.coefficients],
        lambda: result.lambda,
        alpha: result.alpha,
        rSquared: result.r_squared,
        adjRSquared: result.adj_r_squared,
        mse: result.mse,
        stdError: result.rmse,
        rmse: result.rmse,
        mae: result.mae,
        predictions: result.fitted_values,
        residuals: result.residuals,
        standardizedResiduals: standardizedResiduals,
        nNonzero: result.n_nonzero,
        iterations: result.iterations,
        converged: result.converged,
        n: n,
        k: k,
        variableNames: names,
        logLikelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
        method: 'elastic_net'
    };
}

/**
 * Generate lambda path for regularized regression
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @param {number} nLambda - Number of lambda values
 * @param {number} lambdaMinRatio - Minimum lambda ratio
 * @returns {Promise<Object>} Lambda path results
 */
export async function generateLambdaPath(yVar, xVars, nLambda = 100, lambdaMinRatio = 0.0001) {
    const n = STATE.rawData.length;

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = xVars.map(v => STATE.rawData.map(row => row[v]));

    const resultJson = WasmRegression.lambdaPath(yData, xData, nLambda, lambdaMinRatio);
    const result = JSON.parse(resultJson);

    if (result.error) {
        throw new Error(result.error);
    }

    return result;
}

/**
 * Calculate LOESS regression (Locally Estimated Scatterplot Smoothing)
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @param {number} span - Fraction of data used in each local fit (0.0-1.0)
 * @param {number} degree - Polynomial degree (0=constant, 1=linear, 2=quadratic)
 * @param {number} robustIterations - Number of robustness iterations (0 or 2)
 * @param {string} surface - Surface method ('direct' or 'interpolate')
 * @returns {Promise<Object>} Regression results
 */
export async function calculateLoessRegression(yVar, xVars, span = 0.75, degree = 1, robustIterations = 0, surface = 'direct') {
    const n = STATE.rawData.length;
    const k = xVars.length;

    if (n <= k + 1) {
        throw new Error(`Need at least ${k + 2} data points for ${k} predictor(s). You have ${n}.`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = xVars.map(v => STATE.rawData.map(row => row[v]));
    const names = ['Intercept', ...xVars];

    const resultJson = WasmRegression.loess(yData, xData, span, degree, robustIterations, surface);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] LOESS regression: n=${n}, k=${k}, span=${span}, degree=${degree}`);

    if (result.error) {
        throw new Error(result.error);
    }

    // Transform LOESS result to match expected format
    const residuals = yData.map((yi, i) => yi - result.fitted[i]);
    const rmse = Math.sqrt(residuals.reduce((sum, r) => sum + r * r, 0) / residuals.length);
    const mae = residuals.reduce((sum, r) => sum + Math.abs(r), 0) / residuals.length;

    // Calculate pseudo-R² for LOESS (1 - SS_res / SS_tot)
    const yMean = Stats.mean(yData);
    const ssTot = yData.reduce((sum, yi) => sum + (yi - yMean) ** 2, 0);
    const ssRes = residuals.reduce((sum, r) => sum + r * r, 0);
    const rSquared = 1 - (ssRes / ssTot);
    const adjRSquared = rSquared - (k * (1 - rSquared)) / (n - k - 1);

    return {
        coefficients: null, // LOESS doesn't have fixed coefficients
        fitted: result.fitted,
        rSquared: isNaN(rSquared) ? 0 : rSquared,
        adjRSquared: isNaN(adjRSquared) ? 0 : adjRSquared,
        mse: ssRes / n,
        rmse: rmse,
        mae: mae,
        predictions: result.fitted,
        residuals: residuals,
        standardizedResiduals: residuals.map(r => r / rmse),
        n: n,
        k: k,
        variableNames: names,
        span: result.span,
        degree: result.degree,
        robustIterations: result.robust_iterations,
        surface: result.surface,
        method: 'loess'
    };
}

/**
 * Calculate WLS (Weighted Least Squares) regression
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @param {string} weightsVar - Weights variable name
 * @returns {Promise<Object>} Regression results
 */
export async function calculateWlsRegression(yVar, xVars, weightsVar) {
    const n = STATE.rawData.length;
    const k = xVars.length;

    if (n <= k + 1) {
        throw new Error(`Need at least ${k + 2} data points for ${k} predictor(s). You have ${n}.`);
    }

    // Validate that weights variable exists
    if (typeof weightsVar !== 'string' || (STATE.rawData.length > 0 && !(weightsVar in STATE.rawData[0]))) {
        throw new Error(`Weights variable "${weightsVar}" not found in data. Available columns: ${Object.keys(STATE.rawData[0]).join(', ')}`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = xVars.map(v => STATE.rawData.map(row => row[v]));
    const weightsData = STATE.rawData.map(row => row[weightsVar]);
    const names = ['Intercept', ...xVars];

    // Validate weights are non-negative
    if (weightsData.some(w => w < 0)) {
        throw new Error('Weights must be non-negative. Found negative weight(s) in the data.');
    }

    const resultJson = WasmRegression.wls(yData, xData, weightsData);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] WLS regression: n=${n}, k=${k}, R²=${result.r_squared?.toFixed(4) || 'N/A'}`);

    if (result.error) {
        throw new Error(result.error);
    }

    // Standardized residuals
    const standardizedResiduals = result.residuals.map(r => r / result.rmse);

    return {
        coefficients: result.coefficients,
        stdErrors: result.standard_errors,
        tStats: result.t_statistics,
        pValues: result.p_values,
        confidenceIntervals: result.conf_int_lower.map((lower, i) => [lower, result.conf_int_upper[i]]),
        confIntLower: result.conf_int_lower,
        confIntUpper: result.conf_int_upper,
        rSquared: result.r_squared,
        adjRSquared: result.adj_r_squared,
        mse: result.mse,
        stdError: result.residual_std_error,
        rmse: result.rmse,
        mae: result.mae,
        predictions: result.fitted_values,
        residuals: result.residuals,
        standardizedResiduals: standardizedResiduals,
        fStat: result.f_statistic,
        fPValue: result.f_p_value,
        dfResiduals: result.df_residuals,
        dfModel: result.df_model,
        df: result.df_residuals,
        n: result.n,
        k: result.k,
        variableNames: names,
        weights: weightsVar,
        method: 'wls'
    };
}

/**
 * Predict values using LOESS at new points
 * @param {Array<Array<number>>} newX - New X values to predict (one array per predictor)
 * @param {Array<string>} xVarNames - Original X variable names
 * @param {string} yVarName - Original Y variable name
 * @param {number} span - Span parameter used in original fit
 * @param {number} degree - Degree parameter used in original fit
 * @param {number} robustIterations - Robust iterations used in original fit
 * @param {string} surface - Surface method used in original fit
 * @returns {Promise<Array<number>>} Predicted values
 */
export async function predictLoess(newX, xVarNames, yVarName, span, degree, robustIterations, surface) {
    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    // Get original data
    const originalX = xVarNames.map(v => STATE.rawData.map(row => row[v]));
    const originalY = STATE.rawData.map(row => row[yVarName]);

    const resultJson = WasmRegression.loessPredict(newX, originalX, originalY, span, degree, robustIterations, surface);
    const result = JSON.parse(resultJson);

    if (result.error) {
        throw new Error(result.error);
    }

    return result.predictions || result.fitted || [];
}

// ============================================================================
// POLYNOMIAL REGRESSION FUNCTIONS
// ============================================================================

/**
 * Calculate Polynomial OLS regression
 * @param {string} yVar - Y variable name
 * @param {string} xVar - Single X variable name (polynomial regression works with one predictor)
 * @param {number} degree - Polynomial degree (1 = linear, 2 = quadratic, etc.)
 * @param {boolean} center - Whether to center X before expansion (reduces multicollinearity)
 * @param {boolean} standardize - Whether to standardize polynomial features
 * @returns {Promise<Object>} Regression results
 */
export async function calculatePolynomialRegression(yVar, xVar, degree = 2, center = true, standardize = false) {
    const n = STATE.rawData.length;

    if (n <= degree + 1) {
        throw new Error(`Need at least ${degree + 2} data points for degree ${degree} polynomial. You have ${n}.`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = STATE.rawData.map(row => row[xVar]);

    const resultJson = WasmRegression.polynomialOls(yData, xData, degree, center, standardize);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] Polynomial OLS: n=${n}, degree=${degree}, center=${center}, R²=${result.ols_output?.r_squared?.toFixed(4) || 'N/A'}`);

    if (result.error) {
        throw new Error(result.error);
    }

    const ols = result.ols_output;
    if (!ols) {
        throw new Error('Invalid polynomial response: missing ols_output');
    }

    // Generate feature names for display
    const featureNames = result.feature_names || ['Intercept', 'x', 'x²', 'x³', 'x⁴', 'x⁵'].slice(0, degree + 1);

    // Store the fit JSON for predictions
    const fitJson = resultJson;

    return {
        coefficients: ols.coefficients,
        stdErrors: ols.std_errors,
        tStats: ols.t_stats,
        pValues: ols.p_values,
        rSquared: ols.r_squared,
        adjRSquared: ols.adj_r_squared,
        mse: ols.mse,
        rmse: ols.rmse,
        mae: ols.mae,
        stdError: ols.std_error,
        fStat: ols.f_statistic,
        fPValue: ols.f_p_value,
        predictions: ols.predictions,
        residuals: ols.residuals,
        standardizedResiduals: ols.standardized_residuals,
        leverage: ols.leverage,
        confidenceIntervals: ols.conf_int_lower?.map((lower, i) => [lower, ols.conf_int_upper[i]]) || [],
        confIntLower: ols.conf_int_lower || [],
        confIntUpper: ols.conf_int_upper || [],
        n: ols.n,
        k: ols.k,
        df: ols.df,
        variableNames: featureNames,
        logLikelihood: ols.log_likelihood,
        aic: ols.aic,
        bic: ols.bic,
        method: 'polynomial',
        polynomial: {
            degree: result.degree,
            centered: result.centered,
            xMean: result.x_mean,
            xStd: result.x_std,
            standardized: result.standardized,
            fitJson: fitJson, // Store for predictions
            xVar: xVar
        }
    };
}

/**
 * Calculate Polynomial Ridge regression
 * @param {string} yVar - Y variable name
 * @param {string} xVar - Single X variable name
 * @param {number} degree - Polynomial degree
 * @param {number} lambda - Regularization strength
 * @param {boolean} center - Whether to center X
 * @param {boolean} standardize - Whether to standardize features
 * @returns {Promise<Object>} Regression results
 */
export async function calculatePolynomialRidge(yVar, xVar, degree, lambda, center, standardize) {
    const n = STATE.rawData.length;

    if (n <= degree + 1) {
        throw new Error(`Need at least ${degree + 2} data points for degree ${degree} polynomial. You have ${n}.`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = STATE.rawData.map(row => row[xVar]);

    const resultJson = WasmRegression.polynomialRidge(yData, xData, degree, lambda, center, standardize);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] Polynomial Ridge: n=${n}, degree=${degree}, λ=${lambda}, R²=${result.r_squared?.toFixed(4) || 'N/A'}`);

    if (result.error) {
        throw new Error(result.error);
    }

    // Generate feature names
    const featureNames = ['Intercept'];
    for (let i = 1; i <= degree; i++) {
        featureNames.push(i === 1 ? 'x' : i === 2 ? 'x²' : i === 3 ? 'x³' : `x^${i}`);
    }

    const standardizedResiduals = result.residuals.map(r => r / result.rmse);

    return {
        coefficients: [result.intercept, ...result.coefficients],
        lambda: result.lambda,
        rSquared: result.r_squared,
        adjRSquared: result.adj_r_squared,
        mse: result.mse,
        stdError: result.rmse,
        rmse: result.rmse,
        mae: result.mae,
        predictions: result.fitted_values,
        residuals: result.residuals,
        standardizedResiduals: standardizedResiduals,
        df: result.df,
        n: n,
        k: degree + 1,
        variableNames: featureNames,
        logLikelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
        method: 'polynomial_ridge',
        polynomial: {
            degree: degree,
            centered: center,
            lambda: lambda,
            xVar: xVar
        }
    };
}

/**
 * Calculate Polynomial Lasso regression
 * @param {string} yVar - Y variable name
 * @param {string} xVar - Single X variable name
 * @param {number} degree - Polynomial degree
 * @param {number} lambda - Regularization strength
 * @param {boolean} center - Whether to center X
 * @param {boolean} standardize - Whether to standardize features
 * @returns {Promise<Object>} Regression results
 */
export async function calculatePolynomialLasso(yVar, xVar, degree, lambda, center, standardize) {
    const n = STATE.rawData.length;

    if (n <= degree + 1) {
        throw new Error(`Need at least ${degree + 2} data points for degree ${degree} polynomial. You have ${n}.`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = STATE.rawData.map(row => row[xVar]);

    const resultJson = WasmRegression.polynomialLasso(yData, xData, degree, lambda, center, standardize);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] Polynomial Lasso: n=${n}, degree=${degree}, λ=${lambda}, R²=${result.r_squared?.toFixed(4) || 'N/A'}, nonzero=${result.n_nonzero || 0}`);

    if (result.error) {
        throw new Error(result.error);
    }

    // Generate feature names
    const featureNames = ['Intercept'];
    for (let i = 1; i <= degree; i++) {
        featureNames.push(i === 1 ? 'x' : i === 2 ? 'x²' : i === 3 ? 'x³' : `x^${i}`);
    }

    const standardizedResiduals = result.residuals.map(r => r / result.rmse);

    return {
        coefficients: [result.intercept, ...result.coefficients],
        lambda: result.lambda,
        rSquared: result.r_squared,
        adjRSquared: result.adj_r_squared,
        mse: result.mse,
        stdError: result.rmse,
        rmse: result.rmse,
        mae: result.mae,
        predictions: result.fitted_values,
        residuals: result.residuals,
        standardizedResiduals: standardizedResiduals,
        nNonzero: result.n_nonzero,
        iterations: result.iterations,
        converged: result.converged,
        n: n,
        k: degree + 1,
        variableNames: featureNames,
        logLikelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
        method: 'polynomial_lasso',
        polynomial: {
            degree: degree,
            centered: center,
            lambda: lambda,
            xVar: xVar
        }
    };
}

/**
 * Calculate Polynomial Elastic Net regression
 * @param {string} yVar - Y variable name
 * @param {string} xVar - Single X variable name
 * @param {number} degree - Polynomial degree
 * @param {number} lambda - Regularization strength
 * @param {number} alpha - Mixing parameter (0=Ridge, 1=Lasso)
 * @param {boolean} center - Whether to center X
 * @param {boolean} standardize - Whether to standardize features
 * @returns {Promise<Object>} Regression results
 */
export async function calculatePolynomialElasticNet(yVar, xVar, degree, lambda, alpha, center, standardize) {
    const n = STATE.rawData.length;

    if (n <= degree + 1) {
        throw new Error(`Need at least ${degree + 2} data points for degree ${degree} polynomial. You have ${n}.`);
    }

    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const yData = STATE.rawData.map(row => row[yVar]);
    const xData = STATE.rawData.map(row => row[xVar]);

    const resultJson = WasmRegression.polynomialElasticNet(yData, xData, degree, lambda, alpha, center, standardize);
    const result = JSON.parse(resultJson);
    console.log(`[linreg-core] Polynomial Elastic Net: n=${n}, degree=${degree}, λ=${lambda}, α=${alpha}, R²=${result.r_squared?.toFixed(4) || 'N/A'}, nonzero=${result.n_nonzero || 0}`);

    if (result.error) {
        throw new Error(result.error);
    }

    // Generate feature names
    const featureNames = ['Intercept'];
    for (let i = 1; i <= degree; i++) {
        featureNames.push(i === 1 ? 'x' : i === 2 ? 'x²' : i === 3 ? 'x³' : `x^${i}`);
    }

    const standardizedResiduals = result.residuals.map(r => r / result.rmse);

    return {
        coefficients: [result.intercept, ...result.coefficients],
        lambda: result.lambda,
        alpha: result.alpha,
        rSquared: result.r_squared,
        adjRSquared: result.adj_r_squared,
        mse: result.mse,
        stdError: result.rmse,
        rmse: result.rmse,
        mae: result.mae,
        predictions: result.fitted_values,
        residuals: result.residuals,
        standardizedResiduals: standardizedResiduals,
        nNonzero: result.n_nonzero,
        iterations: result.iterations,
        converged: result.converged,
        n: n,
        k: degree + 1,
        variableNames: featureNames,
        logLikelihood: result.log_likelihood,
        aic: result.aic,
        bic: result.bic,
        method: 'polynomial_elastic_net',
        polynomial: {
            degree: degree,
            centered: center,
            lambda: lambda,
            alpha: alpha,
            xVar: xVar
        }
    };
}

/**
 * Predict using a fitted polynomial model
 * @param {string} fitJson - The JSON string of the fitted model
 * @param {Array<number>} xNew - New X values to predict
 * @returns {Promise<Array<number>>} Predicted values
 */
export async function predictPolynomial(fitJson, xNew) {
    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
    }

    const resultJson = WasmRegression.polynomialPredict(fitJson, xNew);
    const result = JSON.parse(resultJson);

    if (result.error) {
        throw new Error(result.error);
    }

    return result;
}

// ============================================================================
// STATISTICAL FUNCTIONS
// ============================================================================

export const Stats = {
    /**
     * Calculate mean using WASM
     */
    mean: (data) => {
        if (!wasmReady || typeof stats_mean !== 'function') {
            console.warn("WASM stats_mean not ready, using JS fallback");
            // Fallback
            if (!data || data.length === 0) return NaN;
            return data.reduce((a, b) => a + b, 0) / data.length;
        }
        const result = stats_mean(JSON.stringify(data));
        return JSON.parse(result);
    },

    /**
     * Calculate standard deviation using WASM
     */
    std: (data, ddof = 1) => {
        if (!wasmReady || typeof stats_stddev !== 'function') {
            console.warn("WASM stats_stddev not ready, using JS fallback");
            // Fallback
            if (!data || data.length <= 1) return NaN;
            const m = Stats.mean(data);
            const squaredDiffs = data.map(x => (x - m) ** 2);
            const variance = squaredDiffs.reduce((a, b) => a + b, 0) / (data.length - ddof);
            return Math.sqrt(variance);
        }
        const result = stats_stddev(JSON.stringify(data));
        return JSON.parse(result);
    },

    /**
     * Calculate variance using WASM
     */
    variance: (data) => {
        if (!wasmReady || typeof stats_variance !== 'function') {
            console.warn("WASM stats_variance not ready, using JS fallback");
            // Fallback
            if (!data || data.length <= 1) return NaN;
            const m = Stats.mean(data);
            return data.reduce((sum, x) => sum + (x - m) ** 2, 0) / (data.length - 1);
        }
        const result = stats_variance(JSON.stringify(data));
        return JSON.parse(result);
    },

    /**
     * Calculate median using WASM
     */
    median: (data) => {
        if (!wasmReady || typeof stats_median !== 'function') {
            console.warn("WASM stats_median not ready, using JS fallback");
            // Fallback
            if (!data || data.length === 0) return NaN;
            const sorted = [...data].sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 === 0
                ? (sorted[mid - 1] + sorted[mid]) / 2
                : sorted[mid];
        }
        const result = stats_median(JSON.stringify(data));
        return JSON.parse(result);
    },

    /**
     * Calculate quantile using WASM
     */
    quantile: (data, q) => {
        if (!wasmReady || typeof stats_quantile !== 'function') {
            console.warn("WASM stats_quantile not ready, using JS fallback");
            // Fallback
            if (!data || data.length === 0 || q < 0 || q > 1) return NaN;
            const sorted = [...data].sort((a, b) => a - b);
            const pos = q * (sorted.length - 1);
            const base = Math.floor(pos);
            const rest = pos - base;
            if (sorted[base + 1] !== undefined) {
                return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
            }
            return sorted[base];
        }
        const result = stats_quantile(JSON.stringify(data), q);
        return JSON.parse(result);
    },

    /**
     * Calculate correlation using WASM
     */
    correlation: (x, y) => {
        if (!wasmReady || typeof stats_correlation !== 'function') {
            console.warn("WASM stats_correlation not ready, using JS fallback");
            // Fallback
            if (!x || !y || x.length !== y.length || x.length < 2) return NaN;
            const n = x.length;
            const meanX = Stats.mean(x);
            const meanY = Stats.mean(y);
            let num = 0;
            let denX = 0;
            let denY = 0;
            for (let i = 0; i < n; i++) {
                const dx = x[i] - meanX;
                const dy = y[i] - meanY;
                num += dx * dy;
                denX += dx * dx;
                denY += dy * dy;
            }
            return num / Math.sqrt(denX * denY);
        }
        const result = stats_correlation(JSON.stringify(x), JSON.stringify(y));
        return JSON.parse(result);
    },

    /**
     * Get critical t-value using WASM
     */
    tCritical: (alpha, df) => {
        if (!wasmReady || typeof get_t_critical !== 'function') {
            console.warn("WASM get_t_critical not ready");
            return 1.96; // Fallback
        }
        return get_t_critical(alpha, df);
    },

    /**
     * Get inverse normal CDF using WASM
     */
    normalInverse: (p) => {
        if (!wasmReady || typeof get_normal_inverse !== 'function') {
            console.warn("WASM get_normal_inverse not ready");
            return 0; // Fallback
        }
        return get_normal_inverse(p);
    },

    /**
     * Serialize a model with metadata wrapper
     * @param {Object} result - Regression result object
     * @param {string} modelType - Model type ("OLS", "Ridge", "Lasso", "ElasticNet", "WLS", "LOESS")
     * @param {string} name - Optional model name
     * @returns {Promise<string>} Serialized JSON string
     */
    serializeModel: async (result, modelType, name = null) => {
        await initWasm();
        if (typeof serialize_model !== 'function') {
            throw new Error('WASM serialize_model not available');
        }
        const resultJson = JSON.stringify(result);
        return serialize_model(resultJson, modelType, name);
    },

    /**
     * Deserialize a model (extract model data from wrapper)
     * @param {string} serializedJson - Serialized model JSON string
     * @returns {Promise<Object>} Model data object
     */
    deserializeModel: async (serializedJson) => {
        await initWasm();
        if (typeof deserialize_model !== 'function') {
            throw new Error('WASM deserialize_model not available');
        }
        const modelJson = deserialize_model(serializedJson);
        return JSON.parse(modelJson);
    },

    /**
     * Get metadata from serialized model
     * @param {string} serializedJson - Serialized model JSON string
     * @returns {Promise<Object>} Metadata object
     */
    getModelMetadata: async (serializedJson) => {
        await initWasm();
        if (typeof get_model_metadata !== 'function') {
            throw new Error('WASM get_model_metadata not available');
        }
        const metadataJson = get_model_metadata(serializedJson);
        return JSON.parse(metadataJson);
    }
};
