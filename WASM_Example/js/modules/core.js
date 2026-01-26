// ============================================================================
// core.js - Core WASM integration and regression functions
// ============================================================================

import init, {
    ols_regression,
    ridge_regression,
    lasso_regression,
    elastic_net_regression,
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
    reset_test,
    breusch_godfrey_test,
    get_version,
    stats_mean,
    stats_stddev,
    stats_variance,
    stats_median,
    stats_quantile,
    stats_correlation
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
        confidenceIntervals: result.conf_int_lower.map((lower, i) => [lower, result.conf_int_upper[i]]),
        vif: result.vif,
        n: result.n,
        k: result.k,
        df: result.df,
        variableNames: result.variable_names,
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
    }
};
