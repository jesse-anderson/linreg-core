// ============================================================================
// regularized.js - Ridge, Lasso, and Elastic Net regression utilities
// ============================================================================

import { STATE } from './utils.js';
import { WasmRegression } from './core.js';

/**
 * Calculate Ridge regression
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @param {number} lambda - Regularization strength (default: 1.0)
 * @param {boolean} standardize - Whether to standardize predictors (default: true)
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
 * @param {number} lambda - Regularization strength (default: 1.0)
 * @param {boolean} standardize - Whether to standardize predictors (default: true)
 * @param {number} maxIter - Maximum iterations (default: 1000)
 * @param {number} tol - Convergence tolerance (default: 1e-7)
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
 * @param {number} lambda - Regularization strength (default: 1.0)
 * @param {number} alpha - Mixing parameter: 0=Ridge, 1=Lasso (default: 0.5)
 * @param {boolean} standardize - Whether to standardize predictors (default: true)
 * @param {number} maxIter - Maximum iterations (default: 1000)
 * @param {number} tol - Convergence tolerance (default: 1e-7)
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
 * @param {number} nLambda - Number of lambda values (default: 100)
 * @param {number} lambdaMinRatio - Minimum lambda ratio (default: 0.0001)
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
 * Suggest lambda values based on data characteristics
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @returns {Promise<Object>} Suggested lambda values
 */
export async function suggestLambdaValues(yVar, xVars) {
    try {
        const pathResult = await generateLambdaPath(yVar, xVars);

        return {
            lambda_max: pathResult.lambda_max,
            lambda_min: pathResult.lambda_min,
            suggested_lambdas: [
                pathResult.lambda_max,           // All coefficients zero
                pathResult.lambda_max * 0.1,     // Most coefficients zero
                pathResult.lambda_max * 0.01,    // Some coefficients zero
                pathResult.lambda_max * 0.001,   // Few coefficients zero
                pathResult.lambda_min            // Minimal regularization
            ]
        };
    } catch (e) {
        console.error('Failed to generate lambda suggestions:', e);
        return {
            lambda_max: 1.0,
            lambda_min: 0.001,
            suggested_lambdas: [1.0, 0.1, 0.01, 0.001]
        };
    }
}

/**
 * Get coefficient path for visualization
 * @param {string} yVar - Y variable name
 * @param {Array<string>} xVars - X variable names
 * @param {number} nLambda - Number of lambda values
 * @returns {Promise<Object>} Coefficient path data
 */
export async function getCoefficientPath(yVar, xVars, nLambda = 50) {
    // This would require fitting the model at each lambda value
    // For now, return the lambda path only
    return await generateLambdaPath(yVar, xVars, nLambda);
}

/**
 * Format regression method for display
 * @param {string} method - Regression method name
 * @returns {string} Formatted method name
 */
export function formatMethodName(method) {
    const names = {
        'ols': 'OLS',
        'ridge': 'Ridge',
        'lasso': 'Lasso',
        'elastic_net': 'Elastic Net',
        'wls': 'WLS',
        'loess': 'LOESS'
    };
    return names[method] || method.toUpperCase();
}

/**
 * Get method-specific parameters for display
 * @param {Object} result - Regression result object
 * @returns {Object} Parameters to display
 */
export function getMethodParameters(result) {
    switch (result.method) {
        case 'ridge':
            return { Lambda: result.lambda };
        case 'lasso':
            return { Lambda: result.lambda, 'Non-zero': result.nNonzero };
        case 'elastic_net':
            return { Lambda: result.lambda, Alpha: result.alpha, 'Non-zero': result.nNonzero };
        case 'wls':
            return { Weights: result.weights };
        case 'loess':
            return { Span: result.span, Degree: result.degree };
        default:
            return {};
    }
}
