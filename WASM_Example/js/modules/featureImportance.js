// ============================================================================
// featureImportance.js - Feature importance analysis and visualization
// ============================================================================

import { STATE, escapeHtml, showToast } from './utils.js';
import { Stats } from './core.js';
import {
    standardized_coefficients,
    shap_values_linear,
    permutation_importance_ols,
    vif_ranking,
    feature_importance_ols
} from '../linreg_core.js';

// ============================================================================
// FEATURE IMPORTANCE STATE
// ============================================================================

let currentFeatureImportance = null;
let featureImportanceChart = null;

// ============================================================================
// FEATURE IMPORTANCE CALCULATIONS
// ============================================================================

/**
 * Calculate standardized coefficients for feature importance
 * @param {Object} results - Regression results
 * @returns {Object} Standardized coefficients output
 */
export function calculateStandardizedCoefficients(results) {
    if (!results.coefficients || !STATE.xVariables || !STATE.rawData) {
        return null;
    }

    try {
        const xData = STATE.xVariables.map(v => STATE.rawData.map(row => row[v]));
        const yData = STATE.rawData.map(row => row[STATE.yVariable]);
        const yStd = Stats.std(yData);

        const xJson = JSON.stringify(xData);
        const coefJson = JSON.stringify(results.coefficients);
        const namesJson = JSON.stringify(STATE.xVariables);

        const resultJson = standardized_coefficients(xJson, coefJson, namesJson, yStd);
        return JSON.parse(resultJson);
    } catch (e) {
        console.error('[Feature Importance] Error calculating standardized coefficients:', e);
        return null;
    }
}

/**
 * Calculate SHAP values for linear models
 * @param {Object} results - Regression results
 * @returns {Object} SHAP output
 */
export function calculateShapValues(results) {
    if (!results.coefficients || !STATE.xVariables || !STATE.rawData) {
        return null;
    }

    try {
        const xData = STATE.xVariables.map(v => STATE.rawData.map(row => row[v]));

        const xJson = JSON.stringify(xData);
        const coefJson = JSON.stringify(results.coefficients);
        const namesJson = JSON.stringify(STATE.xVariables);

        const resultJson = shap_values_linear(xJson, coefJson, namesJson);
        return JSON.parse(resultJson);
    } catch (e) {
        console.error('[Feature Importance] Error calculating SHAP values:', e);
        return null;
    }
}

/**
 * Calculate VIF ranking from VIF results
 * @param {Object} results - Regression results with VIF
 * @returns {Object} VIF ranking output
 */
export function calculateVifRanking(results) {
    if (!results.vif || !results.vif.vif_values) {
        return null;
    }

    try {
        const vifJson = JSON.stringify(results.vif);
        const resultJson = vif_ranking(vifJson);
        return JSON.parse(resultJson);
    } catch (e) {
        console.error('[Feature Importance] Error calculating VIF ranking:', e);
        return null;
    }
}

/**
 * Calculate permutation importance
 * @param {Object} results - Regression results
 * @param {number} nPermutations - Number of permutation iterations
 * @param {number} seed - Random seed
 * @returns {Object} Permutation importance output
 */
export function calculatePermutationImportance(results, nPermutations = 50, seed = 42) {
    if (!results.coefficients || !STATE.xVariables || !STATE.rawData || !STATE.yVariable) {
        return null;
    }

    try {
        const yData = STATE.rawData.map(row => row[STATE.yVariable]);
        const xData = STATE.xVariables.map(v => STATE.rawData.map(row => row[v]));

        // Build fit object for permutation importance
        const fit = {
            coefficients: results.coefficients,
            predictions: results.predictions,
            residuals: results.residuals
        };

        const yJson = JSON.stringify(yData);
        const xJson = JSON.stringify(xData);
        const fitJson = JSON.stringify(fit);

        // Note: seed parameter is BigInt in WASM
        const resultJson = permutation_importance_ols(yJson, xJson, fitJson, nPermutations, BigInt(seed));
        return JSON.parse(resultJson);
    } catch (e) {
        console.error('[Feature Importance] Error calculating permutation importance:', e);
        return null;
    }
}

/**
 * Calculate complete feature importance analysis
 * @param {Object} results - Regression results
 * @param {Object} options - Options for calculation
 * @returns {Object} Complete feature importance output
 */
export async function calculateFeatureImportance(results, options = {}) {
    const {
        nPermutations = 50,
        seed = 42,
        yStd = null
    } = options;

    if (!results.coefficients || !STATE.xVariables || !STATE.rawData) {
        throw new Error('Cannot calculate feature importance: missing data');
    }

    // Calculate y_std if not provided
    let yStdValue = yStd;
    if (yStdValue === null) {
        const yData = STATE.rawData.map(row => row[STATE.yVariable]);
        yStdValue = Stats.std(yData);
    }

    try {
        const yData = STATE.rawData.map(row => row[STATE.yVariable]);
        const xData = STATE.xVariables.map(v => STATE.rawData.map(row => row[v]));

        const yJson = JSON.stringify(yData);
        const xJson = JSON.stringify(xData);
        const namesJson = JSON.stringify(STATE.xVariables);

        // Use the comprehensive feature_importance_ols function
        const resultJson = feature_importance_ols(
            yJson,
            xJson,
            namesJson,
            yStdValue,
            nPermutations,
            BigInt(seed)
        );

        const result = JSON.parse(resultJson);

        // Check for error in result
        if (result.error) {
            throw new Error(result.error);
        }

        // Validate result structure
        if (!result || typeof result !== 'object') {
            throw new Error('Invalid result structure from WASM');
        }

        // Log the result for debugging
        console.log('[Feature Importance] WASM result:', result);

        currentFeatureImportance = result;
        return result;
    } catch (e) {
        console.error('[Feature Importance] Calculation error:', e);
        if (e.name === 'SyntaxError') {
            throw new Error('Failed to parse WASM result. The response may not be valid JSON: ' + e.message);
        }
        throw e;
    }
}

// ============================================================================
// FEATURE IMPORTANCE DISPLAY
// ============================================================================

/**
 * Update feature importance display in the UI
 * @param {Object} results - Regression results
 */
export async function updateFeatureImportanceDisplay(results) {
    const container = document.getElementById('featureImportanceContent');
    if (!container) return;

    // Only show for OLS regression with multiple predictors
    if (results.method !== 'ols' && results.method !== 'wls') {
        container.innerHTML = `
            <p style="color: var(--text-muted); font-size: 0.875rem;">
                Feature importance is available for OLS and WLS regression.
            </p>
        `;
        return;
    }

    if (STATE.xVariables.length < 2) {
        container.innerHTML = `
            <p style="color: var(--text-muted); font-size: 0.875rem;">
                Feature importance requires at least 2 predictor variables.
            </p>
        `;
        return;
    }

    // Show loading state
    container.innerHTML = `
        <div style="text-align: center; padding: 40px;">
            <div class="loading-spinner"></div>
            <p style="color: var(--text-secondary); margin-top: 16px;">Calculating feature importance...</p>
        </div>
    `;

    try {
        const importance = await calculateFeatureImportance(results);
        renderFeatureImportance(importance, container);
    } catch (e) {
        console.error('[Feature Importance] Calculation failed:', e);
        showToast('Failed to calculate feature importance: ' + (e.message || 'Unknown error'), 'error');
        container.innerHTML = `
            <div style="padding: 16px; background: var(--bg-error); border-radius: 8px; color: var(--text-error);">
                <p style="margin: 0;"><strong>Error:</strong> ${escapeHtml(e.message || 'Failed to calculate feature importance')}</p>
                <p style="margin: 4px 0 0 0; font-size: 0.75rem; opacity: 0.8;">Check the browser console for more details.</p>
            </div>
        `;
    }
}

/**
 * Render feature importance results
 * @param {Object} importance - Feature importance results
 * @param {HTMLElement} container - Container element
 */
function renderFeatureImportance(importance, container) {
    try {
        let html = '';

        // Validate importance object
        if (!importance || typeof importance !== 'object') {
            throw new Error('Invalid importance data');
        }

        // Summary cards
        html += renderImportanceSummary(importance);

        // Standardized coefficients
        if (importance.standardized_coefficients) {
            html += renderStandardizedCoefficients(importance.standardized_coefficients);
        }

        // SHAP values
        if (importance.shap) {
            html += renderShapValues(importance.shap);
        }

        // VIF ranking
        if (importance.vif_ranking) {
            html += renderVifRanking(importance.vif_ranking);
        }

        // Permutation importance
        if (importance.permutation_importance) {
            html += renderPermutationImportance(importance.permutation_importance);
        }

        // Combined importance chart
        html += renderCombinedImportanceChart(importance);

        container.innerHTML = html;

        // Initialize the chart
        initFeatureImportanceChart(importance);
    } catch (e) {
        console.error('[Feature Importance] Render error:', e);
        container.innerHTML = `
            <div style="padding: 16px; background: var(--bg-error); border-radius: 8px; color: var(--text-error);">
                <p style="margin: 0;"><strong>Error rendering results:</strong> ${escapeHtml(e.message || 'Unknown error')}</p>
            </div>
        `;
    }
}

/**
 * Render importance summary cards
 */
function renderImportanceSummary(importance) {
    let html = '<div class="importance-summary">';

    // The WASM returns separate arrays for variable_names and values
    // We need to combine them for display

    // Standardized coefficients
    const stdCoefNames = importance.standardized_coefficients?.variable_names || [];
    const stdCoefValues = importance.standardized_coefficients?.standardized_coefficients || [];

    // SHAP values - mean_abs_shap is an array of numbers
    const shapNames = importance.shap?.variable_names || [];
    const shapValues = importance.shap?.mean_abs_shap || [];

    // Permutation importance
    const permNames = importance.permutation_importance?.variable_names || [];
    const permValues = importance.permutation_importance?.importance || [];

    if (stdCoefValues.length > 0) {
        // Find max by absolute value
        let maxIdx = 0;
        let maxAbs = Math.abs(stdCoefValues[0] ?? 0);
        stdCoefValues.forEach((val, idx) => {
            const absVal = Math.abs(val ?? 0);
            if (absVal > maxAbs) {
                maxAbs = absVal;
                maxIdx = idx;
            }
        });

        html += `
            <div class="importance-card">
                <div class="importance-label">Highest |Std Coef|</div>
                <div class="importance-value">${escapeHtml(stdCoefNames[maxIdx] || 'N/A')}</div>
                <div class="importance-sub">${maxAbs.toFixed(4)}</div>
            </div>
        `;
    }

    if (shapValues.length > 0) {
        // Find max
        let maxIdx = 0;
        let maxVal = shapValues[0] ?? 0;
        shapValues.forEach((val, idx) => {
            if ((val ?? 0) > maxVal) {
                maxVal = val ?? 0;
                maxIdx = idx;
            }
        });

        html += `
            <div class="importance-card">
                <div class="importance-label">Highest Mean |SHAP|</div>
                <div class="importance-value">${escapeHtml(shapNames[maxIdx] || 'N/A')}</div>
                <div class="importance-sub">${maxVal.toFixed(4)}</div>
            </div>
        `;
    }

    if (permValues.length > 0) {
        // Find max
        let maxIdx = 0;
        let maxVal = permValues[0] ?? 0;
        permValues.forEach((val, idx) => {
            if ((val ?? 0) > maxVal) {
                maxVal = val ?? 0;
                maxIdx = idx;
            }
        });

        html += `
            <div class="importance-card">
                <div class="importance-label">Highest Permutation Imp</div>
                <div class="importance-value">${escapeHtml(permNames[maxIdx] || 'N/A')}</div>
                <div class="importance-sub">${maxVal.toFixed(4)}</div>
            </div>
        `;
    }

    html += '</div>';
    return html;
}

/**
 * Render standardized coefficients table
 */
function renderStandardizedCoefficients(stdCoefs) {
    const names = stdCoefs.variable_names || [];
    const values = stdCoefs.standardized_coefficients || [];

    if (names.length === 0 || values.length === 0) {
        return '';
    }

    let html = `
        <div class="importance-section">
            <div class="importance-section-header">
                <h4>Standardized Coefficients</h4>
                <p class="importance-section-desc">
                    Coefficients scaled to compare relative importance. Larger absolute values indicate stronger influence on the response variable.
                </p>
            </div>
            <div class="table-container">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Std Coefficient</th>
                            <th>|Std Coef|</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    // Create array of indices and sort by absolute standardized coefficient
    const indices = names.map((_, i) => i);
    indices.sort((a, b) => Math.abs(values[b] ?? 0) - Math.abs(values[a] ?? 0));

    indices.forEach(i => {
        const name = names[i] || 'N/A';
        const stdCoef = values[i] ?? 0;
        const absStd = Math.abs(stdCoef);
        const signClass = stdCoef >= 0 ? 'positive' : 'negative';
        const intensity = Math.min(absStd / 2, 1);

        html += `
            <tr style="background: linear-gradient(90deg, ${intensity > 0 ? 'rgba(34, 197, 94, 0.1)' : 'transparent'} ${intensity * 100}%, transparent 100%);">
                <td>${escapeHtml(name)}</td>
                <td class="${signClass}">${stdCoef.toFixed(4)}</td>
                <td><strong>${absStd.toFixed(4)}</strong></td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;

    return html;
}

/**
 * Render SHAP values table
 */
function renderShapValues(shap) {
    const names = shap.variable_names || [];
    const meanAbsShap = shap.mean_abs_shap || [];

    if (names.length === 0 || meanAbsShap.length === 0) {
        return '';
    }

    let html = `
        <div class="importance-section">
            <div class="importance-section-header">
                <h4>SHAP Values (Mean Absolute)</h4>
                <p class="importance-section-desc">
                    SHAP (SHapley Additive exPlanations) values measure each feature's contribution to the model's predictions. Mean absolute SHAP shows global importance.
                </p>
            </div>
            <div class="table-container">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Mean |SHAP|</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    // Sort by mean absolute SHAP
    const indices = names.map((_, i) => i);
    indices.sort((a, b) => (meanAbsShap[b] ?? 0) - (meanAbsShap[a] ?? 0));

    const maxShap = Math.max(...meanAbsShap.map(v => v ?? 0), 1);

    indices.forEach(i => {
        const name = names[i] || 'N/A';
        const value = meanAbsShap[i] ?? 0;
        const intensity = Math.min(value / maxShap, 1);

        html += `
            <tr style="background: linear-gradient(90deg, rgba(34, 197, 94, ${intensity * 0.15}) ${intensity * 100}%, transparent 100%);">
                <td>${escapeHtml(name)}</td>
                <td><strong>${value.toFixed(4)}</strong></td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;

    return html;
}

/**
 * Render VIF ranking
 */
function renderVifRanking(vifRanking) {
    const names = vifRanking.variable_names || [];
    const vifValues = vifRanking.vif_values || [];

    if (names.length === 0 || vifValues.length === 0) {
        return '';
    }

    let html = `
        <div class="importance-section">
            <div class="importance-section-header">
                <h4>VIF Ranking (Multicollinearity)</h4>
                <p class="importance-section-desc">
                    VIF (Variance Inflation Factor) measures multicollinearity. Lower values indicate less correlation with other predictors.
                    <span style="color: var(--accent-success);">VIF < 5: Good</span> |
                    <span style="color: var(--accent-warning);">VIF 5-10: Moderate</span> |
                    <span style="color: var(--accent-error);">VIF > 10: High</span>
                </p>
            </div>
            <div class="table-container">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Variable</th>
                            <th>VIF</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    // Sort by VIF (lower is better)
    const indices = names.map((_, i) => i);
    indices.sort((a, b) => (vifValues[a] ?? 0) - (vifValues[b] ?? 0));

    indices.forEach((originalIndex, rankIndex) => {
        const name = names[originalIndex] || 'N/A';
        const vif = vifValues[originalIndex] ?? 0;

        let interpretationClass = 'good';
        let interpretation = 'Low multicollinearity';

        if (vif > 10) {
            interpretationClass = 'high';
            interpretation = 'High multicollinearity';
        } else if (vif > 5) {
            interpretationClass = 'moderate';
            interpretation = 'Moderate multicollinearity';
        }

        const interpretationColors = {
            good: 'var(--accent-success)',
            moderate: 'var(--accent-warning)',
            high: 'var(--accent-error)'
        };

        html += `
            <tr>
                <td>${rankIndex + 1}</td>
                <td>${escapeHtml(name)}</td>
                <td><strong>${vif.toFixed(4)}</strong></td>
                <td style="color: ${interpretationColors[interpretationClass]}; font-weight: 500;">${interpretation}</td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;

    return html;
}

/**
 * Render permutation importance
 */
function renderPermutationImportance(permImp) {
    const names = permImp.variable_names || [];
    const importance = permImp.importance || [];

    if (names.length === 0 || importance.length === 0) {
        return '';
    }

    let html = `
        <div class="importance-section">
            <div class="importance-section-header">
                <h4>Permutation Importance</h4>
                <p class="importance-section-desc">
                    Measures the decrease in model performance when a feature's values are randomly shuffled.
                    Higher values indicate the feature is more important for predictions.
                </p>
            </div>
            <div class="table-container">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Importance</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    // Sort by importance (descending)
    const indices = names.map((_, i) => i);
    indices.sort((a, b) => (importance[b] ?? 0) - (importance[a] ?? 0));

    const maxImp = Math.max(...importance.map(v => v ?? 0), 1);

    indices.forEach(i => {
        const name = names[i] || 'N/A';
        const imp = importance[i] ?? 0;
        const intensity = (imp / maxImp);

        html += `
            <tr style="background: linear-gradient(90deg, rgba(34, 197, 94, ${intensity * 0.15}) ${intensity * 100}%, transparent 100%);">
                <td>${escapeHtml(name)}</td>
                <td><strong>${imp.toFixed(4)}</strong></td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;

    return html;
}

/**
 * Render combined importance chart section
 */
function renderCombinedImportanceChart(importance) {
    return `
        <div class="importance-section">
            <div class="importance-section-header">
                <h4>Combined Feature Importance</h4>
                <p class="importance-section-desc">
                    Visualization comparing all importance metrics (normalized to 0-1 scale for comparison).
                </p>
            </div>
            <div class="chart-wrapper" style="height: 300px;">
                <canvas id="featureImportanceChart"></canvas>
            </div>
        </div>
    `;
}

/**
 * Initialize the combined feature importance chart
 */
function initFeatureImportanceChart(importance) {
    const canvas = document.getElementById('featureImportanceChart');
    if (!canvas) return;

    // Destroy existing chart
    if (featureImportanceChart && typeof featureImportanceChart.destroy === 'function') {
        featureImportanceChart.destroy();
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('[Feature Importance] Could not get canvas context');
        return;
    }

    // Extract data - WASM returns parallel arrays
    const stdCoefNames = importance.standardized_coefficients?.variable_names || [];
    const stdCoefValues = importance.standardized_coefficients?.standardized_coefficients || [];
    const shapNames = importance.shap?.variable_names || [];
    const shapValues = importance.shap?.mean_abs_shap || [];
    const permNames = importance.permutation_importance?.variable_names || [];
    const permValues = importance.permutation_importance?.importance || [];

    // Use STATE.xVariables as the source of truth for variable order
    const variables = STATE.xVariables || [];

    // Create maps for lookup by name
    const stdCoefMap = new Map(stdCoefNames.map((name, i) => [name, stdCoefValues[i] ?? 0]));
    const shapMap = new Map(shapNames.map((name, i) => [name, shapValues[i] ?? 0]));
    const permMap = new Map(permNames.map((name, i) => [name, permValues[i] ?? 0]));

    // Get values for each variable
    const stdCoefNorm = variables.map(v => Math.abs(stdCoefMap.get(v) ?? 0));
    const shapNorm = variables.map(v => shapMap.get(v) ?? 0);
    const permNorm = variables.map(v => permMap.get(v) ?? 0);

    // Normalize each metric to 0-1
    const maxStdCoef = Math.max(...stdCoefNorm, 1e-10);
    const maxShap = Math.max(...shapNorm, 1e-10);
    const maxPerm = Math.max(...permNorm, 1e-10);

    const stdCoefNormScaled = stdCoefNorm.map(v => v / maxStdCoef);
    const shapNormScaled = shapNorm.map(v => v / maxShap);
    const permNormScaled = permNorm.map(v => v / maxPerm);

    // Calculate average rank for sorting
    const avgImportance = variables.map((_, i) => {
        const values = [stdCoefNormScaled[i], shapNormScaled[i], permNormScaled[i]].filter(v => v > 0);
        return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    });

    const sortedIndices = variables.map((_, i) => i)
        .sort((a, b) => avgImportance[b] - avgImportance[a]);

    const sortedVariables = sortedIndices.map(i => variables[i]);
    const sortedStdCoef = sortedIndices.map(i => stdCoefNormScaled[i]);
    const sortedShap = sortedIndices.map(i => shapNormScaled[i]);
    const sortedPerm = sortedIndices.map(i => permNormScaled[i]);

    try {
        featureImportanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sortedVariables,
                datasets: [
                    {
                        label: 'Std Coefficient',
                        data: sortedStdCoef,
                        backgroundColor: 'rgba(34, 197, 94, 0.7)',
                        borderColor: 'rgba(34, 197, 94, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'SHAP',
                        data: sortedShap,
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Permutation',
                        data: sortedPerm,
                        backgroundColor: 'rgba(168, 85, 247, 0.7)',
                        borderColor: 'rgba(168, 85, 247, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Normalized Importance (0-1)',
                            color: 'var(--text-muted)',
                            font: { size: 11 }
                        },
                        grid: {
                            color: 'var(--border-color)'
                        },
                        ticks: {
                            color: 'var(--text-secondary)'
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: 'var(--text-secondary)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: 'var(--text-primary)',
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'var(--bg-tooltip)',
                        titleColor: 'var(--text-primary)',
                        bodyColor: 'var(--text-secondary)',
                        borderColor: 'var(--border-color)',
                        borderWidth: 1
                    }
                }
            }
        });
    } catch (e) {
        console.error('[Feature Importance] Failed to create chart:', e);
    }
}

/**
 * Get current feature importance results
 * @returns {Object|null} Current feature importance results
 */
export function getCurrentFeatureImportance() {
    return currentFeatureImportance;
}

/**
 * Reset feature importance state
 */
export function resetFeatureImportance() {
    currentFeatureImportance = null;
    if (featureImportanceChart && typeof featureImportanceChart.destroy === 'function') {
        featureImportanceChart.destroy();
        featureImportanceChart = null;
    }
}
