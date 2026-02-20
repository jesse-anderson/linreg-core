// ============================================================================
// charts.js - Chart management and visualization
// ============================================================================

import { STATE, escapeHtml, hexToRgba, getChartColors } from './utils.js';
import { Stats, WasmRegression } from './core.js';

// ============================================================================
// CHART MANAGEMENT
// ============================================================================

/**
 * Update all charts with regression results
 * @param {Object} results - Regression results
 */
export function updateCharts(results) {
    updateMainChart(results);
    updateResidualsChart(results);
    updateQQChart(results);
    updateLeverageChart(results);
    updateCoefficientChart(results);

    // Update path chart if it's currently being shown
    const pathContainer = document.getElementById('pathChartContainer');
    if (pathContainer && pathContainer.style.display !== 'none' && STATE.pathResults) {
        updatePathChart(STATE.pathResults);
    }
}

/**
 * Update the main regression chart
 * @param {Object} results - Regression results
 */
export function updateMainChart(results) {
    const canvas = document.getElementById('mainChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colors = getChartColors();

    // Destroy existing chart
    if (STATE.charts.main && typeof STATE.charts.main.destroy === 'function') {
        STATE.charts.main.destroy();
    }

    const hasRawData = STATE.rawData && STATE.rawData.length > 0;
    let yData;

    if (hasRawData && STATE.yVariable && STATE.rawData[0][STATE.yVariable] !== undefined) {
        yData = STATE.rawData.map(row => row[STATE.yVariable]);
    } else if (results.predictions && results.residuals) {
        // Reconstruct actuals
        yData = results.predictions.map((p, i) => p + results.residuals[i]);
    } else {
        yData = [];
    }

    if (results.k === 1 && hasRawData) {
        // Simple regression: scatter with regression line and confidence band
        updateSimpleRegressionChart(ctx, results, yData, colors);
    } else {
        // Multiple regression (or missing raw data): actual vs predicted
        updateMultipleRegressionChart(ctx, results, yData, colors);
    }
}

/**
 * Update simple regression chart (scatter with line)
 */
function updateSimpleRegressionChart(ctx, results, yData, colors) {
    const xVarName = STATE.xVariables[0];
    const xData = STATE.rawData.map(row => row[xVarName]);

    // Check if LOESS regression
    if (results.method === 'loess') {
        updateLoessChart(ctx, results, xData, yData, xVarName, colors);
        return;
    }

    // Sort for line plotting
    const sorted = xData.map((x, i) => ({ x, y: yData[i] }))
        .sort((a, b) => a.x - b.x);

    // Generate regression line
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const b0 = results.coefficients[0];
    const b1 = results.coefficients[1];
    const lineData = [
        { x: xMin, y: b0 + b1 * xMin },
        { x: xMax, y: b0 + b1 * xMax }
    ];

    // Build chart title with method and CI information
    const methodLabel = getMethodLabel(results);
    const ciLabel = results.confidenceIntervals ? '95% CI & PI Bands' :
                    results.method === 'ridge' || results.method === 'lasso' ?
                    'CI/PI not available for regularized regression' : 'No Confidence Interval';

    const titleEl = document.getElementById('mainChartTitle');
    if (titleEl) {
        titleEl.textContent = `Scatter Plot: ${escapeHtml(xVarName)} vs ${escapeHtml(STATE.yVariable)} — ${methodLabel} — ${ciLabel}`;
    }

    let datasets = [];

    // Add 95% prediction interval band via WASM (wider than CI, lighter fill)
    if ((results.method === 'ols' || results.method === 'wls') && results.stdError && results.df) {
        try {
            const piBand = computePredictionBandWasm(xData, yData, xMin, xMax);
            if (piBand) {
                const piColor = '#f59e0b'; // amber/orange for PI
                datasets.push(
                    {
                        label: '95% PI upper',
                        data: piBand.upper,
                        type: 'line',
                        borderColor: piColor,
                        borderWidth: 1,
                        borderDash: [5, 5],
                        backgroundColor: 'transparent',
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3,
                        order: 5,
                        z: 0
                    },
                    {
                        label: '95% PI (individual obs)',
                        data: piBand.lower,
                        type: 'line',
                        borderColor: piColor,
                        borderWidth: 1,
                        borderDash: [5, 5],
                        backgroundColor: hexToRgba(piColor, 0.08),
                        fill: '-1',
                        pointRadius: 0,
                        tension: 0.3,
                        order: 6,
                        z: 0
                    }
                );
            }
        } catch (e) {
            console.warn('Failed to compute prediction intervals via WASM:', e);
        }
    }

    // Add 95% confidence band for OLS or WLS
    if ((results.method === 'ols' || results.method === 'wls') && results.stdError && results.df) {
        const confidenceBand = createConfidenceBand(xData, xMin, xMax, b0, b1, results);
        const ciColor = getMethodColor(results.method, colors);

        datasets.push(
            {
                label: '95% CI upper',
                data: confidenceBand.upper,
                type: 'line',
                borderColor: ciColor,
                borderWidth: 1,
                backgroundColor: 'transparent',
                fill: false,
                pointRadius: 0,
                tension: 0.3,
                order: 10,
                z: 0
            },
            {
                label: '95% CI (mean response)',
                data: confidenceBand.lower,
                type: 'line',
                borderColor: ciColor,
                borderWidth: 1,
                backgroundColor: hexToRgba(ciColor, 0.3),
                fill: '-1',
                pointRadius: 0,
                tension: 0.3,
                order: 11,
                z: 0
            }
        );
    }

    // Add regression line
    datasets.push({
        label: `Regression Line (${methodLabel})`,
        data: lineData,
        type: 'line',
        borderColor: getMethodColor(results.method, colors),
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
        order: 20,
        z: 1
    });

    // Add data points
    datasets.push({
        label: 'Observed Data',
        data: xData.map((x, i) => ({ x, y: yData[i] })),
        backgroundColor: colors.accent || '#22c55e',
        pointRadius: 6,
        pointHoverRadius: 8,
        order: 30,
        z: 2
    });

    STATE.charts.main = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            ...getChartOptions(escapeHtml(xVarName), escapeHtml(STATE.yVariable)),
            plugins: {
                ...getChartOptions(escapeHtml(xVarName), escapeHtml(STATE.yVariable)).plugins,
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const idx = ctx.dataIndex;
                            return `Obs ${idx + 1}: (${xData[idx].toFixed(3)}, ${yData[idx].toFixed(3)})`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update LOESS chart (smooth curve through data)
 */
function updateLoessChart(ctx, results, xData, yData, xVarName, colors) {
    // Sort data by X for smooth curve plotting
    const sortedIndices = xData.map((_, i) => i).sort((a, b) => xData[a] - xData[b]);
    const loessX = sortedIndices.map(i => xData[i]);
    const loessY = sortedIndices.map(i => results.fitted[i]);

    // Build chart title with LOESS parameters
    const methodLabel = `LOESS (span=${results.span}, degree=${results.degree})`;
    const titleEl = document.getElementById('mainChartTitle');
    if (titleEl) {
        titleEl.textContent = `Scatter Plot: ${escapeHtml(xVarName)} vs ${escapeHtml(STATE.yVariable)} — ${methodLabel}`;
    }

    // Create smooth curve dataset and observed data dataset
    const datasets = [
        {
            label: 'LOESS Smooth Curve',
            data: loessX.map((x, i) => ({ x, y: loessY[i] })),
            type: 'line',
            borderColor: getMethodColor('loess', colors),
            borderWidth: 2.5,
            pointRadius: 0,
            fill: false,
            tension: 0.4,
            order: 10,
            z: 1
        },
        {
            label: 'Observed Data',
            data: xData.map((x, i) => ({ x, y: yData[i] })),
            backgroundColor: colors.accent || '#22c55e',
            pointRadius: 6,
            pointHoverRadius: 8,
            order: 20,
            z: 2
        }
    ];

    STATE.charts.main = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            ...getChartOptions(escapeHtml(xVarName), escapeHtml(STATE.yVariable)),
            plugins: {
                ...getChartOptions(escapeHtml(xVarName), escapeHtml(STATE.yVariable)).plugins,
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.datasetIndex === 0) {
                                // LOESS curve point
                                const idx = ctx.dataIndex;
                                return `LOESS: (${loessX[idx].toFixed(3)}, ${loessY[idx].toFixed(3)})`;
                            } else {
                                // Observed data point
                                const idx = ctx.dataIndex;
                                return `Obs: (${xData[idx].toFixed(3)}, ${yData[idx].toFixed(3)})`;
                            }
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update multiple regression chart (actual vs predicted)
 */
function updateMultipleRegressionChart(ctx, results, yData, colors) {
    const methodLabel = getMethodLabel(results);

    let ciLabel = '';
    if ((results.method === 'ols' || results.method === 'wls') && results.stdError) {
        ciLabel = '95% Prediction Band';
    } else if (results.method === 'ridge' || results.method === 'lasso') {
        ciLabel = 'CI not available for regularized regression';
    }

    const titleEl = document.getElementById('mainChartTitle');
    if (titleEl) {
        titleEl.textContent = `Actual vs Predicted Values — ${methodLabel} — k=${results.k}${ciLabel ? ' — ' + ciLabel : ''}`;
    }

    const scatterData = yData.map((actual, i) => ({
        x: actual,
        y: results.predictions[i]
    }));

    const minVal = Math.min(...yData, ...results.predictions);
    const maxVal = Math.max(...yData, ...results.predictions);

    let datasets = [];

    // Add 95% prediction band for OLS or WLS
    if ((results.method === 'ols' || results.method === 'wls') && results.stdError && results.rmse) {
        const predictionBand = createPredictionBand(minVal, maxVal, results.rmse);
        const ciColor = getMethodColor(results.method, colors);

        datasets.push(
            {
                label: '95% PI upper',
                data: predictionBand.upper,
                type: 'line',
                borderColor: ciColor,
                borderWidth: 1,
                backgroundColor: 'transparent',
                fill: false,
                pointRadius: 0,
                tension: 0,
                order: 10,
                z: 0
            },
            {
                label: '95% PI (prediction)',
                data: predictionBand.lower,
                type: 'line',
                borderColor: ciColor,
                borderWidth: 1,
                backgroundColor: hexToRgba(ciColor, 0.25),
                fill: '-1',
                pointRadius: 0,
                tension: 0,
                order: 11,
                z: 0
            }
        );
    }

    // Perfect fit reference line
    datasets.push({
        label: 'Perfect Fit (y = x)',
        data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
        type: 'line',
        borderColor: colors.textMuted || '#999',
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        borderDash: [5, 5],
        order: 20,
        z: 1
    });

    // Add scatter points
    datasets.push({
        label: 'Data Points',
        data: scatterData,
        backgroundColor: colors.accent || '#22c55e',
        pointRadius: 5,
        pointHoverRadius: 7,
        order: 30,
        z: 2
    });

    STATE.charts.main = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            ...getChartOptions('Actual (Y)', 'Predicted (Ŷ)'),
            plugins: {
                ...getChartOptions('Actual (Y)', 'Predicted (Ŷ)').plugins,
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const idx = ctx.dataIndex;
                            return `Obs ${idx + 1}: actual=${yData[idx].toFixed(3)}, predicted=${results.predictions[idx].toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update residuals chart
 * @param {Object} results - Regression results
 */
export function updateResidualsChart(results) {
    const canvas = document.getElementById('residualsChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colors = getChartColors();

    if (STATE.charts.residuals && typeof STATE.charts.residuals.destroy === 'function') {
        STATE.charts.residuals.destroy();
    }

    const residualsData = results.predictions.map((pred, i) => ({
        x: pred,
        y: results.residuals[i]
    }));

    const zeroLine = [
        { x: Math.min(...results.predictions), y: 0 },
        { x: Math.max(...results.predictions), y: 0 }
    ];

    STATE.charts.residuals = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Residuals',
                    data: residualsData,
                    backgroundColor: colors.accent,
                    pointRadius: 5,
                    pointHoverRadius: 7
                },
                {
                    label: 'Zero Line',
                    data: zeroLine,
                    type: 'line',
                    borderColor: colors.error,
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: colors.text } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const idx = ctx.dataIndex;
                            return `Obs ${idx + 1}: fitted=${results.predictions[idx].toFixed(3)}, residual=${results.residuals[idx].toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Fitted Values', color: colors.textMuted },
                    ticks: { color: colors.textMuted },
                    grid: { color: colors.border }
                },
                y: {
                    title: { display: true, text: 'Residuals', color: colors.textMuted },
                    ticks: { color: colors.textMuted },
                    grid: { color: colors.border }
                }
            }
        }
    });
}

/**
 * Update Q-Q plot
 * @param {Object} results - Regression results
 */
export function updateQQChart(results) {
    const canvas = document.getElementById('qqChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colors = getChartColors();

    if (STATE.charts.qq && typeof STATE.charts.qq.destroy === 'function') {
        STATE.charts.qq.destroy();
    }

    // Calculate Q-Q plot data
    const residuals = results.residuals;
    const n = residuals.length;

    // Sort residuals
    const sortedResiduals = [...residuals].sort((a, b) => a - b);

    // Calculate theoretical quantiles
    const theoreticalQuantiles = [];
    for (let i = 0; i < n; i++) {
        const p = (i + 0.5) / n;
        const z = Stats.normalInverse(p);
        theoreticalQuantiles.push(z);
    }

    // Standardize residuals
    const resMean = Stats.mean(residuals);
    const resStd = Stats.std(residuals, 1);

    let standardizedSorted;
    if (resStd === 0 || isNaN(resStd)) {
        standardizedSorted = sortedResiduals.map(() => 0);
    } else {
        standardizedSorted = sortedResiduals.map(r => (r - resMean) / resStd);
    }

    // Create scatter data
    const qqData = theoreticalQuantiles.map((z, i) => ({
        x: z,
        y: standardizedSorted[i]
    }));

    // Reference line
    const minVal = Math.min(...theoreticalQuantiles);
    const maxVal = Math.max(...theoreticalQuantiles);
    const referenceLine = [
        { x: minVal, y: minVal },
        { x: maxVal, y: maxVal }
    ];

    // Calculate correlation for normality assessment
    const correlation = calculateCorrelation(theoreticalQuantiles, standardizedSorted);

    // Assess normality
    const assessment = assessNormality(correlation);
    const correlationColor = getNormalityColor(correlation, colors);

    STATE.charts.qq = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Reference Line (y = x)',
                    data: referenceLine,
                    type: 'line',
                    borderColor: colors.textMuted,
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    borderDash: [6, 6]
                },
                {
                    label: 'Standardized Residuals',
                    data: qqData,
                    backgroundColor: colors.accent,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: colors.text,
                        title: {
                            display: true,
                            text: `Correlation: ${correlation.toFixed(4)} - ${assessment}`,
                            color: correlationColor,
                            font: { size: 11 }
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const idx = ctx.dataIndex;
                            return `Obs ${idx + 1}: theoretical=${theoreticalQuantiles[idx].toFixed(3)}, sample=${standardizedSorted[idx].toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Theoretical Quantiles (Standard Normal)', color: colors.textMuted },
                    ticks: { color: colors.textMuted },
                    grid: { color: colors.border }
                },
                y: {
                    title: { display: true, text: 'Sample Quantiles (Standardized Residuals)', color: colors.textMuted },
                    ticks: { color: colors.textMuted },
                    grid: { color: colors.border }
                }
            }
        }
    });
}

/**
 * Update coefficient chart (with Confidence Intervals)
 * @param {Object} results - Regression results
 */
export function updateCoefficientChart(results) {
    const canvas = document.getElementById('coefficientChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colors = getChartColors();

    if (STATE.charts.coefficients && typeof STATE.charts.coefficients.destroy === 'function') {
        STATE.charts.coefficients.destroy();
    }

    // Skip for LOESS (no fixed coefficients)
    if (results.method === 'loess') {
        ctx.font = '14px "Space Grotesk", sans-serif';
        ctx.fillStyle = colors.textMuted || '#999';
        ctx.textAlign = 'center';
        ctx.fillText('Coefficient plot not available for LOESS', canvas.width / 2, canvas.height / 2);
        return;
    }

    // Prepare data
    const labels = results.variableNames;
    const coeffs = results.coefficients;
    
    // Check if we have CI data
    const hasCI = results.confIntLower && results.confIntLower.length === coeffs.length;
    const lower = hasCI ? results.confIntLower : coeffs.map(c => c);
    const upper = hasCI ? results.confIntUpper : coeffs.map(c => c);

    // Floating bars for CI (low, high)
    // Using simple array for bar chart data [min, max]
    const ciData = labels.map((_, i) => [lower[i], upper[i]]);
    
    // Coefficient points
    const coefData = labels.map((_, i) => coeffs[i]);

    STATE.charts.coefficients = new Chart(ctx, {
        data: {
            labels: labels,
            datasets: [
                {
                    type: 'line',
                    label: 'Coefficient Estimate',
                    data: coefData,
                    backgroundColor: colors.primary,
                    borderColor: colors.primary,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    showLine: false, // Scatter-like appearance
                    order: 1
                },
                {
                    type: 'bar',
                    label: hasCI ? '95% Confidence Interval' : 'Value',
                    data: ciData,
                    backgroundColor: hexToRgba(colors.secondary, 0.2),
                    borderColor: colors.secondary,
                    borderWidth: 1,
                    borderSkipped: false,
                    barPercentage: 0.2, // Thin bars look like error bars
                    order: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: colors.text } },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            if (context.dataset.type === 'bar' && hasCI) {
                                return `95% CI: [${context.raw[0].toFixed(4)}, ${context.raw[1].toFixed(4)}]`;
                            }
                            if (typeof context.raw === 'number') {
                                return `Coef: ${context.raw.toFixed(4)}`;
                            }
                            return `Value: ${context.raw}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: { color: colors.grid },
                    ticks: { color: colors.text },
                    title: { display: true, text: 'Value', color: colors.textMuted }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: colors.text }
                }
            }
        }
    });
}

/**
 * Update leverage chart (Residuals vs Leverage)
 * @param {Object} results - Regression results
 */
export function updateLeverageChart(results) {
    const canvas = document.getElementById('leverageChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colors = getChartColors();

    if (STATE.charts.leverage && typeof STATE.charts.leverage.destroy === 'function') {
        STATE.charts.leverage.destroy();
    }

    // Only available for OLS with leverage data
    if (!results.leverage || results.method !== 'ols') {
        // Show placeholder message
        ctx.font = '14px "Space Grotesk", sans-serif';
        ctx.fillStyle = colors.textMuted || '#999';
        ctx.textAlign = 'center';
        ctx.fillText('Leverage plot only available for OLS regression', canvas.width / 2, canvas.height / 2);
        return;
    }

    const leverage = results.leverage;
    const residuals = results.residuals;
    const n = results.n;
    const p = results.k + 1; // parameters including intercept

    // Calculate thresholds for high leverage
    const leverageThreshold1 = 2 * p / n;  // Common threshold: 2p/n
    const leverageThreshold2 = 3 * p / n;  // More stringent: 3p/n

    // Create scatter data
    const scatterData = leverage.map((h, i) => ({
        x: h,
        y: residuals[i]
    }));

    // Reference lines
    const maxLeverage = Math.max(...leverage) * 1.1;
    const maxResidual = Math.max(...residuals.map(Math.abs)) * 1.1;
    const minResidual = -maxResidual;

    const zeroLine = [
        { x: 0, y: 0 },
        { x: maxLeverage, y: 0 }
    ];

    const leverageLine1 = [
        { x: leverageThreshold1, y: minResidual },
        { x: leverageThreshold1, y: maxResidual }
    ];

    const leverageLine2 = [
        { x: leverageThreshold2, y: minResidual },
        { x: leverageThreshold2, y: maxResidual }
    ];

    // Color points by influence (high leverage OR high residual)
    const pointColors = leverage.map((h, i) => {
        const hasHighLeverage = h > leverageThreshold2;
        const hasHighResidual = Math.abs(residuals[i]) > 2;
        if (hasHighLeverage || hasHighResidual) {
            return colors.error || '#ef4444';
        }
        if (h > leverageThreshold1) {
            return colors.line || '#f59e0b';
        }
        return colors.accent || '#22c55e';
    });

    const pointRadius = leverage.map((h) => {
        if (h > leverageThreshold2) return 8;
        if (h > leverageThreshold1) return 6;
        return 4;
    });

    STATE.charts.leverage = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: `Threshold (2p/n = ${leverageThreshold1.toFixed(3)})`,
                    data: leverageLine1,
                    type: 'line',
                    borderColor: colors.line || '#f59e0b',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false,
                    borderDash: [5, 5]
                },
                {
                    label: `Threshold (3p/n = ${leverageThreshold2.toFixed(3)})`,
                    data: leverageLine2,
                    type: 'line',
                    borderColor: colors.error || '#ef4444',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false,
                    borderDash: [5, 5]
                },
                {
                    label: 'Zero Line',
                    data: zeroLine,
                    type: 'line',
                    borderColor: colors.textMuted || '#999',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Observations',
                    data: scatterData,
                    backgroundColor: pointColors,
                    pointRadius: pointRadius,
                    pointHoverRadius: (ctx) => pointRadius[ctx.dataIndex] + 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: colors.text } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const idx = ctx.dataIndex;
                            return `Obs ${idx + 1}: h=${leverage[idx].toFixed(3)}, residual=${residuals[idx].toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Leverage (hᵢᵢ)', color: colors.textMuted },
                    ticks: { color: colors.textMuted },
                    grid: { color: colors.border },
                    min: 0,
                    max: maxLeverage
                },
                y: {
                    title: { display: true, text: 'Residuals', color: colors.textMuted },
                    ticks: { color: colors.textMuted },
                    grid: { color: colors.border }
                }
            }
        }
    });
}

/**
 * Get chart options
 * @param {string} xLabel - X-axis label
 * @param {string} yLabel - Y-axis label
 * @returns {Object} Chart.js options
 */
export function getChartOptions(xLabel, yLabel) {
    const colors = getChartColors();
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: colors.text } },
            tooltip: {
                callbacks: {
                    label: (ctx) => `(${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(3)})`
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: { display: true, text: xLabel, color: colors.textMuted },
                ticks: { color: colors.textMuted },
                grid: { color: colors.border }
            },
            y: {
                title: { display: true, text: yLabel, color: colors.textMuted },
                ticks: { color: colors.textMuted },
                grid: { color: colors.border }
            }
        }
    };
}

// ============================================================================
// CHART EXPORT
// ============================================================================

/**
 * Update regularization path chart
 * @param {Object} pathResults - Results from tracePath
 */
export function updatePathChart(pathResults) {
    const canvas = document.getElementById('pathChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colors = getChartColors();

    if (STATE.charts.path && typeof STATE.charts.path.destroy === 'function') {
        STATE.charts.path.destroy();
    }

    const { lambdas, coefficients, r_squared, aic, bic } = pathResults;
    const varNames = STATE.xVariables;
    
    // X-axis is log(lambda). Handle infinity (the first lambda).
    const logLambdas = lambdas.map(l => (l === Infinity || l === null) ? null : Math.log10(l));
    
    // Filter out the null logLambdas for the chart data
    const validIndices = logLambdas.map((l, i) => l !== null ? i : -1).filter(i => i !== -1);
    const chartX = validIndices.map(i => logLambdas[i]);
    
    // Create one dataset per predictor
    const datasets = varNames.map((name, coefIdx) => {
        // coefficients is [lambda_idx][coef_idx]
        // Note: coef_idx 0 is intercept, slopes start at 1
        const data = validIndices.map(lambdaIdx => coefficients[lambdaIdx][coefIdx + 1]);
        
        return {
            label: name,
            data: data,
            fill: false,
            borderColor: getPredictorColor(coefIdx),
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
        };
    });

    STATE.charts.path = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartX,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: colors.text, boxWidth: 12, font: { size: 10 } }
                },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const idx = validIndices[items[0].dataIndex];
                            return `Step ${idx + 1}: \u03BB = ${lambdas[idx].toExponential(4)}`;
                        },
                        afterBody: (items) => {
                            const idx = validIndices[items[0].dataIndex];
                            return `R\u00B2: ${r_squared[idx].toFixed(4)}\nAIC: ${aic[idx].toFixed(2)}\nBIC: ${bic[idx].toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Log10(\u03BB)', color: colors.textMuted },
                    ticks: { color: colors.textMuted },
                    grid: { color: colors.border },
                    reverse: true // Path usually shown from large lambda to small
                },
                y: {
                    title: { display: true, text: 'Coefficients', color: colors.textMuted },
                    ticks: { color: colors.textMuted },
                    grid: { color: colors.border }
                }
            }
        }
    });
}

/**
 * Export all charts as a single PNG image
 */
export function exportChartsAsPNG() {
    if (!STATE.charts.main) {
        showToast('No charts to export', 'warning');
        return;
    }

    const canvas = document.createElement('canvas');
    const mainCanvas = document.getElementById('mainChart');
    const residualsCanvas = document.getElementById('residualsChart');
    const qqCanvas = document.getElementById('qqChart');
    const leverageCanvas = document.getElementById('leverageChart');
    const coefficientCanvas = document.getElementById('coefficientChart');
    const pathCanvas = document.getElementById('pathChart');
    const hasPath = STATE.pathResults && pathCanvas && document.getElementById('pathChartContainer').style.display !== 'none';

    let totalHeight = mainCanvas.height + residualsCanvas.height + qqCanvas.height + leverageCanvas.height + coefficientCanvas.height + 150;
    if (hasPath) totalHeight += pathCanvas.height + 50;

    const width = Math.max(mainCanvas.width, residualsCanvas.width, qqCanvas.width, leverageCanvas.width, coefficientCanvas.width, hasPath ? pathCanvas.width : 0);
    
    canvas.width = width;
    canvas.height = totalHeight;
    const ctx = canvas.getContext('2d');

    // Background
    const bgColor = getComputedStyle(document.body).getPropertyValue('--bg-card').trim() || '#ffffff';
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, totalHeight);

    // Title
    const textColor = getComputedStyle(document.body).getPropertyValue('--text-primary').trim() || '#000000';
    ctx.fillStyle = textColor;
    ctx.font = '16px "Space Grotesk", sans-serif';
    ctx.fillText('Linear Regression Analysis', 20, 25);

    // Draw charts
    let yOffset = 40;
    ctx.drawImage(mainCanvas, 0, yOffset);
    yOffset += mainCanvas.height + 10;
    ctx.drawImage(residualsCanvas, 0, yOffset);
    yOffset += residualsCanvas.height + 10;
    ctx.drawImage(qqCanvas, 0, yOffset);
    yOffset += qqCanvas.height + 10;
    ctx.drawImage(leverageCanvas, 0, yOffset);
    yOffset += leverageCanvas.height + 10;
    ctx.drawImage(coefficientCanvas, 0, yOffset);
    
    if (hasPath) {
        yOffset += coefficientCanvas.height + 10;
        ctx.drawImage(pathCanvas, 0, yOffset);
    }

    // Export
    const link = document.createElement('a');
    link.download = 'linear-regression-charts.png';
    link.href = canvas.toDataURL('image/png');
    link.click();

    showToast('Charts exported as PNG', 'success');
}

/**
 * Update the Correlation Heatmap (Exploratory Data Analysis)
 */
export function updateCorrelationHeatmap() {
    const container = document.getElementById('correlationHeatmapContainer');
    if (!container || !STATE.numericColumns || STATE.numericColumns.length === 0) return;

    const cols = STATE.numericColumns;
    const n = cols.length;
    
    // Build table header
    let html = '<table class="correlation-table" style="border-collapse: separate; border-spacing: 2px; width: 100%; min-width: 600px; font-family: inherit;">';
    html += '<thead><tr><th style="padding: 10px; background: transparent;"></th>';
    cols.forEach(col => {
        html += `<th style="padding: 10px; font-size: 0.75rem; color: var(--text-muted); text-align: center; max-width: 80px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(col)}">${escapeHtml(col)}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Compute and build rows
    for (let i = 0; i < n; i++) {
        const rowCol = cols[i];
        html += `<tr><th style="padding: 10px; font-size: 0.75rem; color: var(--text-muted); text-align: right; white-space: nowrap; max-width: 120px; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(rowCol)}">${escapeHtml(rowCol)}</th>`;
        
        for (let j = 0; j < n; j++) {
            const colCol = cols[j];
            let corr = 1.0;
            
            if (i === j) {
                corr = 1.0;
            } else {
                const xData = STATE.rawData.map(r => r[rowCol]);
                const yData = STATE.rawData.map(r => r[colCol]);
                corr = Stats.correlation(xData, yData);
            }

            // Determine color based on correlation
            // Positive: Blue/Green, Negative: Red/Orange
            let bgColor;
            let textColor = '#fff';
            const alpha = Math.abs(corr);
            
            if (corr > 0) {
                // Blue theme for positive
                bgColor = `rgba(59, 130, 246, ${alpha})`;
                if (alpha < 0.4) textColor = 'var(--text-primary)';
            } else {
                // Red theme for negative
                bgColor = `rgba(239, 68, 68, ${alpha})`;
                if (alpha < 0.4) textColor = 'var(--text-primary)';
            }

            html += `<td style="background-color: ${bgColor}; color: ${textColor}; padding: 12px 8px; text-align: center; font-size: 0.8125rem; font-weight: 600; border-radius: 4px; transition: transform 0.1s;" title="${escapeHtml(rowCol)} vs ${escapeHtml(colCol)}: ${corr.toFixed(4)}">
                ${corr.toFixed(2)}
            </td>`;
        }
        html += '</tr>';
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get a color for a predictor line
 */
function getPredictorColor(index) {
    const palette = [
        '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', 
        '#ec4899', '#06b6d4', '#f97316', '#6366f1', '#84cc16'
    ];
    return palette[index % palette.length];
}

/**
 * Get method label for display
 */
function getMethodLabel(results) {
    if (results.method === 'ols') return 'OLS';
    if (results.method === 'wls') return 'WLS';
    if (results.method === 'ridge') return `Ridge (\u03BB=${results.lambda?.toFixed(2) || 'N/A'})`;
    if (results.method === 'lasso') return `Lasso (\u03BB=${results.lambda?.toFixed(2) || 'N/A'})`;
    if (results.method === 'elastic_net') return `Elastic Net (\u03BB=${results.lambda?.toFixed(2) || 'N/A'}, \u03B1=${results.alpha?.toFixed(2) || 'N/A'})`;
    if (results.method === 'loess') return `LOESS (span=${results.span?.toFixed(2) || 'N/A'}, degree=${results.degree || 1})`;
    return 'Regression';
}

/**
 * Get color for regression method
 */
function getMethodColor(method, colors) {
    if (method === 'ols') return colors.line || '#06b6d4';
    if (method === 'wls') return '#f97316';
    if (method === 'ridge') return '#10b981';
    if (method === 'lasso') return '#f59e0b';
    if (method === 'elastic_net') return '#8b5cf6';
    if (method === 'loess') return '#ec4899';
    return '#06b6d4';
}

/**
 * Compute prediction interval band via WASM (Rust) for simple regression.
 * Generates a grid of x-values and calls the Rust PI function for accurate results.
 * @returns {{ upper: Array, lower: Array }} or null on failure
 */
function computePredictionBandWasm(xData, yData, xMin, xMax) {
    if (!WasmRegression.isReady()) return null;

    // Check if olsPredictionIntervals is available (may not be in cached version)
    if (typeof WasmRegression.olsPredictionIntervals !== 'function') {
        console.warn('[Charts] WasmRegression.olsPredictionIntervals not available. This may be due to browser caching. Please hard refresh (Ctrl+Shift+R).');
        return null;
    }

    const bandPoints = 50;
    const step = (xMax - xMin) / bandPoints;
    const gridX = Array.from({ length: bandPoints + 1 }, (_, i) => xMin + step * i);

    try {
        const piResult = WasmRegression.olsPredictionIntervals(yData, [xData], [gridX], 0.05);
        const pi = JSON.parse(piResult);

        if (pi.error) {
            console.warn('WASM PI error:', pi.error);
            return null;
        }

        return {
            upper: gridX.map((x, i) => ({ x, y: pi.upper_bound[i] })),
            lower: gridX.map((x, i) => ({ x, y: pi.lower_bound[i] }))
        };
    } catch (e) {
        console.warn('[Charts] Failed to compute prediction intervals via WASM:', e);
        return null;
    }
}

/**
 * Create confidence band for simple regression
 */
function createConfidenceBand(xData, xMin, xMax, b0, b1, results) {
    const xMean = Stats.mean(xData);
    const ssx = xData.reduce((sum, x) => sum + Math.pow(x - xMean, 2), 0);
    const seFit = (x) => results.stdError * Math.sqrt(1/results.n + Math.pow(x - xMean, 2) / ssx);
    const tCrit = Stats.tCritical(0.05, results.df);

    const bandPoints = 50;
    const xRange = xMax - xMin;
    const upperBand = [];
    const lowerBand = [];

    for (let i = 0; i <= bandPoints; i++) {
        const x = xMin + (xRange * i / bandPoints);
        const y = b0 + b1 * x;
        const se = seFit(x);
        const margin = tCrit * se;
        upperBand.push({ x: x, y: y + margin });
        lowerBand.push({ x: x, y: y - margin });
    }

    return { upper: upperBand, lower: lowerBand };
}

/**
 * Create prediction band for multiple regression
 */
function createPredictionBand(minVal, maxVal, rmse) {
    const margin = 1.96 * rmse;
    const bandPoints = 50;
    const range = maxVal - minVal;
    const upperBand = [];
    const lowerBand = [];

    for (let i = 0; i <= bandPoints; i++) {
        const x = minVal + (range * i / bandPoints);
        upperBand.push({ x: x, y: x + margin });
        lowerBand.push({ x: x, y: x - margin });
    }

    return { upper: upperBand, lower: lowerBand };
}

/**
 * Calculate correlation coefficient
 */
function calculateCorrelation(x, y) {
    const meanX = Stats.mean(x);
    const meanY = Stats.mean(y);
    let numerator = 0, denomX = 0, denomY = 0;

    for (let i = 0; i < x.length; i++) {
        numerator += (x[i] - meanX) * (y[i] - meanY);
        denomX += (x[i] - meanX) ** 2;
        denomY += (y[i] - meanY) ** 2;
    }

    const denomProduct = denomX * denomY;
    if (denomProduct === 0 || isNaN(denomProduct)) return 0;
    return numerator / Math.sqrt(denomProduct);
}

/**
 * Assess normality based on correlation
 */
function assessNormality(correlation) {
    if (correlation > 0.98) return 'Good (residuals appear normal)';
    if (correlation > 0.95) return 'Moderate (some deviation from normal)';
    return 'Poor (residuals deviate from normality)';
}

/**
 * Get color based on normality assessment
 */
function getNormalityColor(correlation, colors) {
    if (correlation > 0.98) return colors.accent;
    if (correlation > 0.95) return colors.line || '#888';
    return colors.error;
}
