// ============================================================================
// charts.js - Chart management and visualization
// ============================================================================

import { STATE, escapeHtml, hexToRgba, getChartColors } from './utils.js';
import { Stats } from './core.js';

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

    const yData = STATE.rawData.map(row => row[STATE.yVariable]);

    if (results.k === 1) {
        // Simple regression: scatter with regression line and confidence band
        updateSimpleRegressionChart(ctx, results, yData, colors);
    } else {
        // Multiple regression: actual vs predicted
        updateMultipleRegressionChart(ctx, results, yData, colors);
    }
}

/**
 * Update simple regression chart (scatter with line)
 */
function updateSimpleRegressionChart(ctx, results, yData, colors) {
    const xVarName = STATE.xVariables[0];
    const xData = STATE.rawData.map(row => row[xVarName]);

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
    const ciLabel = results.confidenceIntervals ? '95% Confidence Band (Mean Response)' :
                    results.method === 'ridge' || results.method === 'lasso' ?
                    'CI not available for regularized regression' : 'No Confidence Interval';

    const titleEl = document.getElementById('mainChartTitle');
    if (titleEl) {
        titleEl.textContent = `Scatter Plot: ${escapeHtml(xVarName)} vs ${escapeHtml(STATE.yVariable)} — ${methodLabel} — ${ciLabel}`;
    }

    let datasets = [];

    // Add 95% confidence band for OLS
    if (results.method === 'ols' && results.stdError && results.df) {
        const confidenceBand = createConfidenceBand(xData, xMin, xMax, b0, b1, results);
        const ciColor = colors.line || '#06b6d4';

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
        options: getChartOptions(escapeHtml(xVarName), escapeHtml(STATE.yVariable))
    });
}

/**
 * Update multiple regression chart (actual vs predicted)
 */
function updateMultipleRegressionChart(ctx, results, yData, colors) {
    const methodLabel = getMethodLabel(results);

    let ciLabel = '';
    if (results.method === 'ols' && results.stdError) {
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

    // Add 95% prediction band for OLS
    if (results.method === 'ols' && results.stdError && results.rmse) {
        const predictionBand = createPredictionBand(minVal, maxVal, results.rmse);
        const ciColor = colors.line || '#06b6d4';

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
        options: getChartOptions('Actual (Y)', 'Predicted (Ŷ)')
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
                        label: (ctx) => `(${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(3)})`
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
                        label: (ctx) => `(${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(3)})`
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

    const width = Math.max(mainCanvas.width, residualsCanvas.width, qqCanvas.width);
    const height = mainCanvas.height + residualsCanvas.height + qqCanvas.height + 100;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Background
    const bgColor = getComputedStyle(document.body).getPropertyValue('--bg-card').trim() || '#ffffff';
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);

    // Title
    const textColor = getComputedStyle(document.body).getPropertyValue('--text-primary').trim() || '#000000';
    ctx.fillStyle = textColor;
    ctx.font = '16px "Space Grotesk", sans-serif';
    ctx.fillText('Linear Regression Analysis', 20, 25);

    // Draw charts
    ctx.drawImage(mainCanvas, 0, 40);
    ctx.drawImage(residualsCanvas, 0, mainCanvas.height + 50);
    ctx.drawImage(qqCanvas, 0, mainCanvas.height + residualsCanvas.height + 60);

    // Export
    const link = document.createElement('a');
    link.download = 'linear-regression-charts.png';
    link.href = canvas.toDataURL('image/png');
    link.click();

    showToast('Charts exported as PNG', 'success');
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get method label for display
 */
function getMethodLabel(results) {
    if (results.method === 'ols') return 'OLS';
    if (results.method === 'ridge') return `Ridge (λ=${results.lambda?.toFixed(2) || 'N/A'})`;
    if (results.method === 'lasso') return `Lasso (λ=${results.lambda?.toFixed(2) || 'N/A'})`;
    if (results.method === 'elastic_net') return `Elastic Net (λ=${results.lambda?.toFixed(2) || 'N/A'}, α=${results.alpha?.toFixed(2) || 'N/A'})`;
    return 'Regression';
}

/**
 * Get color for regression method
 */
function getMethodColor(method, colors) {
    if (method === 'ols') return colors.line || '#06b6d4';
    if (method === 'ridge') return '#10b981';
    if (method === 'lasso') return '#f59e0b';
    if (method === 'elastic_net') return '#8b5cf6';
    return '#06b6d4';
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
