// ============================================================================
// ui.js - DOM manipulation and UI management
// ============================================================================

import { STATE, updateState, escapeHtml, formatPValue, showToast, getCurrentTheme } from './utils.js';
import { updateCharts } from './charts.js';
import { runDiagnostics } from './diagnostics.js';
import * as FeatureImportance from './featureImportance.js';
import {
    calculateRegression,
    calculateRidgeRegression,
    calculateLassoRegression,
    calculateElasticNetRegression,
    calculateWlsRegression,
    calculateLoessRegression,
    calculatePolynomialRegression,
    calculatePolynomialRidge,
    calculatePolynomialLasso,
    calculatePolynomialElasticNet,
    WasmRegression
} from './core.js';
import { formatMethodName } from './regularized.js';
import { getExampleDescription } from './data.js';

// ============================================================================
// DATA PREVIEW UI
// ============================================================================

/**
 * Update the data preview table
 */
export function updateDataPreview() {
    const table = document.getElementById('previewTable');
    if (!table) return;

    const previewRows = STATE.rawData.slice(0, 10);
    const headers = STATE.headers;

    // Build header row
    let html = '<thead><tr><th>#</th>';
    headers.forEach(h => {
        html += `<th>${escapeHtml(h)}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Build data rows
    previewRows.forEach((row, idx) => {
        html += `<tr><td>${idx + 1}</td>`;
        headers.forEach(h => {
            const val = row[h];
            html += `<td>${typeof val === 'number' ? val.toFixed(4) : escapeHtml(String(val))}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';

    table.innerHTML = html;

    const totalRowsEl = document.getElementById('totalRows');
    if (totalRowsEl) {
        totalRowsEl.textContent = STATE.rawData.length;
    }
}

/**
 * Update column selectors for variable selection
 */
export function updateColumnSelectors() {
    const container = document.getElementById('columnSelectors');
    if (!container) return;

    const numericCols = STATE.numericColumns;

    if (numericCols.length < 2) {
        container.innerHTML = '<p style="color: var(--text-muted);">Need at least 2 numeric columns</p>';
        return;
    }

    // Default selections
    if (!STATE.yVariable) {
        updateState({ yVariable: numericCols[numericCols.length - 1] });
    }
    if (STATE.xVariables.length === 0) {
        updateState({ xVariables: [numericCols[numericCols.length - 2]] });
    }

    let html = '';

    // Y Variable Selector
    html += '<div class="selector-group">';
    html += '<label>Y Variable (Response)</label>';
    html += '<select id="yVarSelect">';
    numericCols.forEach(col => {
        const selected = col === STATE.yVariable ? 'selected' : '';
        html += `<option value="${escapeHtml(col)}" ${selected}>${escapeHtml(col)}</option>`;
    });
    html += '</select>';
    html += '</div>';

    // X Variables (Multi-select)
    html += '<div class="selector-group">';
    html += '<label>X Variables (Predictors)</label>';
    html += '<div class="variable-checkboxes" id="xVarCheckboxes">';

    numericCols.forEach(col => {
        if (col === STATE.yVariable) return;
        const checked = STATE.xVariables.includes(col) ? 'checked' : '';
        html += `<label class="variable-checkbox">
            <input type="checkbox" value="${escapeHtml(col)}" ${checked}>
            <span>${escapeHtml(col)}</span>
        </label>`;
    });

    html += '</div>';
    html += '<button class="tool-btn" id="selectAllXBtn" style="margin-top: 8px; padding: 6px 12px; font-size: 0.75rem;">Select All</button>';
    html += '</div>';

    container.innerHTML = html;

    // Show Run Regression button
    const runBtn = document.getElementById('runRegressionBtn');
    if (runBtn) runBtn.style.display = 'block';

    const methodPanel = document.getElementById('regressionMethodPanel');
    if (methodPanel) methodPanel.style.display = 'block';

    const cvPanel = document.getElementById('cvPanel');
    if (cvPanel) cvPanel.style.display = 'block';

    // Refresh method-specific panels (e.g., WLS weights selector) with current data columns
    updateWlsWeightsSelector();

    // Add event listeners
    setTimeout(() => {
        const ySelect = document.getElementById('yVarSelect');
        if (ySelect) {
            ySelect.addEventListener('change', (e) => {
                updateState({ yVariable: e.target.value });
                updateColumnSelectors();
            });
        }

        const xCheckboxes = document.getElementById('xVarCheckboxes');
        if (xCheckboxes) {
            xCheckboxes.addEventListener('change', (e) => {
                if (e.target.type === 'checkbox') {
                    updateXVariablesFromDOM();
                }
            });
        }

        const selectAllBtn = document.getElementById('selectAllXBtn');
        if (selectAllBtn) {
            selectAllBtn.addEventListener('click', toggleSelectAllX);
        }
    }, 0);
}

/**
 * Update X variables from DOM checkboxes
 */
function updateXVariablesFromDOM() {
    const checkboxes = document.querySelectorAll('#xVarCheckboxes input[type="checkbox"]:checked');
    STATE.xVariables = Array.from(checkboxes).map(cb => cb.value);
}

/**
 * Toggle select all X variables
 */
function toggleSelectAllX() {
    const checkboxes = document.querySelectorAll('#xVarCheckboxes input[type="checkbox"]');
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    checkboxes.forEach(cb => {
        cb.checked = !allChecked;
    });
    updateXVariablesFromDOM();
}

// ============================================================================
// RESULTS DISPLAY
// ============================================================================

/**
 * Update all results displays
 * @param {Object} results - Regression results
 */
export function updateResultsDisplay(results) {
    const isRegularized = results.method === 'ridge' || results.method === 'lasso' || results.method === 'elastic_net';

    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) resultsSection.style.display = 'block';

    // Show diagnostic buttons
    const diagnosticButtons = document.getElementById('diagnosticButtons');
    if (diagnosticButtons) diagnosticButtons.style.display = 'block';

    // Show and update feature importance section (for OLS/WLS with multiple predictors)
    const featureImportanceToggle = document.getElementById('featureImportanceToggle');
    const featureImportanceContent = document.getElementById('featureImportanceContent');

    if (featureImportanceToggle && featureImportanceContent) {
        if ((results.method === 'ols' || results.method === 'wls') && STATE.xVariables.length >= 2) {
            // Auto-expand and calculate feature importance
            featureImportanceToggle.classList.remove('collapsed');
            featureImportanceToggle.setAttribute('aria-expanded', 'true');
            featureImportanceContent.classList.remove('collapsed');
            FeatureImportance.updateFeatureImportanceDisplay(results);
        } else {
            // Collapse and show message
            featureImportanceToggle.classList.add('collapsed');
            featureImportanceToggle.setAttribute('aria-expanded', 'false');
            featureImportanceContent.classList.add('collapsed');
        }
    }

    // Update statistics cards
    updateStatisticsCards(results, isRegularized);

    // Update equation
    updateEquation(results);

    // Update coefficients table
    updateCoefficientsTable(results, isRegularized);

    // Update residuals table
    updateResidualsTable(results);

    // Update charts
    updateCharts(results);

    // Initialize interactive predictor
    initPredictionTool(results);
}

/**
 * Update statistics cards
 */
function updateStatisticsCards(results, isRegularized) {
    const rSquaredEl = document.getElementById('rSquared');
    const adjRSquaredEl = document.getElementById('adjRSquared');
    const rmseEl = document.getElementById('rmse');
    const maeEl = document.getElementById('mae');
    const fStatEl = document.getElementById('fStat');
    const pValueEl = document.getElementById('pValue');
    const aicEl = document.getElementById('aic');
    const bicEl = document.getElementById('bic');
    const logLikelihoodEl = document.getElementById('logLikelihood');

    if (rSquaredEl) rSquaredEl.textContent = isNaN(results.rSquared) ? 'N/A' : results.rSquared.toFixed(4);
    if (adjRSquaredEl) adjRSquaredEl.textContent = isNaN(results.adjRSquared) ? 'N/A' : results.adjRSquared.toFixed(4);
    if (rmseEl) rmseEl.textContent = isNaN(results.rmse) ? 'N/A' : results.rmse.toFixed(4);
    if (maeEl) maeEl.textContent = isNaN(results.mae) ? 'N/A' : results.mae.toFixed(4);

    if (isRegularized) {
        if (fStatEl) fStatEl.textContent = 'N/A';
        if (pValueEl) pValueEl.textContent = 'N/A';
    } else {
        if (fStatEl) fStatEl.textContent = results.fStat?.toFixed(4) || 'N/A';
        if (pValueEl) pValueEl.textContent = formatPValue(results.fPValue);
    }

    // Model selection criteria (available for all regression methods)
    if (aicEl) {
        if (results.aic !== undefined && !isNaN(results.aic)) {
            aicEl.textContent = results.aic.toFixed(2);
        } else {
            aicEl.textContent = 'N/A';
        }
    }
    if (bicEl) {
        if (results.bic !== undefined && !isNaN(results.bic)) {
            bicEl.textContent = results.bic.toFixed(2);
        } else {
            bicEl.textContent = 'N/A';
        }
    }
    if (logLikelihoodEl) {
        if (results.logLikelihood !== undefined && !isNaN(results.logLikelihood)) {
            logLikelihoodEl.textContent = results.logLikelihood.toFixed(2);
        } else {
            logLikelihoodEl.textContent = 'N/A';
        }
    }
}

/**
 * Update regression equation display
 */
function updateEquation(results) {
    const equationEl = document.getElementById('equation');
    if (!equationEl) return;

    // Handle LOESS (no traditional equation)
    if (results.method === 'loess') {
        const xVars = STATE.xVariables;
        const yVar = STATE.yVariable;

        let equationContent = `
            <div style="margin-bottom: 8px;">
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Method:</div>
                <div><span style="color: var(--accent-educational); font-weight: 600;">LOESS</span> — Locally Estimated Scatterplot Smoothing</div>
            </div>
            <div style="margin-bottom: 8px;">
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Model:</div>
                <div>fitted(<strong>${escapeHtml(yVar)}</strong>) = smooth(${xVars.map(v => `<strong>${escapeHtml(v)}</strong>`).join(', ')})</div>
            </div>
        `;

        // Build parameters section
        let params = '<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color);">';
        params += '<div style="font-size: 0.6875rem; color: var(--text-muted); margin-bottom: 8px;">Parameters:</div>';
        params += `<div><span style="color: var(--text-muted);">Span</span> = <strong>${results.span.toFixed(2)}</strong> (smoothing window)</div>`;
        params += `<div><span style="color: var(--text-muted);">Degree</span> = <strong>${results.degree}</strong> (local polynomial)`;
        if (results.degree === 0) params += ' — constant';
        else if (results.degree === 1) params += ' — linear';
        else if (results.degree === 2) params += ' — quadratic';
        params += '</div>';
        params += `<div><span style="color: var(--text-muted);">Surface</span> = <strong>${results.surface}</strong></div>`;
        if (results.robustIterations > 0) {
            params += `<div><span style="color: var(--text-muted);">Robust iterations</span> = <strong>${results.robustIterations}</strong> (outlier resistance)</div>`;
        }
        params += '</div>';

        equationEl.innerHTML = equationContent + params;
        return;
    }

    const names = results.variableNames;
    const coeffs = results.coefficients;
    const pValues = results.pValues;
    const isSimple = results.k === 1;
    const isRegularized = results.method === 'ridge' || results.method === 'lasso' || results.method === 'elastic_net';

    // Build symbolic form
    let symbolic = '';
    if (isSimple) {
        const xName = names[1];
        symbolic = `<span style="color: var(--accent-educational);">y</span> = m×${escapeHtml(xName)} + b`;
    } else {
        let parts = ['b'];
        for (let i = 1; i < names.length; i++) {
            parts.push(`m<sub>${i}</sub>×${escapeHtml(names[i])}`);
        }
        symbolic = `<span style="color: var(--accent-educational);">y</span> = ${parts.join(' + ')}`;
    }

    // Build numeric form
    let numeric = `<span style="color: var(--accent-educational);">ŷ</span> = `;

    coeffs.forEach((coef, i) => {
        const name = escapeHtml(names[i]);
        const sign = coef >= 0 ? (i === 0 ? '' : ' + ') : ' - ';
        const absCoef = Math.abs(coef).toFixed(4);

        let sig = '';
        if (!isRegularized && pValues) {
            const pv = pValues[i];
            if (pv < 0.001) sig = '***';
            else if (pv < 0.01) sig = '**';
            else if (pv < 0.05) sig = '*';
        }

        const isZero = Math.abs(coef) < 1e-10;
        const zeroIndicator = isZero ? ' <span style="color: var(--text-muted);">(zero)</span>' : '';

        if (i === 0) {
            numeric += `<span title="Intercept (b)">${coef.toFixed(4)}${sig}</span>`;
        } else {
            const label = isSimple ? 'm' : `m<sub>${i}</sub>`;
            numeric += `${sign}<span title="${name} coefficient (${label})">${absCoef}×${name}${sig}${zeroIndicator}</span>`;
        }
    });

    // Build parameters section
    let params = '<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color);">';
    params += '<div style="font-size: 0.6875rem; color: var(--text-muted); margin-bottom: 8px;">Parameters:</div>';

    coeffs.forEach((coef, i) => {
        let sig = '';
        if (!isRegularized && pValues) {
            const pv = pValues[i];
            if (pv < 0.001) sig = '***';
            else if (pv < 0.01) sig = '**';
            else if (pv < 0.05) sig = '*';
        }

        const isZero = Math.abs(coef) < 1e-10;
        const zeroIndicator = isZero ? ' <span style="color: var(--text-muted);">(zero)</span>' : '';

        if (i === 0) {
            params += `<div><span style="color: var(--text-muted);">b</span> (intercept) = <strong>${coef.toFixed(4)}</strong>${sig}</div>`;
        } else {
            const label = isSimple ? 'm' : `m<sub>${i}</sub>`;
            params += `<div><span style="color: var(--text-muted);">${label}</span> (${escapeHtml(names[i])}) = <strong>${coef.toFixed(4)}</strong>${sig}${zeroIndicator}</div>`;
        }
    });

    // Add lambda/alpha info for regularized regression
    if (isRegularized) {
        params += `<div style="margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border-color);">`;
        params += `<span style="color: var(--text-muted);">Lambda (λ)</span> = <strong>${results.lambda.toFixed(4)}</strong>`;
        if (results.alpha !== undefined) {
            params += `<br><span style="color: var(--text-muted);">Alpha (α)</span> = <strong>${results.alpha.toFixed(2)}</strong>`;
        }
        if (results.nNonzero !== undefined) {
            params += `<br><span style="color: var(--text-muted);">Non-zero coeffs</span> = <strong>${results.nNonzero}</strong>`;
        }
        params += `</div>`;
    }

    params += '</div>';

    equationEl.innerHTML = `
        <div style="margin-bottom: 8px;">
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Form:</div>
            ${symbolic}
        </div>
        <div style="margin-bottom: 8px;">
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Fitted:</div>
            ${numeric}
        </div>
        ${params}
    `;
}

/**
 * Update coefficients table
 */
function updateCoefficientsTable(results, isRegularized = false) {
    const tbody = document.getElementById('coefficientsBody');
    const table = document.getElementById('coefficientsTable');
    if (!tbody || !table) return;

    // Handle LOESS - show parameters instead of coefficients
    if (results.method === 'loess') {
        let html = '';
        html += `<tr>
            <td>Span (smoothing)</td>
            <td>${results.span.toFixed(2)}</td>
        </tr>`;
        html += `<tr>
            <td>Degree (polynomial)</td>
            <td>${results.degree}</td>
        </tr>`;
        html += `<tr>
            <td>Surface method</td>
            <td>${results.surface}</td>
        </tr>`;
        html += `<tr>
            <td>Robust iterations</td>
            <td>${results.robustIterations}</td>
        </tr>`;
        tbody.innerHTML = html;

        const thead = table.querySelector('thead tr');
        if (thead) {
            thead.innerHTML = '<th>Parameter</th><th>Value</th>';
        }
        return;
    }

    const names = results.variableNames;
    const coeffs = results.coefficients;

    if (isRegularized) {
        let html = '';
        names.forEach((name, i) => {
            const signClass = coeffs[i] >= 0 ? 'positive' : 'negative';
            const isZero = Math.abs(coeffs[i]) < 1e-10;
            html += `<tr>
                <td>${escapeHtml(name)}</td>
                <td class="${signClass}">${coeffs[i].toFixed(4)}${isZero ? ' (zero)' : ''}</td>
            </tr>`;
        });
        tbody.innerHTML = html;

        const thead = table.querySelector('thead tr');
        if (thead) {
            thead.innerHTML = '<th>Variable</th><th>Coefficient</th>';
        }
        return;
    }

    // OLS / WLS full table
    const stdErrs = results.stdErrors;
    const tStats = results.tStats;
    const pValues = results.pValues;
    const ci = results.confidenceIntervals;
    const hasCi = ci && ci.length > 0;
    const hasStdErr = stdErrs && stdErrs.length > 0;
    const hasTStats = tStats && tStats.length > 0;
    const hasPValues = pValues && pValues.length > 0;

    let html = '';
    names.forEach((name, i) => {
        const signClass = coeffs[i] >= 0 ? 'positive' : 'negative';
        html += `<tr>
            <td>${escapeHtml(name)}</td>
            <td class="${signClass}">${coeffs[i].toFixed(4)}</td>`;
        if (hasStdErr) {
            html += `<td>${stdErrs[i].toFixed(4)}</td>`;
        } else {
            html += `<td>-</td>`;
        }
        if (hasTStats) {
            html += `<td>${tStats[i].toFixed(4)}</td>`;
        } else {
            html += `<td>-</td>`;
        }
        if (hasPValues) {
            html += `<td>${formatPValue(pValues[i])}</td>`;
        } else {
            html += `<td>-</td>`;
        }
        if (hasCi) {
            html += `<td>${ci[i][0].toFixed(4)}</td>
            <td>${ci[i][1].toFixed(4)}</td>`;
        }
        html += `</tr>`;
    });

    tbody.innerHTML = html;

    const thead = table.querySelector('thead tr');
    if (thead) {
        let headerHtml = '<th>Variable</th><th>Coefficient</th>';
        if (hasStdErr) headerHtml += '<th>Std Error</th>';
        if (hasTStats) headerHtml += '<th>t-stat</th>';
        if (hasPValues) headerHtml += '<th>p-value</th>';
        if (hasCi) {
            headerHtml += '<th>95% CI Lower</th><th>95% CI Upper</th>';
        }
        thead.innerHTML = headerHtml;
    }
}

/**
 * Update residuals table
 */
function updateResidualsTable(results) {
    const tbody = document.getElementById('residualsBody');
    const thead = document.querySelector('#residualsTable thead tr');
    if (!tbody) return;

    const { residuals, predictions, standardizedResiduals, leverage } = results;
    
    // Determine Y data (Actuals)
    let yData;
    if (STATE.rawData && STATE.rawData.length > 0 && STATE.yVariable && STATE.rawData[0][STATE.yVariable] !== undefined) {
        yData = STATE.rawData.map(row => row[STATE.yVariable]);
    } else {
        // Reconstruct actuals from model (Actual = Fitted + Residual)
        yData = predictions.map((pred, i) => pred + residuals[i]);
    }

    // Store residuals data for sorting and detail panel
    STATE.residualsData = yData.map((actual, i) => ({
        index: i,
        index1Based: i + 1,
        actual: actual,
        predicted: predictions[i],
        residual: residuals[i],
        stdResidual: standardizedResiduals[i],
        leverage: leverage?.[i] || null
    }));

    // Add click handlers to sortable headers
    const sortHeaders = thead.querySelectorAll('th[data-sort]');
    sortHeaders.forEach(th => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => sortResidualsTable(th.getAttribute('data-sort')));
    });

    renderResidualsTable();

    // Setup observation click handlers
    setupObservationClickHandlers();
}

/**
 * Render residuals table from STATE.residualsData
 */
function renderResidualsTable(data = STATE.residualsData) {
    const tbody = document.getElementById('residualsBody');
    if (!tbody || !data) return;

    const results = STATE.regressionResults;
    const leverageThreshold = results ? 2 * (results.k + 1) / results.n : 0;

    let html = '';
    data.forEach((row) => {
        const stdClass = Math.abs(row.stdResidual) > 2 ? 'outlier' : '';
        const highLeverage = row.leverage && row.leverage > leverageThreshold ? 'high-leverage' : '';
        const isSelected = STATE.selectedObservation === row.index;
        const clickClass = 'obs-clickable';

        html += `<tr class="${clickClass} ${stdClass} ${highLeverage} ${isSelected ? 'obs-selected' : ''}" data-obs-index="${row.index}">
            <td>${row.index1Based}</td>
            <td>${row.actual.toFixed(4)}</td>
            <td>${row.predicted.toFixed(4)}</td>
            <td class="${stdClass}">${row.residual.toFixed(4)}</td>
            <td class="${stdClass}">${row.stdResidual.toFixed(4)}</td>
            <td class="${highLeverage}">${row.leverage !== null ? row.leverage.toFixed(4) : 'N/A'}</td>
        </tr>`;
    });

    tbody.innerHTML = html;

    // Re-setup click handlers after rendering
    setupObservationClickHandlers();
}

/**
 * Setup click handlers for observation rows
 */
function setupObservationClickHandlers() {
    const rows = document.querySelectorAll('#residualsBody .obs-clickable');
    rows.forEach(row => {
        row.addEventListener('click', () => {
            const obsIndex = parseInt(row.getAttribute('data-obs-index'));
            showObservationDetail(obsIndex);
        });
    });
}

/**
 * Sort residuals table by column
 */
function sortResidualsTable(column) {
    if (!STATE.residualsData) return;

    // Toggle sort direction
    if (STATE.residualsSortColumn === column) {
        STATE.residualsSortDirection = STATE.residualsSortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        STATE.residualsSortColumn = column;
        STATE.residualsSortDirection = 'desc'; // default to desc for residuals (largest first)
    }

    // Update header indicators
    document.querySelectorAll('#residualsTable th').forEach(th => {
        const sortCol = th.getAttribute('data-sort');
        if (sortCol === column) {
            const arrow = STATE.residualsSortDirection === 'asc' ? ' ↑' : ' ↓';
            th.textContent = th.textContent.replace(' ↕', '').replace(' ↑', '').replace(' ↓', '') + arrow;
        } else {
            th.textContent = th.textContent.replace(' ↑', ' ↕').replace(' ↓', ' ↕');
        }
    });

    // Sort data
    STATE.residualsData.sort((a, b) => {
        let valA, valB;

        switch (column) {
            case 'index':
                valA = a.index;
                valB = b.index;
                break;
            case 'actual':
                valA = a.actual;
                valB = b.actual;
                break;
            case 'predicted':
                valA = a.predicted;
                valB = b.predicted;
                break;
            case 'residual':
                valA = Math.abs(a.residual); // sort by absolute residual
                valB = Math.abs(b.residual);
                break;
            case 'stdResidual':
                valA = Math.abs(a.stdResidual);
                valB = Math.abs(b.stdResidual);
                break;
            case 'leverage':
                valA = a.leverage || 0;
                valB = b.leverage || 0;
                break;
            default:
                return 0;
        }

        if (STATE.residualsSortDirection === 'asc') {
            return valA - valB;
        } else {
            return valB - valA;
        }
    });

    renderResidualsTable();
}

/**
 * Show observation detail modal
 * @param {number} obsIndex - Zero-based observation index
 */
function showObservationDetail(obsIndex) {
    const results = STATE.regressionResults;
    if (!results || !STATE.diagnostics) {
        showToast('Run regression and diagnostic tests first', 'warning');
        return;
    }

    const obsData = STATE.residualsData?.[obsIndex];
    if (!obsData) return;

    // Update selected observation
    STATE.selectedObservation = obsIndex;
    renderResidualsTable(); // Re-render to show selection

    // Build detail content
    const modal = document.getElementById('obsDetailModal');
    const content = document.getElementById('obsDetailContent');

    if (!modal || !content) return;

    // Gather raw variable values
    const rawValues = STATE.xVariables.map(v => STATE.rawData[obsIndex][v]);
    const yValue = STATE.rawData[obsIndex][STATE.yVariable];

    // Get diagnostic values
    const cooksDistance = STATE.diagnostics.influence?.find(d => d.shortName === "Cook's Distance");
    const cooksValue = cooksDistance?.distances?.[obsIndex];
    const dfbetasResult = STATE.diagnostics.influence?.find(d => d.shortName === 'DFBETAS');
    const dffitsResult = STATE.diagnostics.influence?.find(d => d.shortName === 'DFFITS');

    let html = `
        <div class="obs-detail-grid">
            <div class="obs-detail-section">
                <h4>Basic Info</h4>
                <div class="obs-detail-row">
                    <span class="label">Observation:</span>
                    <span class="value">#${obsData.index1Based}</span>
                </div>
                <div class="obs-detail-row">
                    <span class="label">Y Value (${escapeHtml(STATE.yVariable)}):</span>
                    <span class="value">${yValue.toFixed(4)}</span>
                </div>
            </div>

            <div class="obs-detail-section">
                <h4>Regression Values</h4>
                <div class="obs-detail-row">
                    <span class="label">Predicted:</span>
                    <span class="value">${obsData.predicted.toFixed(4)}</span>
                </div>
                <div class="obs-detail-row">
                    <span class="label">Residual:</span>
                    <span class="value ${Math.abs(obsData.residual) > 2 ? 'text-error' : ''}">${obsData.residual.toFixed(4)}</span>
                </div>
                <div class="obs-detail-row">
                    <span class="label">Std. Residual:</span>
                    <span class="value ${Math.abs(obsData.stdResidual) > 2 ? 'text-error' : ''}">${obsData.stdResidual.toFixed(4)}</span>
                </div>
                <div class="obs-detail-row">
                    <span class="label">Leverage:</span>
                    <span class="value ${obsData.leverage > 2 * (results.k + 1) / results.n ? 'text-warning' : ''}">${obsData.leverage?.toFixed(4) || 'N/A'}</span>
                </div>
            </div>
    `;

    // Predictor values
    if (rawValues.length > 0) {
        html += `<div class="obs-detail-section">
            <h4>Predictor Values</h4>`;
        rawValues.forEach((val, i) => {
            html += `<div class="obs-detail-row">
                <span class="label">${escapeHtml(STATE.xVariables[i])}:</span>
                <span class="value">${val?.toFixed(4) || 'N/A'}</span>
            </div>`;
        });
        html += `</div>`;
    }

    // Influence diagnostics
    html += `<div class="obs-detail-section">
        <h4>Influence Diagnostics</h4>`;

    if (cooksValue !== undefined) {
        const cookThreshold = cooksDistance?.threshold_4_over_n || 0;
        const isHigh = cooksValue > cookThreshold;
        html += `<div class="obs-detail-row">
            <span class="label">Cook's D:</span>
            <span class="value ${isHigh ? 'text-error' : ''}">${cooksValue.toFixed(4)} ${isHigh ? '(high)' : ''}</span>
        </div>`;
    }

    if (dffitsResult?.dffits) {
        const dffitsValue = dffitsResult.dffits[obsIndex];
        const dffitsThreshold = dffitsResult.threshold || 0;
        const isHigh = Math.abs(dffitsValue) > dffitsThreshold;
        html += `<div class="obs-detail-row">
            <span class="label">DFFITS:</span>
            <span class="value ${isHigh ? 'text-error' : ''}">${dffitsValue.toFixed(4)} ${isHigh ? '(high)' : ''}</span>
        </div>`;
    }

    if (dfbetasResult?.dfbetas) {
        html += `<div class="obs-detail-row">
            <span class="label">DFBETAS:</span>
            <div class="dfbetas-list">`;
        dfbetasResult.dfbetas[obsIndex].forEach((val, i) => {
            const coefName = i === 0 ? 'Intercept' : results.variableNames?.[i] || `X${i}`;
            const threshold = dfbetasResult.threshold || 0;
            const isHigh = Math.abs(val) > threshold;
            html += `<span class="dfbetas-item ${isHigh ? 'text-error' : ''}">
                ${coefName}: ${val.toFixed(4)}${isHigh ? ' ⚠️' : ''}
            </span>`;
        });
        html += `</div></div>`;
    }

    html += `</div>`;

    content.innerHTML = html;
    modal.classList.add('active');
}

/**
 * Close observation detail modal
 */
export function closeObsDetailModal() {
    const modal = document.getElementById('obsDetailModal');
    if (modal) modal.classList.remove('active');
    STATE.selectedObservation = null;
    renderResidualsTable();
}

// ============================================================================
// DIAGNOSTICS UI
// ============================================================================

/**
 * Update diagnostics display
 * @param {Object} diagnostics - Diagnostic test results
 */
export function updateDiagnosticsDisplay(diagnostics) {
    const container = document.getElementById('diagnosticsResults');
    if (!container) {
        console.warn('Diagnostics container not found');
        return;
    }

    const hasTests = Object.values(diagnostics).some(cat => cat && cat.length > 0);

    if (!hasTests) {
        container.innerHTML = `
            <div class="diagnostics-empty">
                <p>No diagnostic tests available</p>
            </div>
        `;
        return;
    }

    let html = '';

    // Linearity Tests
    if (diagnostics.linearity?.length > 0) {
        html += createDiagnosticsSection('Linearity Tests', 'Tests for linear relationship between variables', diagnostics.linearity);
    }

    // Heteroscedasticity Tests
    if (diagnostics.heteroscedasticity?.length > 0) {
        html += createDiagnosticsSection('Heteroscedasticity Tests', 'Tests for constant variance of residuals', diagnostics.heteroscedasticity);
    }

    // Normality Tests
    if (diagnostics.normality?.length > 0) {
        html += createDiagnosticsSection('Normality Tests', 'Tests for normal distribution of residuals', diagnostics.normality);
    }

    // Autocorrelation Tests
    if (diagnostics.autocorrelation?.length > 0) {
        html += createDiagnosticsSection('Autocorrelation Tests', 'Tests for independence of residuals', diagnostics.autocorrelation);
    }

    // Multicollinearity Tests
    if (diagnostics.multicollinearity?.length > 0) {
        html += createDiagnosticsSection('Multicollinearity Tests', 'Tests for correlation among predictor variables', diagnostics.multicollinearity);
    }

    // Influence Tests
    if (diagnostics.influence?.length > 0) {
        html += createDiagnosticsSection('Influence Tests', 'Tests for influential observations', diagnostics.influence);
    }

    container.innerHTML = html || '<div class="diagnostics-empty"><p>No diagnostic tests available</p></div>';
}

/**
 * Create diagnostics section HTML
 */
function createDiagnosticsSection(title, description, tests) {
    let html = `
        <div class="diagnostics-section">
            <h4 class="diagnostics-section-title">${title}</h4>
            <p class="diagnostics-section-desc">${description}</p>
    `;

    tests.forEach(test => {
        const statusClass = test.is_passed ? 'pass' : 'fail';
        const statusText = test.is_passed ? 'Pass' : 'Fail';
        const statusIcon = test.is_passed ? '✓' : '✗';

        // Special handling for VIF test
        if (test.shortName === 'VIF') {
            html += `
                <div class="diagnostic-test-card ${statusClass}">
                    <div class="test-header">
                        <span class="test-name">${test.name}</span>
                        <span class="test-status ${statusClass}">${statusIcon} ${statusText}</span>
                    </div>
                    <div class="test-details">
                        <span class="test-stat">Max VIF = ${test.max_vif === Infinity ? '∞' : test.max_vif.toFixed(4)}</span>
                    </div>
                    <div class="test-interpretation">${test.interpretation}</div>
                    ${test.vif_results ? `
                        <table class="vif-table" style="margin-top: 12px;">
                            <thead>
                                <tr>
                                    <th>Variable</th>
                                    <th>VIF</th>
                                    <th>R²</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${test.vif_results.map(vif => {
                                    const vifValue = vif.vif === Infinity ? '∞' : vif.vif.toFixed(2);
                                    const colorClass = vif.vif > 10 ? 'error' : (vif.vif > 5 ? 'warning' : 'success');
                                    return `
                                        <tr>
                                            <td>${vif.variable}</td>
                                            <td class="${colorClass}">${vifValue}</td>
                                            <td>${vif.rsquared.toFixed(4)}</td>
                                            <td class="${colorClass}">${vif.interpretation}</td>
                                        </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    ` : ''}
                    ${!test.is_passed && test.guidance ? `<div class="test-guidance"><strong>Guidance:</strong> ${test.guidance}</div>` : ''}
                </div>
            `;
        } else {
            html += `
                <div class="diagnostic-test-card ${statusClass}">
                    <div class="test-header">
                        <span class="test-name">${test.name}</span>
                        <span class="test-status ${statusClass}">${statusIcon} ${statusText}</span>
                    </div>
                    <div class="test-details">
                        ${test.statistic !== undefined ? `<span class="test-stat">${getTestStatLabel(test)} = ${test.statistic.toFixed(4)}</span>` : ''}
                        ${test.p_value !== undefined ? `<span class="test-p-value">p = ${test.p_value < 0.0001 ? '< 0.0001' : test.p_value.toFixed(4)}</span>` : ''}
                    </div>
                    <div class="test-interpretation">${test.interpretation}</div>
                    ${!test.is_passed && test.guidance ? `<div class="test-guidance"><strong>Guidance:</strong> ${test.guidance}</div>` : ''}
                </div>
            `;
        }
    });

    html += '</div>';
    return html;
}

/**
 * Get test statistic label
 */
function getTestStatLabel(test) {
    if (test.shortName === "Cook's Distance") return "Max D";
    if (test.shortName === 'Durbin-Watson') return "DW";
    if (test.shortName === 'Shapiro-Wilk') return "W";
    if (test.shortName === 'Jarque-Bera') return "JB";
    if (test.shortName === 'Anderson-Darling') return "A²";
    if (test.shortName === 'DFBETAS') return "Max |DFBETAS|";
    if (test.shortName === 'DFFITS') return "Max |DFFITS|";
    return 'F';
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

/**
 * Run regression based on selected method
 */
export async function runRegression() {
    const method = document.getElementById('regressionMethodSelect')?.value || 'ols';

    try {
        let results;
        switch (method) {
            case 'ridge':
                const lambda = parseFloat(document.getElementById('lambdaSlider')?.value) || 1.0;
                const standardize = document.getElementById('standardizeCheck')?.checked ?? true;
                results = await calculateRidgeRegression(STATE.yVariable, STATE.xVariables, lambda, standardize);
                break;
            case 'lasso':
                const lassoLambda = parseFloat(document.getElementById('lambdaSlider')?.value) || 1.0;
                const lassoStandardize = document.getElementById('standardizeCheck')?.checked ?? true;
                const maxIter = parseInt(document.getElementById('maxIterSlider')?.value) || 1000;
                const tolSelect = document.getElementById('tolSelect')?.value || '1e-7';
                const tol = parseFloat(tolSelect);
                results = await calculateLassoRegression(STATE.yVariable, STATE.xVariables, lassoLambda, lassoStandardize, maxIter, tol);
                break;
            case 'elastic_net':
                const enLambda = parseFloat(document.getElementById('enLambdaSlider')?.value) || 1.0;
                const enAlpha = parseFloat(document.getElementById('enAlphaSlider')?.value) || 0.5;
                const enStandardize = document.getElementById('enStandardizeCheck')?.checked ?? true;
                const enMaxIter = parseInt(document.getElementById('enMaxIterSlider')?.value) || 1000;
                const enTolSelect = document.getElementById('enTolSelect')?.value || '1e-7';
                const enTol = parseFloat(enTolSelect);
                results = await calculateElasticNetRegression(STATE.yVariable, STATE.xVariables, enLambda, enAlpha, enStandardize, enMaxIter, enTol);
                break;
            case 'loess':
                const loessSpan = parseFloat(document.getElementById('loessSpanSlider')?.value) || 0.75;
                const loessDegree = parseInt(document.getElementById('loessDegreeSelect')?.value) || 1;
                const loessSurface = document.getElementById('loessSurfaceSelect')?.value || 'direct';
                const loessRobust = document.getElementById('loessRobustCheck')?.checked ? 2 : 0;
                results = await calculateLoessRegression(STATE.yVariable, STATE.xVariables, loessSpan, loessDegree, loessRobust, loessSurface);
                break;
            case 'wls':
                const wlsWeights = document.getElementById('wlsWeightsSelect')?.value;
                if (!wlsWeights) {
                    throw new Error('Please select a weights variable for WLS regression.');
                }
                results = await calculateWlsRegression(STATE.yVariable, STATE.xVariables, wlsWeights);
                break;
            case 'polynomial':
                // Polynomial regression requires exactly one predictor
                if (STATE.xVariables.length !== 1) {
                    throw new Error('Polynomial regression requires exactly one predictor. Please select only one X variable.');
                }
                const polyDegree = parseInt(document.getElementById('polyDegreeSlider')?.value) || 2;
                const polyCenter = document.getElementById('polyCenterCheck')?.checked ?? true;
                const polyStandardize = document.getElementById('polyStandardizeCheck')?.checked ?? false;
                results = await calculatePolynomialRegression(STATE.yVariable, STATE.xVariables[0], polyDegree, polyCenter, polyStandardize);
                break;
            case 'polynomial_ridge':
                if (STATE.xVariables.length !== 1) {
                    throw new Error('Polynomial regression requires exactly one predictor. Please select only one X variable.');
                }
                const polyRDegree = parseInt(document.getElementById('polyRegDegreeSlider')?.value) || 2;
                const polyRLambda = parseFloat(document.getElementById('polyLambdaSlider')?.value) || 0.1;
                const polyRCenter = document.getElementById('polyRegCenterCheck')?.checked ?? true;
                const polyRStandardize = document.getElementById('polyRegStandardizeCheck')?.checked ?? true;
                results = await calculatePolynomialRidge(STATE.yVariable, STATE.xVariables[0], polyRDegree, polyRLambda, polyRCenter, polyRStandardize);
                break;
            case 'polynomial_lasso':
                if (STATE.xVariables.length !== 1) {
                    throw new Error('Polynomial regression requires exactly one predictor. Please select only one X variable.');
                }
                const polyLDegree = parseInt(document.getElementById('polyRegDegreeSlider')?.value) || 2;
                const polyLLambda = parseFloat(document.getElementById('polyLambdaSlider')?.value) || 0.1;
                const polyLCenter = document.getElementById('polyRegCenterCheck')?.checked ?? true;
                const polyLStandardize = document.getElementById('polyRegStandardizeCheck')?.checked ?? true;
                results = await calculatePolynomialLasso(STATE.yVariable, STATE.xVariables[0], polyLDegree, polyLLambda, polyLCenter, polyLStandardize);
                break;
            case 'polynomial_enet':
                if (STATE.xVariables.length !== 1) {
                    throw new Error('Polynomial regression requires exactly one predictor. Please select only one X variable.');
                }
                const polyEDegree = parseInt(document.getElementById('polyRegDegreeSlider')?.value) || 2;
                const polyELambda = parseFloat(document.getElementById('polyLambdaSlider')?.value) || 0.1;
                const polyEAlpha = parseFloat(document.getElementById('polyAlphaSlider')?.value) || 0.5;
                const polyECenter = document.getElementById('polyRegCenterCheck')?.checked ?? true;
                const polyEStandardize = document.getElementById('polyRegStandardizeCheck')?.checked ?? true;
                results = await calculatePolynomialElasticNet(STATE.yVariable, STATE.xVariables[0], polyEDegree, polyELambda, polyEAlpha, polyECenter, polyEStandardize);
                break;
            default:
                results = await calculateRegression(STATE.yVariable, STATE.xVariables);
        }

        updateState({ regressionResults: results });
        updateResultsDisplay(results);
        showToast(`Regression complete (${formatMethodName(results.method)})`, 'success');

        // Hide path chart when running a single regression
        const pathChartContainer = document.getElementById('pathChartContainer');
        if (pathChartContainer) pathChartContainer.style.display = 'none';

        // Run cross-validation if enabled (checkbox is disabled for unsupported methods)
        const cvEnabled = document.getElementById('cvEnableCheck')?.checked;
        if (cvEnabled) {
            try {
                const nFolds = parseInt(document.getElementById('cvFoldsSelect')?.value) || 5;
                const shuffle = document.getElementById('cvShuffleCheck')?.checked ?? true;
                const seedVal = document.getElementById('cvSeedInput')?.value;
                const seed = seedVal ? parseInt(seedVal) : null;

                const cvResult = runCrossValidation(method, nFolds, shuffle, seed);
                updateState({ cvResults: cvResult });
                updateCVDisplay(cvResult);
                showToast(`Cross-validation complete (${nFolds}-fold)`, 'success');
            } catch (cvError) {
                console.error('Cross-validation error:', cvError);
                showToast(`CV error: ${cvError.message}`, 'error');
            }
        } else {
            updateState({ cvResults: null });
            resetCVDisplay();
        }

        // Auto-trace path for regularized methods to keep chart in sync
        if (['ridge', 'lasso', 'elastic_net'].includes(method)) {
            runTracePath();
        }

    } catch (error) {
        console.error('Regression error:', error);
        showToast(error.message, 'error');
    }
}

/**
 * Run regularization path tracing
 */
export async function runTracePath() {
    const method = document.getElementById('regressionMethodSelect')?.value || 'ols';
    
    if (!['ridge', 'lasso', 'elastic_net'].includes(method)) {
        showToast('Trace Path is only available for regularized regression methods.', 'warning');
        return;
    }

    try {
        const n = STATE.rawData.length;
        const k = STATE.xVariables.length;

        if (n <= k + 1) {
            throw new Error(`Need at least ${k + 2} data points for ${k} predictor(s). You have ${n}.`);
        }

        if (!WasmRegression.isReady()) {
            throw new Error('WASM module is not ready yet. Please wait a moment and try again.');
        }

        // Determine parameters based on method
        let alpha = 1.0; // Default for Lasso
        let standardize = true;
        let maxIter = 1000;
        let tol = 1e-7;

        if (method === 'ridge') {
            alpha = 0.0;
            standardize = document.getElementById('standardizeCheck')?.checked ?? true;
        } else if (method === 'lasso') {
            alpha = 1.0;
            standardize = document.getElementById('standardizeCheck')?.checked ?? true;
            maxIter = parseInt(document.getElementById('maxIterSlider')?.value) || 1000;
            tol = parseFloat(document.getElementById('tolSelect')?.value || '1e-7');
        } else if (method === 'elastic_net') {
            alpha = parseFloat(document.getElementById('enAlphaSlider')?.value) || 0.5;
            standardize = document.getElementById('enStandardizeCheck')?.checked ?? true;
            maxIter = parseInt(document.getElementById('enMaxIterSlider')?.value) || 1000;
            tol = parseFloat(document.getElementById('enTolSelect')?.value || '1e-7');
        }

        const yData = STATE.rawData.map(row => row[STATE.yVariable]);
        const xData = STATE.xVariables.map(v => STATE.rawData.map(row => row[v]));

        // Call WASM trace path
        const resultJson = WasmRegression.tracePath(
            yData, xData, 100, 0.0001, alpha, standardize, maxIter, tol
        );
        
        const result = JSON.parse(resultJson);
        if (result.error) throw new Error(result.error);

                updateState({ pathResults: result });
        
                // Show results section if it was hidden
                const resultsSection = document.getElementById('resultsSection');
                if (resultsSection) resultsSection.style.display = 'block';
        
                // Show path chart container and update it
                const pathChartContainer = document.getElementById('pathChartContainer');
                if (pathChartContainer) pathChartContainer.style.display = 'block';
        
                // Scroll to the chart
                pathChartContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
                // Update the path chart
                const { updatePathChart } = await import('./charts.js');
                updatePathChart(result);
        
                showToast(`Coefficient path traced (100 steps, \u03B1=${alpha.toFixed(2)})`, 'success');
    } catch (error) {
        console.error('Trace Path error:', error);
        showToast(error.message, 'error');
    }
}

// ============================================================================
// CROSS-VALIDATION
// ============================================================================

/**
 * Run cross-validation for the current regression method
 * @param {string} method - Regression method
 * @param {number} nFolds - Number of folds
 * @param {boolean} shuffle - Whether to shuffle data
 * @param {number|null} seed - Random seed or null
 * @returns {Object} Parsed CV results
 */
function runCrossValidation(method, nFolds, shuffle, seed) {
    const yData = STATE.rawData.map(row => row[STATE.yVariable]);
    const xData = STATE.xVariables.map(v => STATE.rawData.map(row => row[v]));
    const names = ['Intercept', ...STATE.xVariables];

    let resultJson;
    switch (method) {
        case 'ridge': {
            const lambda = parseFloat(document.getElementById('lambdaSlider')?.value) || 1.0;
            const standardize = document.getElementById('standardizeCheck')?.checked ?? true;
            resultJson = WasmRegression.kfoldCvRidge(yData, xData, lambda, standardize, nFolds, shuffle, seed);
            break;
        }
        case 'lasso': {
            const lambda = parseFloat(document.getElementById('lambdaSlider')?.value) || 1.0;
            const standardize = document.getElementById('standardizeCheck')?.checked ?? true;
            resultJson = WasmRegression.kfoldCvLasso(yData, xData, lambda, standardize, nFolds, shuffle, seed);
            break;
        }
        case 'elastic_net': {
            const lambda = parseFloat(document.getElementById('enLambdaSlider')?.value) || 1.0;
            const alpha = parseFloat(document.getElementById('enAlphaSlider')?.value) || 0.5;
            const standardize = document.getElementById('enStandardizeCheck')?.checked ?? true;
            resultJson = WasmRegression.kfoldCvElasticNet(yData, xData, lambda, alpha, standardize, nFolds, shuffle, seed);
            break;
        }
        default:
            resultJson = WasmRegression.kfoldCvOls(yData, xData, names, nFolds, shuffle, seed);
    }

    const result = JSON.parse(resultJson);
    if (result.error) {
        throw new Error(result.error);
    }

    console.log(`[linreg-core] CV complete: ${nFolds}-fold, mean R²=${result.mean_r_squared?.toFixed(4)}, mean RMSE=${result.mean_rmse?.toFixed(4)}`);
    return result;
}

/**
 * Reset CV display to placeholder state
 */
function resetCVDisplay() {
    const container = document.getElementById('cvResultsContainer');
    if (container) {
        container.innerHTML = `
            <div class="diagnostics-empty">
                <p style="color: var(--text-muted); font-size: 0.875rem;">Enable cross-validation and run regression to see results</p>
            </div>
        `;
    }

    // Collapse the section
    const toggle = document.getElementById('cvResultsToggle');
    const content = document.getElementById('cvResultsContent');
    if (toggle && !toggle.classList.contains('collapsed')) {
        toggle.classList.add('collapsed');
        toggle.setAttribute('aria-expanded', 'false');
    }
    if (content && !content.classList.contains('collapsed')) {
        content.classList.add('collapsed');
    }

    // Destroy CV chart
    if (STATE.charts.cv && typeof STATE.charts.cv.destroy === 'function') {
        STATE.charts.cv.destroy();
        STATE.charts.cv = null;
    }
}

/**
 * Render cross-validation results
 * @param {Object} cv - CV result object from WASM
 */
function updateCVDisplay(cv) {
    const container = document.getElementById('cvResultsContainer');
    if (!container) return;

    const folds = cv.fold_results || [];
    const nFolds = cv.n_folds || folds.length;

    // Format helper
    const fmt = (v, d = 4) => (v != null && !isNaN(v)) ? v.toFixed(d) : 'N/A';
    const fmtPm = (mean, std, d = 4) => `${fmt(mean, d)} &plusmn; ${fmt(std, d)}`;

    // Build summary stats cards
    let html = `
        <div class="stats-grid" style="margin-bottom: 16px;">
            <div class="stat-card">
                <div class="value" style="font-size: 1rem;">${fmtPm(cv.mean_rmse, cv.std_rmse)}</div>
                <div class="label">Mean RMSE</div>
            </div>
            <div class="stat-card">
                <div class="value" style="font-size: 1rem;">${fmtPm(cv.mean_r_squared, cv.std_r_squared)}</div>
                <div class="label">Mean R²</div>
            </div>
            <div class="stat-card">
                <div class="value" style="font-size: 1rem;">${fmtPm(cv.mean_mae, cv.std_mae)}</div>
                <div class="label">Mean MAE</div>
            </div>
            <div class="stat-card">
                <div class="value" style="font-size: 1rem;">${fmt(cv.mean_mse)}</div>
                <div class="label">Mean MSE</div>
            </div>
        </div>
    `;

    // Fold results table
    html += `
        <div style="margin-bottom: 16px;">
            <div style="font-weight: 600; font-size: 0.8125rem; margin-bottom: 8px;">Fold Results</div>
            <div class="table-container" style="max-height: 250px;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Fold</th>
                            <th>Train Size</th>
                            <th>Test Size</th>
                            <th>Test RMSE</th>
                            <th>Test R²</th>
                            <th>Train R²</th>
                            <th>Overfit Gap</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    folds.forEach((fold, i) => {
        const gap = (fold.train_r_squared != null && fold.r_squared != null)
            ? (fold.train_r_squared - fold.r_squared)
            : null;
        const gapClass = (gap != null && gap > 0.1) ? 'color: var(--accent-danger);' : '';

        html += `
            <tr>
                <td>${fold.fold_index != null ? fold.fold_index : i + 1}</td>
                <td>${fold.train_size}</td>
                <td>${fold.test_size}</td>
                <td>${fmt(fold.rmse)}</td>
                <td>${fmt(fold.r_squared)}</td>
                <td>${fmt(fold.train_r_squared)}</td>
                <td style="${gapClass}">${gap != null ? fmt(gap) : 'N/A'}</td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;

    // Chart canvas
    html += `
        <div style="margin-bottom: 8px;">
            <div style="font-weight: 600; font-size: 0.8125rem; margin-bottom: 8px;">Test RMSE by Fold</div>
            <div style="position: relative; height: 200px;">
                <canvas id="cvChart"></canvas>
            </div>
        </div>
    `;

    container.innerHTML = html;

    // Auto-expand the section
    const toggle = document.getElementById('cvResultsToggle');
    const content = document.getElementById('cvResultsContent');
    if (toggle && toggle.classList.contains('collapsed')) {
        toggle.classList.remove('collapsed');
        toggle.setAttribute('aria-expanded', 'true');
    }
    if (content && content.classList.contains('collapsed')) {
        content.classList.remove('collapsed');
    }

    // Create chart
    createCVChart(folds, cv.mean_rmse);
}

/**
 * Create the CV RMSE bar chart
 * @param {Array} folds - Array of fold results
 * @param {number} meanRmse - Mean RMSE across folds
 */
function createCVChart(folds, meanRmse) {
    // Destroy previous chart
    if (STATE.charts.cv && typeof STATE.charts.cv.destroy === 'function') {
        STATE.charts.cv.destroy();
    }

    const canvas = document.getElementById('cvChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const theme = getCurrentTheme();
    const isDark = theme === 'dark';

    const labels = folds.map((f, i) => `Fold ${f.fold_index != null ? f.fold_index : i + 1}`);
    const rmseValues = folds.map(f => f.rmse);

    STATE.charts.cv = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Test RMSE',
                    data: rmseValues,
                    backgroundColor: isDark ? 'rgba(96, 165, 250, 0.6)' : 'rgba(59, 130, 246, 0.6)',
                    borderColor: isDark ? '#60a5fa' : '#3b82f6',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                annotation: undefined
            },
            scales: {
                x: {
                    ticks: { color: isDark ? '#e5e7eb' : '#374151', font: { size: 11 } },
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: isDark ? '#e5e7eb' : '#374151', font: { size: 11 } },
                    grid: { color: isDark ? '#374151' : '#e5e7eb' },
                    title: {
                        display: true,
                        text: 'RMSE',
                        color: isDark ? '#e5e7eb' : '#374151',
                        font: { size: 12 }
                    }
                }
            }
        },
        plugins: [{
            id: 'meanLine',
            afterDraw(chart) {
                if (meanRmse == null || isNaN(meanRmse)) return;
                const yAxis = chart.scales.y;
                const yPixel = yAxis.getPixelForValue(meanRmse);
                const ctx = chart.ctx;
                ctx.save();
                ctx.strokeStyle = isDark ? '#f87171' : '#ef4444';
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 3]);
                ctx.beginPath();
                ctx.moveTo(chart.chartArea.left, yPixel);
                ctx.lineTo(chart.chartArea.right, yPixel);
                ctx.stroke();

                // Label
                ctx.fillStyle = isDark ? '#f87171' : '#ef4444';
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(`Mean: ${meanRmse.toFixed(4)}`, chart.chartArea.right - 4, yPixel - 6);
                ctx.restore();
            }
        }]
    });
}

/**
 * Run diagnostic tests
 */
export async function runDiagnosticTests() {
    await runDiagnosticTestByType('all');
}

/**
 * Run a specific category of diagnostic tests
 * @param {string} testType - Test category: 'linearity', 'heteroscedasticity', 'normality', 'autocorrelation', 'influence', or 'all'
 */
export async function runDiagnosticTestByType(testType) {
    if (!STATE.regressionResults) {
        showToast('Please run regression first', 'warning');
        return;
    }

    const yData = STATE.rawData.map(row => row[STATE.yVariable]);
    const xData = STATE.xVariables.map(v => STATE.rawData.map(row => row[v]));

    const rainbowMethod = document.getElementById('rainbowMethod')?.value || 'r';
    const whiteMethod = document.getElementById('whiteMethod')?.value || 'r';

    try {
        const diagnostics = await runDiagnostics(yData, xData, rainbowMethod, whiteMethod, testType);
        updateState({ diagnostics });
        updateDiagnosticsDisplay(diagnostics);

        const testTypeName = testType === 'all' ? 'all' : testType.replace('_', ' ');
        showToast(`Diagnostic tests complete (${testTypeName})`, 'success');
    } catch (error) {
        console.error('Diagnostics error:', error);
        showToast(error.message, 'error');
    }
}

/**
 * Export results as CSV
 */
export function exportResultsAsCSV() {
    if (!STATE.regressionResults) {
        showToast('No results to export', 'warning');
        return;
    }

    const results = STATE.regressionResults;
    let csv = 'Row,Actual,Predicted,Residual,Standardized_Residual\n';

    results.residuals.forEach((resid, i) => {
        csv += `${i + 1},${STATE.rawData[i][STATE.yVariable]},${results.predictions[i]},${resid},${results.standardizedResiduals[i]}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'regression_residuals.csv';
    link.click();

    showToast('Results exported as CSV', 'success');
}

/**
 * Export augmented data with all diagnostic metrics
 */
export function exportAugmentedData() {
    if (!STATE.regressionResults || !STATE.diagnostics) {
        showToast('Run regression and diagnostic tests first', 'warning');
        return;
    }

    const results = STATE.regressionResults;
    const diagnostics = STATE.diagnostics;

    // Build header row
    let header = ['Row', STATE.yVariable, 'Predicted', 'Residual', 'Std_Residual', 'Leverage'];

    // Add predictor values
    STATE.xVariables.forEach(v => header.push(v));

    // Add diagnostic metrics
    header.push('Cooks_D');

    // Get DFFITS values
    const dffitsResult = diagnostics.influence?.find(d => d.shortName === 'DFFITS');
    if (dffitsResult?.dffits) {
        header.push('DFFITS');
    }

    // Get DFBETAS values (one column per coefficient)
    const dfbetasResult = diagnostics.influence?.find(d => d.shortName === 'DFBETAS');
    const dfbetasNames = [];
    if (dfbetasResult?.dfbetas) {
        results.variableNames.forEach((name, i) => {
            dfbetasNames.push(name);
            header.push(`DFBETAS_${name}`);
        });
    }

    // Build CSV content
    let csv = header.join(',') + '\n';

    const n = results.n;
    for (let i = 0; i < n; i++) {
        let row = [
            i + 1,
            STATE.rawData[i][STATE.yVariable].toFixed(6),
            results.predictions[i].toFixed(6),
            results.residuals[i].toFixed(6),
            results.standardizedResiduals[i].toFixed(6),
            (results.leverage?.[i] || 0).toFixed(6)
        ];

        // Add predictor values
        STATE.xVariables.forEach(v => {
            row.push((STATE.rawData[i][v] || 0).toFixed(6));
        });

        // Add Cook's D
        const cooksResult = diagnostics.influence?.find(d => d.shortName === "Cook's Distance");
        const cooksValue = cooksResult?.distances?.[i] || 0;
        row.push(cooksValue.toFixed(6));

        // Add DFFITS
        if (dffitsResult?.dffits) {
            row.push(dffitsResult.dffits[i].toFixed(6));
        }

        // Add DFBETAS
        if (dfbetasResult?.dfbetas) {
            dfbetasResult.dfbetas[i].forEach(val => {
                row.push(val.toFixed(6));
            });
        }

        csv += row.join(',') + '\n';
    }

    // Add metadata section
    csv += '\n# Model Summary\n';
    csv += `# Method,${results.method}\n`;
    csv += `# R_Squared,${results.rSquared.toFixed(6)}\n`;
    csv += `# Adj_R_Squared,${results.adjRSquared.toFixed(6)}\n`;
    csv += `# RMSE,${results.rmse.toFixed(6)}\n`;
    csv += `# AIC,${results.aic?.toFixed(4) || 'N/A'}\n`;
    csv += `# BIC,${results.bic?.toFixed(4) || 'N/A'}\n`;
    csv += `# F_Statistic,${results.fStat?.toFixed(4) || 'N/A'}\n`;
    csv += `# F_P_Value,${results.fPValue?.toFixed(6) || 'N/A'}\n`;

    // Add coefficients
    csv += '\n# Coefficients\n';
    csv += '# Variable,Coefficient,Std_Error,t_Stat,p_Value\n';
    results.variableNames.forEach((name, i) => {
        csv += `#${name},${results.coefficients[i].toFixed(6)},${results.stdErrors?.[i]?.toFixed(6) || 'N/A'},${results.tStats?.[i]?.toFixed(6) || 'N/A'},${results.pValues?.[i]?.toFixed(6) || 'N/A'}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'regression_augmented_data.csv';
    link.click();

    showToast('Augmented data exported', 'success');
}

/**
 * Export trained model as JSON file
 */
export function exportModelAsJSON() {
    if (!STATE.regressionResults) {
        showToast('No regression results to export', 'warning');
        return;
    }

    const results = STATE.regressionResults;

    // Determine model type
    const modelTypeMap = {
        'ols': 'OLS',
        'ridge': 'Ridge',
        'lasso': 'Lasso',
        'elastic_net': 'ElasticNet',
        'wls': 'WLS',
        'loess': 'LOESS'
    };
    const modelType = modelTypeMap[results.method] || 'OLS';

    // Create a complete model object with all data needed for reconstruction
    const modelData = {
        coefficients: results.coefficients,
        stdErrors: results.stdErrors,
        tStats: results.tStats,
        pValues: results.pValues,
        confIntLower: results.confIntLower,
        confIntUpper: results.confIntUpper,
        rSquared: results.rSquared,
        adjRSquared: results.adjRSquared,
        fStat: results.fStat,
        fPValue: results.fPValue,
        residualStdError: results.stdError,
        dfResiduals: results.dfResiduals,
        dfModel: results.dfModel,
        fittedValues: results.predictions,
        residuals: results.residuals,
        mse: results.mse,
        rmse: results.rmse,
        mae: results.mae,
        nObservations: results.n,
        nPredictors: results.k,
        variableNames: results.variableNames,
        logLikelihood: results.logLikelihood,
        aic: results.aic,
        bic: results.bic,
        leverage: results.leverage,
        standardizedResiduals: results.standardizedResiduals,
        yVariable: STATE.yVariable,
        xVariables: STATE.xVariables,
        diagnostics: STATE.diagnostics, // Save diagnostic test results
        pathResults: STATE.pathResults // Save regularization path if traced
    };

    // Add method-specific fields
    if (results.method === 'ridge' || results.method === 'lasso' || results.method === 'elastic_net') {
        modelData.intercept = results.intercept;
        modelData.lambda = results.lambda;
    }
    if (results.method === 'elastic_net') {
        modelData.alpha = results.alpha;
    }
    if (results.method === 'loess') {
        modelData.fitted = results.fittedValues || results.predictions;
        modelData.span = results.span;
        modelData.degree = results.degree;
        modelData.robustIterations = results.robustIterations;
        modelData.surface = results.surface;
    }

    // Use WASM serialization function
    (async () => {
        try {
            const serialized = await WasmRegression.serializeModel(
                modelData,
                modelType,
                `Model from ${STATE.dataSourceName || 'Unknown'} (${new Date().toISOString().split('T')[0]})`
            );

            // Download the file
            const blob = new Blob([serialized], { type: 'application/json' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `linreg_${results.method}_model.json`;
            link.click();

            showToast('Model exported as JSON', 'success');
        } catch (e) {
            console.error('Error exporting model:', e);
            showToast('Failed to export model', 'error');
        }
    })();
}

/**
 * Import trained model from JSON file
 */
export function importModelFromJSON() {
    // Create a file input
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';

    input.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (event) => {
            try {
                const serialized = event.target.result;

                // Use WASM deserialization function to get metadata
                const metadata = await WasmRegression.getModelMetadata(serialized);

                // Show metadata and confirm
                const confirmed = confirm(
                    `Load model?\n\n` +
                    `Type: ${metadata.model_type}\n` +
                    `Library Version: ${metadata.library_version}\n` +
                    `Format Version: ${metadata.format_version}\n` +
                    `Created: ${metadata.created_at}\n` +
                    (metadata.name ? `Name: ${metadata.name}\n` : '')
                );

                if (confirmed) {
                    const modelData = await WasmRegression.deserializeModel(serialized);

                    // Store in STATE for later use
                    STATE.importedModel = {
                        ...modelData,
                        metadata: metadata
                    };

                    // Restore variable names if available
                    if (modelData.yVariable) updateState({ yVariable: modelData.yVariable });
                    if (modelData.xVariables) updateState({ xVariables: modelData.xVariables });

                    // Restore diagnostics if available
                    if (modelData.diagnostics) {
                        updateState({ diagnostics: modelData.diagnostics });
                        updateDiagnosticsDisplay(modelData.diagnostics);
                    } else {
                        // Clear diagnostics if none saved
                        updateState({ diagnostics: null });
                        const diagContainer = document.getElementById('diagnosticsResults');
                        if (diagContainer) {
                            diagContainer.innerHTML = '<div class="diagnostics-empty"><p style="color: var(--text-muted); font-size: 0.875rem;">No diagnostic results saved with this model</p></div>';
                        }
                    }

                    // Restore path results if available
                    if (modelData.pathResults) {
                        updateState({ pathResults: modelData.pathResults });
                        const pathChartContainer = document.getElementById('pathChartContainer');
                        if (pathChartContainer) pathChartContainer.style.display = 'block';
                    } else {
                        updateState({ pathResults: null });
                        const pathChartContainer = document.getElementById('pathChartContainer');
                        if (pathChartContainer) pathChartContainer.style.display = 'none';
                    }

                    // Update UI with imported model
                    if (modelData.coefficients) {
                        updateResultsDisplay({
                            method: metadata.model_type.toLowerCase(),
                            coefficients: modelData.coefficients,
                            stdErrors: modelData.stdErrors,
                            tStats: modelData.tStats,
                            pValues: modelData.pValues,
                            confIntLower: modelData.confIntLower,
                            confIntUpper: modelData.confIntUpper,
                            rSquared: modelData.rSquared,
                            adjRSquared: modelData.adjRSquared,
                            fStat: modelData.fStat,
                            fPValue: modelData.fPValue,
                            stdError: modelData.residualStdError,
                            rmse: modelData.rmse,
                            mae: modelData.mae,
                            mse: modelData.mse,
                            predictions: modelData.fittedValues,
                            residuals: modelData.residuals,
                            leverage: modelData.leverage,
                            standardizedResiduals: modelData.standardizedResiduals || modelData.residuals?.map((r, i) => {
                                // Recalculate if we have stdError
                                return modelData.residualStdError ?
                                    r / modelData.residualStdError : 0;
                            }),
                            n: modelData.nObservations,
                            k: modelData.nPredictors,
                            variableNames: modelData.variableNames || [],
                            logLikelihood: modelData.logLikelihood,
                            aic: modelData.aic,
                            bic: modelData.bic,
                            lambda: modelData.lambda,
                            alpha: modelData.alpha,
                            isImported: true
                        });

                        showToast(`Loaded ${metadata.model_type} model`, 'success');
                    }
                }
            } catch (e) {
                console.error('Error importing model:', e);
                showToast('Failed to import model. Invalid JSON format.', 'error');
            }
        };

        reader.readAsText(file);
    };

    input.click();
}

/**
 * Filter residuals table by influential observations
 */
export function filterInfluential() {
    if (!STATE.regressionResults || !STATE.diagnostics) {
        showToast('Run regression and diagnostic tests first', 'warning');
        return;
    }

    // Determine threshold for high leverage
    const results = STATE.regressionResults;
    const leverageThreshold = 2 * (results.k + 1) / results.n;

    // Get Cook's D threshold
    const cooksResult = STATE.diagnostics.influence?.find(d => d.shortName === "Cook's Distance");
    const cooksThreshold = cooksResult?.threshold_4_over_n || 4 / results.n;

    // Get DFFITS threshold
    const dffitsResult = STATE.diagnostics.influence?.find(d => d.shortName === 'DFFITS');
    const dffitsThreshold = dffitsResult?.threshold || 0;

    // Filter observations that exceed any threshold
    const filtered = STATE.residualsData.filter(obs => {
        // High leverage
        if (obs.leverage && obs.leverage > leverageThreshold) return true;
        // High Cook's D
        if (cooksResult?.distances && cooksResult.distances[obs.index] > cooksThreshold) return true;
        // High DFFITS
        if (dffitsResult?.dffits && Math.abs(dffitsResult.dffits[obs.index]) > dffitsThreshold) return true;
        return false;
    });

    if (filtered.length === 0) {
        showToast('No influential observations found', 'info');
        renderResidualsTable(STATE.residualsData);
        return;
    }

    renderResidualsTable(filtered);
    showToast(`Showing ${filtered.length} influential observation${filtered.length > 1 ? 's' : ''}`, 'info');

    // Update active filter button
    updateActiveFilterButton('influential');
}

/**
 * Filter residuals table by outliers
 */
export function filterOutliers() {
    if (!STATE.regressionResults) {
        showToast('Run regression first', 'warning');
        return;
    }

    // Filter by |std residual| > 2
    const filtered = STATE.residualsData.filter(obs => Math.abs(obs.stdResidual) > 2);

    if (filtered.length === 0) {
        showToast('No outliers found (|std residual| > 2)', 'info');
        renderResidualsTable(STATE.residualsData);
        return;
    }

    renderResidualsTable(filtered);
    showToast(`Showing ${filtered.length} outlier${filtered.length > 1 ? 's' : ''}`, 'info');

    // Update active filter button
    updateActiveFilterButton('outliers');
}

/**
 * Reset residuals table filter
 */
export function resetResidualsFilter() {
    renderResidualsTable(STATE.residualsData);
    showToast('Showing all observations', 'info');
    updateActiveFilterButton(null);
}

/**
 * Update active filter button state
 */
function updateActiveFilterButton(activeType) {
    document.querySelectorAll('.quick-action-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-filter') === activeType) {
            btn.classList.add('active');
        }
    });
}

/**
 * Save current model for comparison
 */
export function saveModel() {
    if (!STATE.regressionResults) {
        showToast('No regression results to save', 'warning');
        return;
    }

    const results = STATE.regressionResults;
    const timestamp = new Date().toISOString();

    // Create model summary
    const model = {
        id: Date.now(),
        timestamp: timestamp,
        dataSourceName: STATE.dataSourceName || 'Unknown',
        method: results.method,
        yVariable: STATE.yVariable,
        xVariables: [...STATE.xVariables],
        n: results.n,
        k: results.k,
        rSquared: results.rSquared,
        adjRSquared: results.adjRSquared,
        rmse: results.rmse,
        mae: results.mae,
        aic: results.aic,
        bic: results.bic,
        fStat: results.fStat,
        fPValue: results.fPValue,
        coefficients: results.variableNames.map((name, i) => ({
            variable: name,
            coefficient: results.coefficients[i],
            stdError: results.stdErrors?.[i],
            pValue: results.pValues?.[i]
        })),
        lambda: results.lambda,
        alpha: results.alpha,
        // Include diagnostic test results if available
        diagnostics: STATE.diagnostics ? {
            linearity: STATE.diagnostics.linearity?.map(t => ({
                name: t.shortName,
                pValue: t.p_value,
                isPassed: t.is_passed
            })) || [],
            heteroscedasticity: STATE.diagnostics.heteroscedasticity?.map(t => ({
                name: t.shortName,
                pValue: t.p_value,
                isPassed: t.is_passed
            })) || [],
            normality: STATE.diagnostics.normality?.map(t => ({
                name: t.shortName,
                pValue: t.p_value,
                isPassed: t.is_passed
            })) || [],
            autocorrelation: STATE.diagnostics.autocorrelation?.map(t => ({
                name: t.shortName,
                pValue: t.p_value,
                statistic: t.statistic,
                isPassed: t.is_passed
            })) || [],
            influence: STATE.diagnostics.influence?.map(t => ({
                name: t.shortName,
                maxValue: t.statistic, // Now computed in diagnostics.js
                isPassed: t.is_passed
            })) || []
        } : null
    };

    // Initialize saved models array if needed
    if (!STATE.savedModels) {
        STATE.savedModels = [];
    }

    // Add to saved models (max 5)
    STATE.savedModels.unshift(model);
    if (STATE.savedModels.length > 5) {
        STATE.savedModels.pop();
    }

    showToast(`Model saved (${STATE.savedModels.length}/5)`, 'success');

    // Update model comparison UI if visible
    if (document.getElementById('modelComparisonModal')?.classList.contains('active')) {
        renderModelComparison();
    }
}

/**
 * Show model comparison modal
 */
export function showModelComparison() {
    if (!STATE.savedModels || STATE.savedModels.length === 0) {
        showToast('No saved models to compare. Save a model first.', 'warning');
        return;
    }

    renderModelComparison();

    const modal = document.getElementById('modelComparisonModal');
    if (modal) modal.classList.add('active');
}

/**
 * Render model comparison table
 */
function renderModelComparison() {
    const container = document.getElementById('modelComparisonContent');
    if (!container || !STATE.savedModels) return;

    const models = STATE.savedModels;

    // Build comparison table
    let html = '<div class="model-comparison-table-wrapper"><table class="model-comparison-table"><thead><tr>';
    html += '<th>Metric</th>';

    models.forEach((model, i) => {
        const label = `${model.method.toUpperCase()}${model.lambda !== undefined ? ` (λ=${model.lambda.toFixed(2)})` : ''}`;
        html += `<th>${label}</th>`;
    });

    html += '</tr></thead><tbody>';

    // Data source name row
    html += '<tr><td><strong>Data Source</strong></td>';
    models.forEach(m => {
        html += `<td style="font-weight: 500;">${escapeHtml(m.dataSourceName)}</td>`;
    });
    html += '</tr>';

    // Model info rows
    html += '<tr><td><strong>Y Variable</strong></td>';
    models.forEach(m => {
        html += `<td>${escapeHtml(m.yVariable)}</td>`;
    });
    html += '</tr>';

    html += '<tr><td><strong>Predictors</strong></td>';
    models.forEach(m => {
        html += `<td>${m.xVariables.join(', ')}</td>`;
    });
    html += '</tr>';

    html += '<tr><td><strong>n (obs)</strong></td>';
    models.forEach(m => {
        html += `<td>${m.n}</td>`;
    });
    html += '</tr>';

    html += '<tr><td><strong>k (predictors)</strong></td>';
    models.forEach(m => {
        html += `<td>${m.k}</td>`;
    });
    html += '</tr>';

    // Fit statistics rows
    const metrics = [
        { key: 'rSquared', label: 'R²', format: v => v?.toFixed(4) || 'N/A' },
        { key: 'adjRSquared', label: 'Adj R²', format: v => v?.toFixed(4) || 'N/A' },
        { key: 'rmse', label: 'RMSE', format: v => v?.toFixed(4) || 'N/A', lowerBetter: true },
        { key: 'mae', label: 'MAE', format: v => v?.toFixed(4) || 'N/A', lowerBetter: true },
        { key: 'aic', label: 'AIC', format: v => v?.toFixed(2) || 'N/A', lowerBetter: true },
        { key: 'bic', label: 'BIC', format: v => v?.toFixed(2) || 'N/A', lowerBetter: true },
        { key: 'fStat', label: 'F-Stat', format: v => v?.toFixed(4) || 'N/A' },
        { key: 'fPValue', label: 'F p-value', format: v => v ? formatPValue(v) : 'N/A' }
    ];

    metrics.forEach(metric => {
        html += `<tr><td><strong>${metric.label}</strong></td>`;

        const values = models.map(m => m[metric.key]);
        const bestValue = metric.lowerBetter
            ? Math.min(...values.filter(v => v !== undefined && !isNaN(v)))
            : Math.max(...values.filter(v => v !== undefined && !isNaN(v)));

        models.forEach(m => {
            const val = m[metric.key];
            const formatted = metric.format(val);
            const isBest = val !== undefined && !isNaN(val) && val === bestValue;
            html += `<td class="${isBest ? 'best-metric' : ''}">${formatted}</td>`;
        });

        html += '</tr>';
    });

    html += '</tbody></table></div>';

    // Add best metric indicator
    html += '<p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 12px;">';
    html += '<span style="color: var(--accent-success); font-weight: 600;">●</span> Highlighted values indicate best performance (lower is better for error metrics, higher for R²)';
    html += '</p>';

    // Add diagnostic tests section (if any models have diagnostics)
    const modelsWithDiagnostics = models.filter(m => m.diagnostics);
    if (modelsWithDiagnostics.length > 0) {
        html += '<div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border-color);">';
        html += '<h4 style="font-size: 0.875rem; margin-bottom: 12px;">Diagnostic Test Results</h4>';

        // Get all unique test names across all models
        const testCategories = ['linearity', 'heteroscedasticity', 'normality', 'autocorrelation', 'influence'];
        const categoryLabels = {
            linearity: 'Linearity Tests',
            heteroscedasticity: 'Heteroscedasticity Tests',
            normality: 'Normality Tests',
            autocorrelation: 'Autocorrelation Tests',
            influence: 'Influence Tests'
        };

        testCategories.forEach(category => {
            const testsInCategory = new Set();
            modelsWithDiagnostics.forEach(m => {
                if (m.diagnostics && m.diagnostics[category]) {
                    m.diagnostics[category].forEach(t => testsInCategory.add(t.name));
                }
            });

            if (testsInCategory.size === 0) return;

            // Create collapsible for this category
            const categoryId = `diag-${category}`;
            html += `
                <div class="model-diag-category">
                    <div class="model-diag-header" data-toggle="${categoryId}">
                        <span style="font-weight: 600; font-size: 0.8125rem;">${categoryLabels[category]}</span>
                        <svg class="model-diag-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"/>
                        </svg>
                    </div>
                    <div class="model-diag-content" id="${categoryId}">
                        <table class="model-diag-table">
                            <thead>
                                <tr>
                                    <th>Test</th>
                                    ${models.map(m => `<th>${m.method.toUpperCase()}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
            `;

            // Add rows for each test in this category
            Array.from(testsInCategory).forEach(testName => {
                html += `<tr><td>${escapeHtml(testName)}</td>`;

                models.forEach(m => {
                    if (!m.diagnostics || !m.diagnostics[category]) {
                        html += '<td style="color: var(--text-muted);">No tests</td>';
                        return;
                    }

                    const testResult = m.diagnostics[category].find(t => t.name === testName);
                    if (!testResult) {
                        html += '<td style="color: var(--text-muted);">—</td>';
                        return;
                    }

                    const statusIcon = testResult.isPassed ? '✓' : '✗';
                    const statusClass = testResult.isPassed ? 'pass' : 'fail';

                    // Influence tests (Cook's D, DFBETAS, DFFITS) use maxValue, others use pValue
                    let valueText;
                    let valueLabel;
                    if (testResult.maxValue !== undefined) {
                        valueText = testResult.maxValue < 0.0001 ? '< 0.0001' : testResult.maxValue.toFixed(4);
                        valueLabel = 'Max =';
                    } else if (testResult.pValue !== undefined) {
                        valueText = testResult.pValue < 0.0001 ? '< 0.0001' : testResult.pValue.toFixed(4);
                        valueLabel = 'p =';
                    } else if (testResult.statistic !== undefined) {
                        valueText = testResult.statistic.toFixed(4);
                        valueLabel = 'stat =';
                    } else {
                        valueText = 'N/A';
                        valueLabel = '';
                    }

                    html += `<td class="model-diag-cell">
                        <span class="model-diag-status ${statusClass}">${statusIcon}</span>
                        <span style="font-size: 0.75rem;">${valueLabel} ${valueText}</span>
                    </td>`;
                });

                html += '</tr>';
            });

            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        });

        html += '</div>';
    }

    // Add delete buttons section
    html += '<div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">';
    html += '<h4 style="font-size: 0.875rem; margin-bottom: 8px;">Saved Models</h4>';
    html += '<div style="display: flex; flex-wrap: wrap; gap: 8px;">';

    models.forEach((model, i) => {
        html += `<div style="background: var(--bg-secondary); padding: 8px 12px; border-radius: 6px; display: flex; align-items: center; gap: 8px;">`;
        html += `<span style="font-size: 0.75rem;">${model.method.toUpperCase()} ${new Date(model.timestamp).toLocaleTimeString()}</span>`;
        html += `<button class="tool-btn" data-delete-model="${model.id}" style="padding: 4px 8px; font-size: 0.7rem;">×</button>`;
        html += `</div>`;
    });

    html += '</div></div>';

    container.innerHTML = html;

    // Add delete handlers
    container.querySelectorAll('[data-delete-model]').forEach(btn => {
        btn.addEventListener('click', () => {
            const modelId = parseInt(btn.getAttribute('data-delete-model'));
            deleteModel(modelId);
        });
    });

    // Add collapsible handlers for diagnostic categories
    container.querySelectorAll('[data-toggle]').forEach(header => {
        header.addEventListener('click', () => {
            const targetId = header.getAttribute('data-toggle');
            const content = document.getElementById(targetId);
            if (content) {
                const isCollapsed = content.classList.contains('collapsed');
                content.classList.toggle('collapsed');
                header.classList.toggle('collapsed');
            }
        });
    });
}

/**
 * Delete a saved model
 */
function deleteModel(modelId) {
    if (!STATE.savedModels) return;

    STATE.savedModels = STATE.savedModels.filter(m => m.id !== modelId);
    renderModelComparison();

    if (STATE.savedModels.length === 0) {
        closeModelComparison();
        showToast('All saved models deleted', 'info');
    } else {
        showToast('Model deleted', 'info');
    }
}

/**
 * Close model comparison modal
 */
export function closeModelComparison() {
    const modal = document.getElementById('modelComparisonModal');
    if (modal) modal.classList.remove('active');
}

/**
 * Export model comparison as CSV
 */
export function exportModelComparison() {
    if (!STATE.savedModels || STATE.savedModels.length === 0) {
        showToast('No models to export', 'warning');
        return;
    }

    const models = STATE.savedModels;

    // Build CSV content
    let csv = 'Metric,' + models.map(m => `${m.method.toUpperCase()}${m.lambda !== undefined ? ` (λ=${m.lambda.toFixed(2)})` : ''}`).join(',') + '\n';

    // Data source
    csv += 'Data Source,' + models.map(m => m.dataSourceName).join(',') + '\n';

    // Y Variable
    csv += 'Y Variable,' + models.map(m => m.yVariable).join(',') + '\n';

    // Predictors
    csv += 'Predictors,' + models.map(m => `"${m.xVariables.join(', ')}"`).join(',') + '\n';

    // Sample size and predictors count
    csv += 'n (obs),' + models.map(m => m.n).join(',') + '\n';
    csv += 'k (predictors),' + models.map(m => m.k).join(',') + '\n';

    // Fit statistics
    csv += '\nFit Statistics\n';
    csv += 'R-Squared,' + models.map(m => m.rSquared?.toFixed(4) || 'N/A').join(',') + '\n';
    csv += 'Adj_R_Squared,' + models.map(m => m.adjRSquared?.toFixed(4) || 'N/A').join(',') + '\n';
    csv += 'RMSE,' + models.map(m => m.rmse?.toFixed(4) || 'N/A').join(',') + '\n';
    csv += 'MAE,' + models.map(m => m.mae?.toFixed(4) || 'N/A').join(',') + '\n';
    csv += 'AIC,' + models.map(m => m.aic?.toFixed(2) || 'N/A').join(',') + '\n';
    csv += 'BIC,' + models.map(m => m.bic?.toFixed(2) || 'N/A').join(',') + '\n';
    csv += 'F-Statistic,' + models.map(m => m.fStat?.toFixed(4) || 'N/A').join(',') + '\n';
    csv += 'F p-value,' + models.map(m => m.fPValue?.toFixed(4) || 'N/A').join(',') + '\n';

    // Coefficients
    csv += '\nCoefficients\n';
    const allVars = new Set();
    models.forEach(m => m.coefficients.forEach(c => allVars.add(c.variable)));

    allVars.forEach(varName => {
        csv += varName + ',';
        csv += models.map(m => {
            const coef = m.coefficients.find(c => c.variable === varName);
            return coef ? coef.coefficient.toFixed(4) : 'N/A';
        }).join(',');
        csv += '\n';
    });

    // Diagnostic tests (if any model has them)
    const modelsWithDiagnostics = models.filter(m => m.diagnostics);
    if (modelsWithDiagnostics.length > 0) {
        const testCategories = ['linearity', 'heteroscedasticity', 'normality', 'autocorrelation', 'influence'];

        testCategories.forEach(category => {
            const testsInCategory = new Set();
            modelsWithDiagnostics.forEach(m => {
                if (m.diagnostics && m.diagnostics[category]) {
                    m.diagnostics[category].forEach(t => testsInCategory.add(t.name));
                }
            });

            if (testsInCategory.size === 0) return;

            csv += `\n${category.charAt(0).toUpperCase() + category.slice(1)} Tests\n`;
            Array.from(testsInCategory).forEach(testName => {
                csv += `"${testName}",`;
                csv += models.map(m => {
                    if (!m.diagnostics || !m.diagnostics[category]) {
                        return 'No tests';
                    }
                    const testResult = m.diagnostics[category].find(t => t.name === testName);
                    if (!testResult) return '—';

                    // Influence tests use maxValue, others use pValue
                    let valueText;
                    let valueLabel;
                    if (testResult.maxValue !== undefined) {
                        valueText = testResult.maxValue < 0.0001 ? '<0.0001' : testResult.maxValue.toFixed(4);
                        valueLabel = 'max';
                    } else if (testResult.pValue !== undefined) {
                        valueText = testResult.pValue < 0.0001 ? '<0.0001' : testResult.pValue.toFixed(4);
                        valueLabel = 'p';
                    } else if (testResult.statistic !== undefined) {
                        valueText = testResult.statistic.toFixed(4);
                        valueLabel = 'stat';
                    } else {
                        valueText = 'N/A';
                        valueLabel = '';
                    }

                    return `${testResult.isPassed ? 'PASS' : 'FAIL'} (${valueLabel ? valueLabel + '=' : ''}${valueText})`;
                }).join(',');
                csv += '\n';
            });
        });
    }

    // Create and download file
    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `model_comparison_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();

    showToast('Model comparison exported', 'success');
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize all UI event listeners
 */
export function initUI() {
    // Run regression button
    const runBtn = document.getElementById('runRegressionBtn');
    if (runBtn) {
        runBtn.addEventListener('click', runRegression);
    } else {
        console.warn('[linreg-core] runRegressionBtn not found');
    }

    // Export buttons
    const exportCsvBtn = document.getElementById('exportCsvBtn');
    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', exportResultsAsCSV);
    } else {
        console.warn('[linreg-core] exportCsvBtn not found');
    }

    const exportPngBtn = document.getElementById('exportPngBtn');
    if (exportPngBtn) {
        exportPngBtn.addEventListener('click', () => {
            import('./charts.js').then(({ exportChartsAsPNG }) => {
                exportChartsAsPNG();
            }).catch(err => {
                console.error('Failed to load charts module:', err);
                showToast('Could not export charts', 'error');
            });
        });
    } else {
        console.warn('[linreg-core] exportPngBtn not found');
    }

    // Diagnostic test buttons
    const diagnosticButtons = document.querySelectorAll('.diagnostic-btn');
    if (diagnosticButtons.length > 0) {
        diagnosticButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const testType = btn.getAttribute('data-test');
                runDiagnosticTestByType(testType);
            });
        });
    } else {
        console.warn('[linreg-core] No diagnostic buttons found');
    }

    // Example selector and load button
    const exampleSelect = document.getElementById('exampleSelect');
    if (exampleSelect) {
        exampleSelect.addEventListener('change', () => {
            const descEl = document.getElementById('exampleDescription');
            if (descEl) {
                descEl.textContent = getExampleDescription(exampleSelect.value);
            }
        });
    } else {
        console.warn('[linreg-core] exampleSelect not found');
    }

    const loadExampleBtn = document.getElementById('loadExampleBtn');
    if (loadExampleBtn) {
        loadExampleBtn.addEventListener('click', () => {
            const select = document.getElementById('exampleSelect');
            if (select && window.loadExampleDataset) {
                window.loadExampleDataset(select.value);
            } else {
                console.error('loadExampleDataset not available or exampleSelect not found');
            }
        });
    } else {
        console.warn('[linreg-core] loadExampleBtn not found');
    }

    // Clear paste button
    const clearPasteBtn = document.getElementById('clearPasteBtn');
    if (clearPasteBtn) {
        clearPasteBtn.addEventListener('click', () => {
            const pasteArea = document.getElementById('pasteArea');
            if (pasteArea) pasteArea.value = '';
        });
    }

    // File drop zone click
    const fileDropZone = document.getElementById('fileDropZone');
    if (fileDropZone) {
        fileDropZone.addEventListener('click', () => {
            const fileInput = document.getElementById('fileInput');
            if (fileInput) fileInput.click();
        });
    }

    // File input change
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0 && window.handleFileSelect) {
                window.handleFileSelect(e.target.files);
            }
        });
    }

    // Collapsible headers (disclaimer, help, residuals, licenses)
    setupCollapsible('disclaimerToggle', 'disclaimerContent');
    setupCollapsible('helpToggle', 'helpContent');
    setupCollapsible('correlationToggle', 'correlationContent');
    setupCollapsible('residualsToggle', 'residualsContent');
    setupCollapsible('cvResultsToggle', 'cvResultsContent');
    setupCollapsible('licensesToggle', 'licensesContent');
    setupCollapsible('featureImportanceToggle', 'featureImportanceContent');

    // Cross-validation enable checkbox toggle
    const cvEnableCheck = document.getElementById('cvEnableCheck');
    if (cvEnableCheck) {
        cvEnableCheck.addEventListener('change', () => {
            const cvOptions = document.getElementById('cvOptions');
            if (cvOptions) {
                cvOptions.style.display = cvEnableCheck.checked ? 'block' : 'none';
            }
        });
    }

    // Theme toggle buttons
    const themeButtons = document.querySelectorAll('[data-theme-toggle]');
    themeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const theme = btn.getAttribute('data-theme-toggle');
            if (window.ThemeManager) {
                window.ThemeManager.setTheme(theme);
            }
        });
    });

    // Regression method change
    const methodSelect = document.getElementById('regressionMethodSelect');
    if (methodSelect) {
        methodSelect.addEventListener('change', updateRegressionMethodUI);
    }

    // Lambda slider value display
    const lambdaSlider = document.getElementById('lambdaSlider');
    if (lambdaSlider) {
        lambdaSlider.addEventListener('input', (e) => {
            const lambdaValue = document.getElementById('lambdaValue');
            if (lambdaValue) lambdaValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // Max Iter slider value display
    const maxIterSlider = document.getElementById('maxIterSlider');
    if (maxIterSlider) {
        maxIterSlider.addEventListener('input', (e) => {
            const maxIterValue = document.getElementById('maxIterValue');
            if (maxIterValue) maxIterValue.textContent = e.target.value;
        });
    }

    // Tolerance select value display
    const tolSelect = document.getElementById('tolSelect');
    if (tolSelect) {
        tolSelect.addEventListener('change', (e) => {
            const tolValue = document.getElementById('tolValue');
            if (tolValue) tolValue.textContent = e.target.value;
        });
    }

    // Elastic Net Lambda slider
    const enLambdaSlider = document.getElementById('enLambdaSlider');
    if (enLambdaSlider) {
        enLambdaSlider.addEventListener('input', (e) => {
            const enLambdaValue = document.getElementById('enLambdaValue');
            if (enLambdaValue) enLambdaValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // Elastic Net Alpha slider
    const enAlphaSlider = document.getElementById('enAlphaSlider');
    if (enAlphaSlider) {
        enAlphaSlider.addEventListener('input', (e) => {
            const enAlphaValue = document.getElementById('enAlphaValue');
            if (enAlphaValue) enAlphaValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // Elastic Net Max Iter slider
    const enMaxIterSlider = document.getElementById('enMaxIterSlider');
    if (enMaxIterSlider) {
        enMaxIterSlider.addEventListener('input', (e) => {
            const enMaxIterValue = document.getElementById('enMaxIterValue');
            if (enMaxIterValue) enMaxIterValue.textContent = e.target.value;
        });
    }

    // Elastic Net Tolerance select
    const enTolSelect = document.getElementById('enTolSelect');
    if (enTolSelect) {
        enTolSelect.addEventListener('change', (e) => {
            const enTolValue = document.getElementById('enTolValue');
            if (enTolValue) enTolValue.textContent = e.target.value;
        });
    }

    // LOESS Span slider
    const loessSpanSlider = document.getElementById('loessSpanSlider');
    if (loessSpanSlider) {
        loessSpanSlider.addEventListener('input', (e) => {
            const loessSpanValue = document.getElementById('loessSpanValue');
            if (loessSpanValue) loessSpanValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // Polynomial Degree slider
    const polyDegreeSlider = document.getElementById('polyDegreeSlider');
    if (polyDegreeSlider) {
        polyDegreeSlider.addEventListener('input', (e) => {
            const polyDegreeValue = document.getElementById('polyDegreeValue');
            if (polyDegreeValue) polyDegreeValue.textContent = e.target.value;
        });
    }

    // Polynomial Regularized Degree slider
    const polyRegDegreeSlider = document.getElementById('polyRegDegreeSlider');
    if (polyRegDegreeSlider) {
        polyRegDegreeSlider.addEventListener('input', (e) => {
            const polyRegDegreeValue = document.getElementById('polyRegDegreeValue');
            if (polyRegDegreeValue) polyRegDegreeValue.textContent = e.target.value;
        });
    }

    // Polynomial Lambda slider
    const polyLambdaSlider = document.getElementById('polyLambdaSlider');
    if (polyLambdaSlider) {
        polyLambdaSlider.addEventListener('input', (e) => {
            const polyLambdaValue = document.getElementById('polyLambdaValue');
            if (polyLambdaValue) polyLambdaValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // Polynomial Alpha slider
    const polyAlphaSlider = document.getElementById('polyAlphaSlider');
    if (polyAlphaSlider) {
        polyAlphaSlider.addEventListener('input', (e) => {
            const polyAlphaValue = document.getElementById('polyAlphaValue');
            if (polyAlphaValue) polyAlphaValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    // Observation detail modal close button
    const closeObsDetailBtn = document.getElementById('closeObsDetailBtn');
    if (closeObsDetailBtn) {
        closeObsDetailBtn.addEventListener('click', closeObsDetailModal);
    }

    // Close modal on overlay click
    const obsDetailModal = document.getElementById('obsDetailModal');
    if (obsDetailModal) {
        obsDetailModal.addEventListener('click', (e) => {
            if (e.target === obsDetailModal) {
                closeObsDetailModal();
            }
        });
    }

    // Quick action buttons
    const filterInfluentialBtn = document.getElementById('filterInfluentialBtn');
    if (filterInfluentialBtn) {
        filterInfluentialBtn.addEventListener('click', filterInfluential);
    }

    const filterOutliersBtn = document.getElementById('filterOutliersBtn');
    if (filterOutliersBtn) {
        filterOutliersBtn.addEventListener('click', filterOutliers);
    }

    const resetFilterBtn = document.getElementById('resetFilterBtn');
    if (resetFilterBtn) {
        resetFilterBtn.addEventListener('click', resetResidualsFilter);
    }

    const saveModelBtn = document.getElementById('saveModelBtn');
    if (saveModelBtn) {
        saveModelBtn.addEventListener('click', saveModel);
    }

    const compareModelsBtn = document.getElementById('compareModelsBtn');
    if (compareModelsBtn) {
        compareModelsBtn.addEventListener('click', showModelComparison);
    }

    // Export augmented data button
    const exportAugmentedBtn = document.getElementById('exportAugmentedBtn');
    if (exportAugmentedBtn) {
        exportAugmentedBtn.addEventListener('click', exportAugmentedData);
    }

    // Export/Import model buttons
    const exportModelBtn = document.getElementById('exportModelBtn');
    if (exportModelBtn) {
        exportModelBtn.addEventListener('click', exportModelAsJSON);
    }

    const importModelBtn = document.getElementById('importModelBtn');
    if (importModelBtn) {
        importModelBtn.addEventListener('click', importModelFromJSON);
    }

    // Trace path button
    const tracePathBtn = document.getElementById('tracePathBtn');
    if (tracePathBtn) {
        tracePathBtn.addEventListener('click', runTracePath);
    }

    // Model comparison modal
    const closeModelComparisonBtn = document.getElementById('closeModelComparisonBtn');
    if (closeModelComparisonBtn) {
        closeModelComparisonBtn.addEventListener('click', closeModelComparison);
    }

    const exportComparisonBtn = document.getElementById('exportComparisonBtn');
    if (exportComparisonBtn) {
        exportComparisonBtn.addEventListener('click', exportModelComparison);
    }

    const modelComparisonModal = document.getElementById('modelComparisonModal');
    if (modelComparisonModal) {
        modelComparisonModal.addEventListener('click', (e) => {
            if (e.target === modelComparisonModal) {
                closeModelComparison();
            }
        });
    }

    // Initialize method-specific UI
    updateRegressionMethodUI();

    console.log('[linreg-core] UI initialized');
}

/**
 * Set up a collapsible header/content pair
 * @param {string} toggleId - ID of the toggle element
 * @param {string} contentId - ID of the content element
 */
function setupCollapsible(toggleId, contentId) {
    const toggle = document.getElementById(toggleId);
    const content = document.getElementById(contentId);

    if (!toggle || !content) {
        console.warn(`[linreg-core] Collapsible not found: ${toggleId} / ${contentId}`);
        return;
    }

    const toggleCollapsible = () => {
        const isCollapsed = toggle.classList.contains('collapsed');
        if (isCollapsed) {
            toggle.classList.remove('collapsed');
            content.classList.remove('collapsed');
            toggle.setAttribute('aria-expanded', 'true');
        } else {
            toggle.classList.add('collapsed');
            content.classList.add('collapsed');
            toggle.setAttribute('aria-expanded', 'false');
        }
    };

    toggle.addEventListener('click', toggleCollapsible);
    toggle.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleCollapsible();
        }
    });
}

/**
 * Update regression method UI based on selection
 */
function updateRegressionMethodUI() {
    const method = document.getElementById('regressionMethodSelect')?.value || 'ols';

    // Show/hide Trace Path button
    const tracePathBtn = document.getElementById('tracePathBtn');
    const isRegularized = ['ridge', 'lasso', 'elastic_net'].includes(method);
    if (tracePathBtn) {
        tracePathBtn.style.display = isRegularized ? 'block' : 'none';
    }

    // Hide all method panels
    const regularizedParams = document.getElementById('regularizedParams');
    const elasticNetOptions = document.getElementById('elasticNetOptions');
    const loessOptions = document.getElementById('loessOptions');
    const wlsOptions = document.getElementById('wlsOptions');
    const polynomialOptions = document.getElementById('polynomialOptions');
    const polyRegularizedOptions = document.getElementById('polyRegularizedOptions');

    if (regularizedParams) regularizedParams.style.display = 'none';
    if (elasticNetOptions) elasticNetOptions.style.display = 'none';
    if (loessOptions) loessOptions.style.display = 'none';
    if (wlsOptions) wlsOptions.style.display = 'none';
    if (polynomialOptions) polynomialOptions.style.display = 'none';
    if (polyRegularizedOptions) polyRegularizedOptions.style.display = 'none';

    // Show selected method panel
    switch (method) {
        case 'ridge':
        case 'lasso':
            if (regularizedParams) {
                regularizedParams.style.display = 'block';
                // Show/hide Lasso-specific params
                const lassoParams = document.getElementById('lassoParams');
                if (lassoParams) {
                    lassoParams.style.display = method === 'lasso' ? 'block' : 'none';
                }
            }
            break;
        case 'elastic_net':
            if (elasticNetOptions) elasticNetOptions.style.display = 'block';
            break;
        case 'polynomial':
            if (polynomialOptions) polynomialOptions.style.display = 'block';
            break;
        case 'polynomial_ridge':
        case 'polynomial_lasso':
        case 'polynomial_enet':
            if (polyRegularizedOptions) {
                polyRegularizedOptions.style.display = 'block';
                // Update description based on method
                const descEl = document.getElementById('polyRegDesc');
                const alphaGroup = document.getElementById('polyAlphaGroup');
                if (descEl) {
                    const methodName = method === 'polynomial_ridge' ? 'Polynomial Ridge' :
                                     method === 'polynomial_lasso' ? 'Polynomial Lasso' :
                                     'Polynomial Elastic Net';
                    const description = method === 'polynomial_ridge' ?
                        '<strong>Polynomial Ridge</strong> adds L2 regularization to polynomial regression, shrinking coefficients to handle multicollinearity inherent in polynomial terms.' :
                        method === 'polynomial_lasso' ?
                        '<strong>Polynomial Lasso</strong> adds L1 regularization to polynomial regression, performing variable selection by potentially eliminating higher-order terms.' :
                        '<strong>Polynomial Elastic Net</strong> combines L1 and L2 penalties for polynomial regression, balancing variable selection with multicollinearity handling.';
                    descEl.innerHTML = description;
                }
                // Show alpha slider only for Elastic Net
                if (alphaGroup) {
                    alphaGroup.style.display = method === 'polynomial_enet' ? 'block' : 'none';
                }
            }
            break;
        case 'loess':
            if (loessOptions) loessOptions.style.display = 'block';
            break;
        case 'wls':
            if (wlsOptions) wlsOptions.style.display = 'block';
            // Populate weights selector
            updateWlsWeightsSelector();
            break;
    }

    // Disable CV checkbox for methods without CV support
    const cvSupported = !['wls', 'loess', 'polynomial', 'polynomial_ridge', 'polynomial_lasso', 'polynomial_enet'].includes(method);
    const cvEnableCheck = document.getElementById('cvEnableCheck');
    const cvOptions = document.getElementById('cvOptions');
    if (cvEnableCheck) {
        cvEnableCheck.disabled = !cvSupported;
        if (!cvSupported) {
            cvEnableCheck.checked = false;
            if (cvOptions) cvOptions.style.display = 'none';
        }
    }
    // Update the label to indicate why it's disabled
    const cvLabel = cvEnableCheck?.closest('label');
    const cvLabelSpan = cvLabel?.querySelector('span');
    if (cvLabelSpan) {
        cvLabelSpan.textContent = cvSupported
            ? 'Enable Cross-Validation'
            : 'Cross-Validation (not available for this method)';
        cvLabelSpan.style.opacity = cvSupported ? '1' : '0.5';
    }
}

/**
 * Update the WLS weights selector with available numeric columns
 */
function updateWlsWeightsSelector() {
    const selector = document.getElementById('wlsWeightsSelect');
    if (!selector) return;

    const numericCols = STATE.numericColumns;
    const currentValue = selector.value;

    // Clear and populate selector
    selector.innerHTML = '<option value="">-- Select weights variable --</option>';
    numericCols.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        if (col === currentValue) {
            option.selected = true;
        }
        selector.appendChild(option);
    });

    // Set default to first available numeric column if nothing selected
    if (!currentValue && numericCols.length > 0) {
        // Prefer a column with 'weight' in the name, otherwise use first column
        const weightCol = numericCols.find(c => c.toLowerCase().includes('weight')) || numericCols[0];
        if (weightCol) {
            selector.value = weightCol;
        }
    }
}

/**
 * Show a modal
 * @param {string} modalId - Modal element ID
 */
export function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
    }
}

/**
 * Hide a modal
 * @param {string} modalId - Modal element ID
 */
export function hideModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
    }
}

// ============================================================================
// PREDICTION TOOL
// ============================================================================

/**
 * Initialize the interactive prediction tool
 * @param {Object} results - Current regression results
 */
export function initPredictionTool(results) {
    const container = document.getElementById('predictionInputs');
    if (!container) return;

    const xVars = STATE.xVariables;
    if (!xVars || xVars.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted);">Select predictors to enable the prediction tool</p>';
        return;
    }

    const allowExtrapolation = document.getElementById('allowExtrapolationCheck')?.checked ?? false;

    // Generate input controls for each predictor
    let html = '';
    xVars.forEach((v, i) => {
        const data = STATE.rawData.map(row => row[v]);
        const min = Math.min(...data);
        const max = Math.max(...data);
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        const step = (max - min) / 100;

        html += `
            <div class="selector-group" style="background: var(--bg-card); padding: 12px; border-radius: 8px; border: 1px solid var(--border-color);">
                <label style="font-size: 0.75rem; color: var(--text-muted); display: block; margin-bottom: 8px;">${escapeHtml(v)}</label>
                
                <input type="number" class="pred-input" id="pred-input-${i}" data-idx="${i}" 
                    value="${mean.toFixed(2)}" 
                    ${!allowExtrapolation ? `min="${min}" max="${max}"` : ''} 
                    step="any"
                    style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--bg-input); font-weight: 600; margin-bottom: ${allowExtrapolation ? '0' : '12px'};">
                
                <div id="slider-cont-${i}" style="display: ${allowExtrapolation ? 'none' : 'block'};">
                    <input type="range" class="pred-slider" id="pred-slider-${i}" data-idx="${i}" 
                        min="${min}" max="${max}" step="${step}" value="${mean}" style="width: 100%;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--text-muted); margin-top: 4px;">
                        <span>${min.toFixed(1)}</span>
                        <span>${max.toFixed(1)}</span>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;

    // Synchronize inputs and sliders
    const inputs = container.querySelectorAll('.pred-input');
    const sliders = container.querySelectorAll('.pred-slider');

    sliders.forEach(slider => {
        slider.addEventListener('input', (e) => {
            const idx = e.target.getAttribute('data-idx');
            const input = document.getElementById(`pred-input-${idx}`);
            if (input) {
                input.value = parseFloat(e.target.value).toFixed(2);
                updatePrediction(results);
            }
        });
    });

    inputs.forEach(input => {
        input.addEventListener('input', (e) => {
            const idx = e.target.getAttribute('data-idx');
            const slider = document.getElementById(`pred-slider-${idx}`);
            if (slider && !isNaN(parseFloat(e.target.value))) {
                slider.value = e.target.value;
            }
            updatePrediction(results);
        });
    });

    // Attach extrapolation toggle listener
    const extrapolationCheck = document.getElementById('allowExtrapolationCheck');
    if (extrapolationCheck) {
        extrapolationCheck.onchange = () => initPredictionTool(results);
    }

    // Reset logic
    const resetBtn = document.getElementById('resetPredictionBtn');
    if (resetBtn) {
        resetBtn.onclick = () => {
            xVars.forEach((v, i) => {
                const data = STATE.rawData.map(row => row[v]);
                const mean = data.reduce((a, b) => a + b, 0) / data.length;
                const input = document.getElementById(`pred-input-${i}`);
                const slider = document.getElementById(`pred-slider-${i}`);
                if (input) input.value = mean.toFixed(2);
                if (slider) slider.value = mean;
            });
            updatePrediction(results);
        };
    }

    updatePrediction(results);
}

/**
 * Calculate and update the predicted value
 * @param {Object} results - Current regression results
 */
async function updatePrediction(results) {
    const resultEl = document.getElementById('predictionResult');
    if (!resultEl) return;

    try {
        const inputs = document.querySelectorAll('.pred-input');
        const values = Array.from(inputs).map(inp => parseFloat(inp.value) || 0);

        let prediction;

        if (results.method === 'loess') {
            const newX = values.map(v => [v]);
            const predictions = await WasmRegression.loessPredict(
                newX,
                STATE.xVariables,
                STATE.yVariable,
                results.span,
                results.degree,
                results.robustIterations,
                results.surface
            );
            prediction = predictions[0];
        } else {
            // Linear Model: intercept + sum(coefficients * variables)
            prediction = results.coefficients[0];
            values.forEach((val, i) => {
                prediction += results.coefficients[i + 1] * val;
            });
        }

        resultEl.textContent = (prediction !== undefined && !isNaN(prediction)) 
            ? prediction.toFixed(4) 
            : '---';
            
    } catch (e) {
        console.error('[linreg-core] Prediction error:', e);
        resultEl.textContent = 'Error';
    }
}
