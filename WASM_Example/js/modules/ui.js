// ============================================================================
// ui.js - DOM manipulation and UI management
// ============================================================================

import { STATE, updateState, escapeHtml, formatPValue, showToast, getCurrentTheme } from './utils.js';
import { updateCharts } from './charts.js';
import { runDiagnostics } from './diagnostics.js';
import { calculateRegression, calculateRidgeRegression, calculateLassoRegression, calculateElasticNetRegression } from './core.js';
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

    // Update statistics cards
    updateStatisticsCards(results, isRegularized);

    // Update equation
    updateEquation(results);

    // Update VIF display (only for OLS)
    if (isRegularized) {
        hideVIFSection();
    } else {
        updateVIFDisplay(results);
    }

    // Update coefficients table
    updateCoefficientsTable(results, isRegularized);

    // Update residuals table
    updateResidualsTable(results);

    // Update charts
    updateCharts(results);
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
 * Update VIF display
 */
function updateVIFDisplay(results) {
    const vifSection = document.getElementById('vifSection');
    const vifNote = document.getElementById('vifNote');
    const vifBody = document.getElementById('vifBody');

    if (!vifSection || !vifNote || !vifBody) return;

    if (results.k <= 1) {
        vifNote.style.display = 'block';
        vifSection.style.display = 'none';
        return;
    }

    vifNote.style.display = 'none';
    vifSection.style.display = 'block';
    vifBody.innerHTML = '';

    results.vif.forEach(vif => {
        const vifValue = vif.vif === Infinity ? '∞' : vif.vif.toFixed(2);
        let colorClass = 'success';
        if (vif.vif > 10) colorClass = 'error';
        else if (vif.vif > 5) colorClass = 'warning';

        vifBody.innerHTML += `
            <tr>
                <td>${escapeHtml(vif.variable)}</td>
                <td class="${colorClass}">${vifValue}</td>
                <td>${vif.rsquared.toFixed(4)}</td>
                <td>${vif.interpretation}</td>
            </tr>
        `;
    });
}

/**
 * Hide VIF section
 */
function hideVIFSection() {
    const vifSection = document.getElementById('vifSection');
    const vifNote = document.getElementById('vifNote');
    if (vifSection) vifSection.style.display = 'none';
    if (vifNote) vifNote.style.display = 'none';
}

/**
 * Update regression equation display
 */
function updateEquation(results) {
    const equationEl = document.getElementById('equation');
    if (!equationEl) return;

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

    // OLS full table
    const stdErrs = results.stdErrors;
    const tStats = results.tStats;
    const pValues = results.pValues;
    const ci = results.confidenceIntervals;

    let html = '';
    names.forEach((name, i) => {
        const signClass = coeffs[i] >= 0 ? 'positive' : 'negative';
        html += `<tr>
            <td>${escapeHtml(name)}</td>
            <td class="${signClass}">${coeffs[i].toFixed(4)}</td>
            <td>${stdErrs[i].toFixed(4)}</td>
            <td>${tStats[i].toFixed(4)}</td>
            <td>${formatPValue(pValues[i])}</td>
            <td>${ci[i][0].toFixed(4)}</td>
            <td>${ci[i][1].toFixed(4)}</td>
        </tr>`;
    });

    tbody.innerHTML = html;

    const thead = table.querySelector('thead tr');
    if (thead) {
        thead.innerHTML = '<th>Variable</th><th>Coefficient</th><th>Std Error</th><th>t-stat</th><th>p-value</th><th>95% CI Lower</th><th>95% CI Upper</th>';
    }
}

/**
 * Update residuals table
 */
function updateResidualsTable(results) {
    const tbody = document.getElementById('residualsBody');
    if (!tbody) return;

    const { residuals, predictions, standardizedResiduals } = results;
    const yData = STATE.rawData.map(row => row[STATE.yVariable]);

    let html = '';
    yData.forEach((actual, i) => {
        const residual = residuals[i];
        const stdResid = standardizedResiduals[i];
        const stdClass = Math.abs(stdResid) > 2 ? 'outlier' : '';
        html += `<tr>
            <td>${i + 1}</td>
            <td>${actual.toFixed(4)}</td>
            <td>${predictions[i].toFixed(4)}</td>
            <td class="${stdClass}">${residual.toFixed(4)}</td>
            <td class="${stdClass}">${stdResid.toFixed(4)}</td>
        </tr>`;
    });

    tbody.innerHTML = html;
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
            default:
                results = await calculateRegression(STATE.yVariable, STATE.xVariables);
        }

        updateState({ regressionResults: results });
        updateResultsDisplay(results);
        showToast(`Regression complete (${formatMethodName(results.method)})`, 'success');

    } catch (error) {
        console.error('Regression error:', error);
        showToast(error.message, 'error');
    }
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

    // Collapsible headers (disclaimer, help, residuals)
    setupCollapsible('disclaimerToggle', 'disclaimerContent');
    setupCollapsible('helpToggle', 'helpContent');
    setupCollapsible('residualsToggle', 'residualsContent');

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

    // Hide all method panels
    const regularizedParams = document.getElementById('regularizedParams');
    const elasticNetOptions = document.getElementById('elasticNetOptions');

    if (regularizedParams) regularizedParams.style.display = 'none';
    if (elasticNetOptions) elasticNetOptions.style.display = 'none';

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
