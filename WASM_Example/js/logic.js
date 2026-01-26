// ============================================================================
// logic.js - Main entry point that ties all modules together
// ============================================================================

// Import WASM initialization and core functions
import { initWasm, WasmRegression } from './modules/core.js';

// Import utility functions and state management
import {
    STATE, ALLOWED_DOMAINS, ThemeManager, showToast
} from './modules/utils.js';

// Import data handling
import {
    handleFileSelect, loadExampleDataset, getExampleDescription,
    parseCSVData, handlePaste
} from './modules/data.js';

// Import regression functions
import {
    calculateRegression,
    calculateRidgeRegression,
    calculateLassoRegression,
    calculateElasticNetRegression
} from './modules/core.js';

// Import diagnostics
import { runDiagnostics } from './modules/diagnostics.js';

// Import UI functions
import {
    updateDataPreview,
    updateColumnSelectors,
    updateResultsDisplay,
    updateDiagnosticsDisplay,
    initUI,
    runRegression,
    runDiagnosticTests,
    exportResultsAsCSV
} from './modules/ui.js';

// Import chart functions
import { updateCharts, exportChartsAsPNG } from './modules/charts.js';

// Import regularized regression helpers
import { formatMethodName } from './modules/regularized.js';

// ============================================================================
// SECURITY CHECK
// ============================================================================

if (!ALLOWED_DOMAINS.includes(window.location.hostname)) {
    document.body.innerHTML = `
        <div style="display:flex;height:100vh;align-items:center;justify-content:center;flex-direction:column;color:#fafafa;background:#09090b;font-family:'Segoe UI', sans-serif;text-align:center;padding:20px;">
            <div style="max-width:600px;background:#18181b;padding:40px;border-radius:12px;border:1px solid #27272a;box-shadow:0 20px 25px -5px rgba(0,0,0,0.5);">
                <h1 style="color:#ef4444;margin-bottom:1.5rem;font-size:2rem;">Access Denied</h1>
                <p style="color:#e4e4e7;font-size:1.1rem;line-height:1.6;margin-bottom:1.5rem;">
                    This application is proprietary software restricted to authorized domains.
                </p>
                <p style="color:#a1a1aa;font-size:0.95rem;line-height:1.6;margin-bottom:2rem;">
                    The developer has implemented domain-locking security protocols.
                    Execution of this tool outside of <strong>jesse-anderson.net</strong> and authorized subdomains is strictly prohibited and functionally disabled.
                </p>
                <div style="font-family:monospace;background:#000;padding:12px;border-radius:6px;color:#ef4444;font-size:0.85rem;">
                    Error: DOMAIN_VALIDATION_FAILURE<br>
                    Origin: ${window.location.hostname}
                </div>
            </div>
        </div>
    `;
    throw new Error(`Unauthorized domain: ${window.location.hostname}`);
}

// ============================================================================
// INITIALIZE THEME
// ============================================================================

ThemeManager.init();

// ============================================================================
// WASM INITIALIZATION
// ============================================================================

let wasmReady = false;
let wasmError = null;

async function initWasmWrapper() {
    try {
        await initWasm();
        wasmReady = true;

        // Make WasmRegression available globally for inline scripts
        window.WasmRegression = WasmRegression;

        // Trigger pending regression if any
        if (window.pendingRegression) {
            runRegressionFromPending();
            window.pendingRegression = null;
        }
    } catch (e) {
        wasmError = e;
        console.error('[linreg-core] Failed to load WASM:', e);
    }
}

initWasmWrapper();

// ============================================================================
// EXPOSE FUNCTIONS TO WINDOW (for inline HTML event handlers)
// ============================================================================

// Data handling
window.handleFileSelect = handleFileSelect;
window.loadExampleDataset = loadExampleDataset;
window.handlePaste = handlePaste;

// UI updates
window.updateDataPreview = updateDataPreview;
window.updateColumnSelectors = updateColumnSelectors;
window.updateResultsDisplay = updateResultsDisplay;
window.updateDiagnosticsDisplay = updateDiagnosticsDisplay;
window.updateCharts = updateCharts;

// Regression
window.runRegression = runRegression;
window.runDiagnosticTests = runDiagnosticTests;

// Export
window.exportResultsAsCSV = exportResultsAsCSV;
window.exportChartsAsPNG = exportChartsAsPNG;

// Utilities
window.showToast = showToast;
window.formatMethodName = formatMethodName;
window.getExampleDescription = getExampleDescription;
window.ThemeManager = ThemeManager;

// ============================================================================
// DOMCONTENTLOADED EVENT LISTENER
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI event listeners
    initUI();

    // Set up drag-and-drop for file input
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        dropZone.addEventListener('drop', handleDrop, false);
    }

    // Set up paste event listener
    document.addEventListener('paste', handlePasteEvent);

    // Listen for data loaded event
    window.addEventListener('dataLoaded', handleDataLoaded);

    // Listen for theme changes
    const themeObserver = new MutationObserver(() => {
        if (STATE.regressionResults) {
            updateCharts(STATE.regressionResults);
        }
    });

    themeObserver.observe(document.body, {
        attributes: true,
        attributeFilter: ['class']
    });
});

// ============================================================================
// DRAG AND DROP HANDLERS
// ============================================================================

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
        dropZone.classList.add('drag-over');
    }
}

function unhighlight() {
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
        dropZone.classList.remove('drag-over');
    }
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFileSelect(files);
}

function handlePasteEvent(e) {
    const pasteArea = document.getElementById('pasteArea');
    if (!pasteArea || document.activeElement !== pasteArea) return;

    e.preventDefault();
    const text = (e.clipboardData || window.clipboardData).getData('text');
    handlePaste(text);
}

// ============================================================================
// DATA LOADED EVENT HANDLER
// ============================================================================

function handleDataLoaded(e) {
    const { headers, numericColumns, rowCount } = e.detail;
    console.log(`[linreg-core] Data loaded: ${rowCount} rows, ${headers.length} columns, ${numericColumns.length} numeric`);

    // Update UI to show the loaded data
    const dataPreview = document.getElementById('dataPreview');
    if (dataPreview) dataPreview.style.display = 'block';

    updateDataPreview();
    updateColumnSelectors();
}

// ============================================================================
// PENDING REGRESSION HANDLER
// ============================================================================

async function runRegressionFromPending() {
    const { yVar, xVars } = window.pendingRegression;
    try {
        const results = await calculateRegression(yVar, xVars);
        STATE.regressionResults = results;
        updateResultsDisplay(results);
        showToast('Regression complete (OLS)', 'success');
    } catch (error) {
        console.error('Regression error:', error);
        showToast(error.message, 'error');
    }
}

// ============================================================================
// EXPORT MODULES FOR TESTING/DEBUGGING
// ============================================================================

if (typeof window !== 'undefined') {
    window.linregModules = {
        core: { calculateRegression, calculateRidgeRegression, calculateLassoRegression, calculateElasticNetRegression },
        data: { handleFileSelect, loadExampleDataset, parseCSVData },
        diagnostics: { runDiagnostics },
        ui: { updateDataPreview, updateColumnSelectors, updateResultsDisplay },
        charts: { updateCharts },
        utils: { STATE, showToast },
        regularized: { formatMethodName }
    };
}
