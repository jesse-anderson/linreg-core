// ============================================================================
// data.js - Data import and processing
// ============================================================================

import { STATE, updateState, showToast, escapeHtml } from './utils.js';
import { WasmRegression } from './core.js';

// ============================================================================
// EXAMPLE DATASETS
// ============================================================================

export const EXAMPLES = {
    simple: {
        description: 'Basic simple linear regression with a clear linear relationship between study hours and test scores.',
        csv: `Study_Hours,Test_Score
1,52
2,58
3,62
4,68
5,75
6,79
7,85
8,88
9,92
10,95
2.5,60
4.5,72
6.5,82
1.5,55
7.5,86
3.5,66
5.5,77
8.5,90
9.5,94
0.5,48`,
        yVar: 'Test_Score',
        xVars: ['Study_Hours'],
        toast: 'Simple X,Y loaded (20 samples)'
    },
    housing: {
        description: 'Demonstrates multiple regression with real-world relationships between price, square footage, bedrooms, and age.',
        csv: `Price,Square_Feet,Bedrooms,Age,Distance_to_City
245.5,1200,3,15,8.2
312.8,1800,4,10,5.5
198.4,950,2,25,12.3
425.6,2400,4,5,3.2
278.9,1450,3,8,6.8
356.2,2000,4,12,4.5
189.5,1100,2,20,15.0
512.3,2800,5,2,1.8
234.7,1350,3,18,9.5
298.1,1650,3,7,5.2
445.8,2200,4,3,2.5
167.9,900,2,30,18.5
367.4,1950,4,6,4.0
289.6,1500,3,14,7.8
198.2,1050,2,22,11.5
478.5,2600,5,1,2.0
256.3,1300,3,16,8.5
334.7,1850,4,9,5.8
178.5,1000,2,28,14.2
398.9,2100,4,4,3.5
223.4,1250,3,19,10.5
312.5,1700,3,11,6.2
156.8,850,2,35,20.0
423.7,2350,4,3,2.8
267.9,1400,3,13,7.2`,
        yVar: 'Price',
        xVars: ['Square_Feet', 'Bedrooms', 'Age'],
        toast: 'Housing Prices loaded (25 samples, 5 variables)'
    },
    singular: {
        description: 'This dataset contains perfect multicollinearity (X3 = X1 + X2 exactly). The matrix cannot be inverted, demonstrating a fundamental limitation of OLS.',
        csv: `Y,X1,X2,X3
10,1,2,3
15,2,3,5
20,3,4,7
25,4,5,9
30,5,6,11
35,6,7,13
40,7,8,15
45,8,9,17
50,9,10,19
55,10,11,21`,
        yVar: 'Y',
        xVars: ['X1', 'X2', 'X3'],
        toast: 'Singular matrix example loaded (will fail regression)'
    },
    messy: {
        description: 'Sales data from 3 different store tiers (Budget, Mid-range, Luxury). The confounding variable (Store_Tier) creates 3 distinct clusters. A single regression line will have poor fit—the data should be analyzed separately by tier.',
        csv: `Store_Tier,Marketing_Spend,Sales
Budget,1000,15000
Budget,1200,16500
Budget,800,14000
Budget,1500,18000
Budget,900,15500
Budget,1100,16200
Budget,1300,17000
Budget,700,13500
Budget,1400,17500
Budget,850,14800
Mid_range,3000,35000
Mid_range,3500,38000
Mid_range,2800,33000
Mid_range,4000,42000
Mid_range,3200,36000
Mid_range,3800,40000
Mid_range,2500,31000
Mid_range,4200,43500
Mid_range,2900,34000
Mid_range,3600,39000
Luxury,8000,55000
Luxury,9000,58000
Luxury,7500,53000
Luxury,10000,62000
Luxury,8500,56000
Luxury,9500,60000
Luxury,7000,51000
Luxury,10500,64000
Luxury,7800,54000
Luxury,9200,59000`,
        yVar: 'Sales',
        xVars: ['Marketing_Spend'],
        toast: 'Messy data loaded (30 samples with hidden clustering)'
    }
};

/**
 * Load an example dataset
 * @param {string} exampleKey - Key of the example to load
 */
export function loadExampleDataset(exampleKey) {
    const example = EXAMPLES[exampleKey];
    if (!example) {
        showToast(`Example "${exampleKey}" not found`, 'error');
        return;
    }

    parseCSVData(example.csv);
    showToast(example.toast, 'success');

    // Set up the regression configuration for this example
    updateState({
        yVariable: example.yVar,
        xVariables: [...example.xVars]
    });

    // Re-render column selectors with the correct pre-selected values
    // Note: We need to do this after the dataLoaded event fires to override defaults
    requestAnimationFrame(() => {
        if (window.updateColumnSelectors) {
            window.updateColumnSelectors();
        }
    });
}

/**
 * Update the example description in the UI
 * @param {string} exampleKey - Key of the example
 * @returns {string} Description text
 */
export function getExampleDescription(exampleKey) {
    const example = EXAMPLES[exampleKey];
    return example ? example.description : '';
}

// ============================================================================
// FILE HANDLING
// ============================================================================

/**
 * Handle file selection from file input
 * @param {FileList} files - Selected files
 */
export async function handleFileSelect(files) {
    if (!files || files.length === 0) return;

    const file = files[0];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (file.size > maxSize) {
        showToast('File exceeds 50MB limit', 'error');
        return;
    }

    const extension = file.name.split('.').pop().toLowerCase();

    try {
        if (extension === 'xlsx' || extension === 'xls') {
            await handleExcelFile(file);
        } else if (extension === 'csv') {
            await handleCSVFile(file);
        } else {
            showToast('Unsupported file type. Please use CSV, XLSX, or XLS.', 'error');
        }
    } catch (error) {
        console.error('File import error:', error);
        showToast(`Import failed: ${error.message}`, 'error');
    }

    // Reset file input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) fileInput.value = '';
}

/**
 * Handle Excel file import
 * @param {File} file - Excel file
 */
export async function handleExcelFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer);

    if (workbook.SheetNames.length > 1) {
        // Show sheet selector modal
        updateState({ pendingWorkbook: workbook });
        showSheetSelector(workbook.SheetNames);
    } else {
        // Single sheet - import directly
        await importExcelSheet(workbook, workbook.SheetNames[0]);
    }
}

/**
 * Import a specific sheet from an Excel workbook
 * @param {Object} workbook - XLSX workbook object
 * @param {string} sheetName - Sheet name to import
 */
export async function importExcelSheet(workbook, sheetName) {
    const worksheet = workbook.Sheets[sheetName];
    const csv = XLSX.utils.sheet_to_csv(worksheet);
    await parseCSVData(csv);
    showToast(`Imported sheet: ${sheetName}`, 'success');
}

/**
 * Handle CSV file import
 * @param {File} file - CSV file
 */
export async function handleCSVFile(file) {
    const text = await file.text();
    await parseCSVData(text);
    showToast('CSV file imported successfully', 'success');
}

/**
 * Parse CSV data using WASM
 * @param {string} csvText - CSV text content
 */
export async function parseCSVData(csvText) {
    if (!csvText || !csvText.trim()) {
        showToast('Empty data', 'error');
        return;
    }

    // Ensure WASM is ready
    if (!WasmRegression.isReady()) {
        showToast('WASM engine loading... please try again in a moment', 'warning');
        return;
    }

    try {
        // Use Rust WASM for robust CSV parsing
        const resultJson = WasmRegression.parseCsv(csvText);
        const result = JSON.parse(resultJson);

        if (result.error) {
            showToast(`Parse error: ${result.error}`, 'error');
            return;
        }

        if (result.data.length === 0) {
            showToast('No data rows found', 'error');
            return;
        }

        updateState({
            headers: result.headers,
            rawData: result.data,
            numericColumns: result.numeric_columns
        });

        if (STATE.numericColumns.length < 2) {
            showToast('Need at least 2 numeric columns for regression', 'error');
            return;
        }

        // Trigger UI updates
        dispatchDataLoadedEvent();

        showToast(`Loaded ${STATE.rawData.length} rows with ${STATE.headers.length} columns (Rust parser)`, 'success');

    } catch (e) {
        console.error('CSV Parse Error:', e);
        showToast(`Error parsing CSV: ${e.message}`, 'error');
    }
}

/**
 * Handle paste event for data input
 * @param {string} text - Pasted text
 */
export function handlePaste(text) {
    if (!text.trim()) return;

    try {
        // Check if it's tab-separated (Excel paste) or CSV
        if (text.includes('\t')) {
            // Convert tabs to commas for CSV parsing
            const csv = text.replace(/\t/g, ',');
            parseCSVData(csv);
            showToast('Data pasted successfully', 'success');
        } else if (text.includes(',') || text.includes('\n')) {
            parseCSVData(text);
            showToast('Data pasted successfully', 'success');
        }
    } catch (error) {
        showToast(`Parse error: ${error.message}`, 'error');
    }
}

/**
 * Show sheet selector modal for Excel files with multiple sheets
 * @param {Array<string>} sheetNames - List of sheet names
 */
function showSheetSelector(sheetNames) {
    // Remove existing modal if present
    const existingModal = document.getElementById('sheetSelectorModal');
    if (existingModal) existingModal.remove();

    const modal = document.createElement('div');
    modal.id = 'sheetSelectorModal';
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 400px;">
            <div class="modal-header">
                <h3>Select Sheet</h3>
                <button class="close-btn" onclick="document.getElementById('sheetSelectorModal').remove()">&times;</button>
            </div>
            <div class="modal-body">
                <p>This workbook contains multiple sheets. Please select one to import:</p>
                <div class="sheet-list">
                    ${sheetNames.map(name => `
                        <button class="sheet-btn" data-sheet="${escapeHtml(name)}">${escapeHtml(name)}</button>
                    `).join('')}
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);

    // Add click handlers
    modal.querySelectorAll('.sheet-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const sheetName = btn.dataset.sheet;
            importExcelSheet(STATE.pendingWorkbook, sheetName);
            modal.remove();
            updateState({ pendingWorkbook: null });
        });
    });
}

/**
 * Dispatch custom event when data is loaded
 */
function dispatchDataLoadedEvent() {
    const event = new CustomEvent('dataLoaded', {
        detail: {
            headers: STATE.headers,
            numericColumns: STATE.numericColumns,
            rowCount: STATE.rawData.length
        }
    });
    window.dispatchEvent(event);
}

// ============================================================================
// DATA VALIDATION
// ============================================================================

/**
 * Validate that a variable exists in the data
 * @param {string} varName - Variable name
 * @returns {boolean} True if variable exists
 */
export function variableExists(varName) {
    return STATE.headers.includes(varName);
}

/**
 * Check if a variable is numeric
 * @param {string} varName - Variable name
 * @returns {boolean} True if variable is numeric
 */
export function isNumericVariable(varName) {
    return STATE.numericColumns.includes(varName);
}

/**
 * Get data for a specific variable
 * @param {string} varName - Variable name
 * @returns {Array<number>} Array of values
 */
export function getVariableData(varName) {
    return STATE.rawData.map(row => row[varName]);
}

/**
 * Get summary statistics for a variable
 * @param {string} varName - Variable name
 * @returns {Object} Summary statistics
 */
export function getVariableSummary(varName) {
    const data = getVariableData(varName).filter(v => typeof v === 'number' && !isNaN(v));

    if (data.length === 0) {
        return null;
    }

    const sorted = [...data].sort((a, b) => a - b);
    const n = data.length;
    const sum = data.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    const variance = data.reduce((acc, val) => acc + (val - mean) ** 2, 0) / (n - 1);
    const std = Math.sqrt(variance);

    return {
        count: n,
        min: sorted[0],
        max: sorted[n - 1],
        mean: mean,
        median: n % 2 === 0
            ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
            : sorted[Math.floor(n / 2)],
        std: std,
        q1: sorted[Math.floor(n * 0.25)],
        q3: sorted[Math.floor(n * 0.75)]
    };
}
