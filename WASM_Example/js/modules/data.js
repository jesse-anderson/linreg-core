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
        name: 'Study Hours vs Test Scores',
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
        name: 'Housing Prices Dataset',
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
        name: 'Singular Matrix (Multicollinear)',
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
        name: 'Store Sales (Confounded)',
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
    },
    fuel_efficiency: {
        name: 'Fuel Efficiency (Heteroscedastic)',
        description: 'Highway MPG vs engine size. Small engines have 40-50 test vehicles (tight clusters). Large engines have only 2-8 test vehicles plus single-vehicle outliers tested under favorable conditions that bias OLS upward. Switch to WLS with Test_Count weights to downweight unreliable single-vehicle tests and recover the true steep decline.',
        csv: `Engine_Size,Highway_MPG,Test_Count
1.0,43.2,50
1.0,42.6,50
1.0,43.5,50
1.0,42.8,50
1.0,43.0,50
1.0,43.4,50
1.5,40.2,45
1.5,39.8,45
1.5,40.8,45
1.5,40.5,45
1.5,41.0,45
2.0,37.5,40
2.0,38.5,40
2.0,37.0,40
2.0,38.2,40
2.0,37.8,40
2.5,35.0,25
2.5,34.5,25
2.5,36.0,25
2.5,33.5,25
3.0,33.0,15
3.0,31.0,15
3.0,34.5,15
3.0,30.5,15
3.5,30.0,8
3.5,28.5,8
3.5,37.0,1
4.0,27.0,5
4.0,25.5,5
4.0,36.0,1
4.5,24.0,3
4.5,22.0,3
4.5,35.0,1
5.0,21.0,2
5.0,19.0,2
5.0,33.0,1
5.5,18.0,3
5.5,32.0,1
6.0,15.0,2
6.0,30.0,1`,
        yVar: 'Highway_MPG',
        xVars: ['Engine_Size'],
        toast: 'Fuel Efficiency loaded (40 samples, heteroscedastic — try WLS with Test_Count weights)'
    },
    cell_activity: {
        name: 'Cell Activity (Substrate Cycles)',
        description: 'Enzyme activity measured hourly as substrate is consumed via Michaelis-Menten kinetics. Fresh substrate is added every 24 hours (visible as activity jumps). Switch to LOESS to capture the non-linear decay pattern.',
        csv: `Hour,Substrate_mM,Cell_Activity
1,100.0,89.1
2,88.7,84.0
3,78.7,84.8
4,69.8,79.1
5,61.9,82.2
6,54.9,78.2
7,48.7,80.6
8,43.2,71.4
9,38.3,72.4
10,34.0,72.6
11,30.1,64.9
12,26.7,66.5
13,23.7,57.2
14,21.0,59.5
15,18.6,54.7
16,16.5,55.9
17,14.7,47.3
18,13.0,47.3
19,11.5,42.0
20,10.2,43.3
21,9.1,34.3
22,8.0,35.8
23,7.1,31.6
24,6.3,33.8
25,100.0,89.0
26,88.7,90.8
27,78.7,84.5
28,69.8,85.9
29,61.9,87.3
30,54.9,80.3
31,48.7,76.5
32,43.2,79.2
33,38.3,76.4
34,34.0,71.6
35,30.1,67.6
36,26.7,70.3
37,23.7,64.5
38,21.0,56.8
39,18.6,60.2
40,16.5,53.8
41,14.7,55.2
42,13.0,49.2
43,11.5,42.6
44,10.2,44.9
45,9.1,40.9
46,8.0,35.2
47,7.1,38.1
48,6.3,31.3
49,100.0,93.6
50,88.7,87.1
51,78.7,89.4
52,69.8,89.5
53,61.9,83.6
54,54.9,87.1
55,48.7,80.9
56,43.2,76.8
57,38.3,80.9
58,34.0,75.5
59,30.1,68.6
60,26.7,69.7
61,23.7,69.1
62,21.0,61.6
63,18.6,60.0
64,16.5,60.8
65,14.7,52.5
66,13.0,52.7
67,11.5,44.8
68,10.2,48.1
69,9.1,43.0
70,8.0,38.7
71,7.1,40.8
72,6.3,31.7`,
        yVar: 'Cell_Activity',
        xVars: ['Hour'],
        toast: 'Cell Activity loaded (72 samples, 3 substrate cycles — try LOESS)'
    },
    projectile: {
        name: 'Projectile Motion (Polynomial)',
        description: 'Height of a projectile over time follows a parabolic arc. With only Time_s, OLS gives a poor linear fit. Add Time_Squared as a predictor to capture the quadratic relationship. Time_Cubed is included for experimentation.',
        csv: `Time_s,Height_m,Time_Squared,Time_Cubed
0.0,2.5,0.00,0.000
0.1,3.2,0.01,0.001
0.2,7.0,0.04,0.008
0.4,8.9,0.16,0.064
0.5,12.3,0.25,0.125
0.6,11.2,0.36,0.216
0.8,15.6,0.64,0.512
0.9,14.8,0.81,0.729
1.0,17.4,1.00,1.000
1.2,20.1,1.44,1.728
1.3,19.2,1.69,2.197
1.4,21.8,1.96,2.744
1.6,21.3,2.56,4.096
1.7,20.5,2.89,4.913
1.8,22.9,3.24,5.832
2.0,21.8,4.00,8.000
2.1,23.4,4.41,9.261
2.2,20.8,4.84,10.648
2.4,22.2,5.76,13.824
2.5,20.5,6.25,15.625
2.6,22.5,6.76,17.576
2.8,19.2,7.84,21.952
3.0,18.1,9.00,27.000
3.1,15.7,9.61,29.791
3.2,16.7,10.24,32.768
3.4,12.7,11.56,39.304
3.5,13.3,12.25,42.875
3.6,10.4,12.96,46.656
3.8,8.0,14.44,54.872
3.9,5.0,15.21,59.319
4.0,3.9,16.00,64.000`,
        yVar: 'Height_m',
        xVars: ['Time_s'],
        toast: 'Projectile loaded (31 samples — add Time_Squared for quadratic fit)'
    },
    cauchy_noise: {
        name: 'Cauchy Noise (OLS Breaker)',
        description: 'Linear data (Y = 10 + 2X) corrupted by heavy-tailed Cauchy noise. A few extreme outliers completely destroy the OLS fit. Compare Y_Observed vs Y_True to see the real relationship underneath. Quantile regression (coming soon) handles this correctly.',
        csv: `X,Y_Observed,Y_True
1,14.3,12
2,12.5,14
3,20.0,16
4,17.2,18
5,45.0,20
6,18.8,22
7,-126.0,24
8,27.5,26
9,22.2,28
10,30.3,30
11,35.7,32
12,14.0,34
13,33.9,36
14,44.5,38
15,39.1,40
16,43.8,42
17,39.5,44
18,246.0,46
19,46.8,48
20,53.0,50
21,44.5,52
22,84.0,54
23,56.8,56
24,55.5,58
25,65.2,60
26,61.0,62
27,68.3,64
28,48.0,66
29,70.0,68
30,66.2,70
31,73.5,72
32,73.5,74
33,-224.0,76
34,81.5,78
35,102.0,80
36,76.0,82
37,84.2,84
38,83.2,86
39,60.0,88
40,94.8,90
41,91.0,92
42,214.0,94
43,90.5,96
44,100.2,98
45,115.0,100
46,99.0,102
47,104.7,104
48,81.0,106
49,109.8,108
50,109.5,110`,
        yVar: 'Y_Observed',
        xVars: ['X'],
        toast: 'Cauchy Noise loaded (50 samples — watch OLS fail on heavy-tailed outliers!)'
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
        xVariables: [...example.xVars],
        dataSourceName: example.name || exampleKey
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

    // Store the file name for later use
    const fileName = file.name.replace(/\.[^/.]+$/, '');

    if (workbook.SheetNames.length > 1) {
        // Show sheet selector modal
        updateState({ pendingWorkbook: workbook, pendingFileName: fileName });
        showSheetSelector(workbook.SheetNames);
    } else {
        // Single sheet - import directly
        await importExcelSheet(workbook, workbook.SheetNames[0], fileName);
    }
}

/**
 * Import a specific sheet from an Excel workbook
 * @param {Object} workbook - XLSX workbook object
 * @param {string} sheetName - Sheet name to import
 * @param {string} fileName - Original file name (optional)
 */
export async function importExcelSheet(workbook, sheetName, fileName = null) {
    const worksheet = workbook.Sheets[sheetName];
    const csv = XLSX.utils.sheet_to_csv(worksheet);
    await parseCSVData(csv);

    // Set data source name
    if (fileName) {
        updateState({ dataSourceName: fileName });
    }

    showToast(`Imported sheet: ${sheetName}`, 'success');
}

/**
 * Handle CSV file import
 * @param {File} file - CSV file
 */
export async function handleCSVFile(file) {
    const text = await file.text();
    await parseCSVData(text);
    // Store the file name (without extension) as data source name
    const fileName = file.name.replace(/\.[^/.]+$/, '');
    updateState({ dataSourceName: fileName });
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
            const fileName = STATE.pendingFileName;
            importExcelSheet(STATE.pendingWorkbook, sheetName, fileName);
            modal.remove();
            updateState({ pendingWorkbook: null, pendingFileName: null });
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
