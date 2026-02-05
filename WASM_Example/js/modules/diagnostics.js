// ============================================================================
// diagnostics.js - All diagnostic test implementations
// ============================================================================

import { WasmRegression } from './core.js';
import { showToast } from './utils.js';

/**
 * Run all diagnostic tests for regression assumptions
 * @param {Array} yData - Response variable data
 * @param {Array} xData - Predictor variables data (array of arrays)
 * @param {string} rainbowMethod - Rainbow test method: 'r', 'python', or 'both' (default: 'r')
 * @param {string} whiteMethod - White test method: 'r', 'python', or 'both' (default: 'r')
 * @param {string|null} testType - Specific test category to run or null for all tests
 * @returns {Promise<Object>} Diagnostic test results organized by category
 */
export async function runDiagnostics(yData, xData, rainbowMethod = 'r', whiteMethod = 'r', testType = null) {
    if (!WasmRegression.isReady()) {
        throw new Error('WASM module is not ready for diagnostics');
    }

    const diagnostics = {
        linearity: [],
        heteroscedasticity: [],
        normality: [],
        autocorrelation: [],
        multicollinearity: [],
        influence: []
    };

    // Helper function to check if we should run a test category
    const shouldRunTest = (category) => !testType || testType === 'all' || testType === category;

    // ========================================================================
    // Linearity Tests
    // ========================================================================

    if (shouldRunTest('linearity')) {
        // Rainbow Test
        try {
            const rainbowJson = WasmRegression.rainbowTest(yData, xData, 0.5, rainbowMethod);
            const rainbowResult = JSON.parse(rainbowJson);
            if (!rainbowResult.error) {
                const method = rainbowMethod.toLowerCase();
                let testResult;

                if (method === 'both' && rainbowResult.r_result && rainbowResult.python_result) {
                    testResult = {
                        name: 'Rainbow Test (R & Python)',
                        shortName: 'Rainbow',
                        test_name: rainbowResult.test_name,
                        r_result: rainbowResult.r_result,
                        python_result: rainbowResult.python_result,
                        interpretation: rainbowResult.interpretation,
                        guidance: rainbowResult.guidance,
                        is_passed: rainbowResult.r_result.is_passed || rainbowResult.python_result.is_passed
                    };
                } else if (rainbowResult.r_result) {
                    testResult = {
                        name: 'Rainbow Test (R)',
                        shortName: 'Rainbow',
                        statistic: rainbowResult.r_result.statistic,
                        p_value: rainbowResult.r_result.p_value,
                        is_passed: rainbowResult.r_result.is_passed,
                        interpretation: rainbowResult.interpretation,
                        guidance: rainbowResult.guidance
                    };
                } else if (rainbowResult.python_result) {
                    testResult = {
                        name: 'Rainbow Test (Python)',
                        shortName: 'Rainbow',
                        statistic: rainbowResult.python_result.statistic,
                        p_value: rainbowResult.python_result.p_value,
                        is_passed: rainbowResult.python_result.is_passed,
                        interpretation: rainbowResult.interpretation,
                        guidance: rainbowResult.guidance
                    };
                } else {
                    testResult = rainbowResult;
                }

                diagnostics.linearity.push(testResult);
            }
        } catch (e) {
            console.warn('Rainbow test failed:', e);
        }

        // Harvey-Collier Test
        try {
            const hcJson = WasmRegression.harveyCollierTest(yData, xData);
            const hcResult = JSON.parse(hcJson);
            if (!hcResult.error) {
                diagnostics.linearity.push({
                    name: 'Harvey-Collier Test',
                    shortName: 'Harvey-Collier',
                    ...hcResult
                });
            }
        } catch (e) {
            console.warn('Harvey-Collier test failed:', e);
        }

        // RESET Test (NEW)
        try {
            const resetJson = WasmRegression.resetTest(yData, xData, [2, 3], 'fitted');
            const resetResult = JSON.parse(resetJson);
            if (!resetResult.error) {
                diagnostics.linearity.push({
                    name: 'RESET Test (Ramsey)',
                    shortName: 'RESET',
                    ...resetResult
                });
            }
        } catch (e) {
            console.warn('RESET test failed:', e);
        }
    }

    // ========================================================================
    // Heteroscedasticity Tests
    // ========================================================================

    if (shouldRunTest('heteroscedasticity')) {
        // White Test
        try {
            const whiteJson = WasmRegression.whiteTest(yData, xData, whiteMethod);
            const whiteResult = JSON.parse(whiteJson);
            if (!whiteResult.error) {
                const method = whiteMethod.toLowerCase();
                let testResult;

                if (method === 'both' && whiteResult.r_result && whiteResult.python_result) {
                    testResult = {
                        name: 'White Test (R & Python)',
                        shortName: 'White',
                        test_name: whiteResult.test_name,
                        r_result: whiteResult.r_result,
                        python_result: whiteResult.python_result,
                        interpretation: whiteResult.interpretation,
                        guidance: whiteResult.guidance,
                        is_passed: whiteResult.r_result.is_passed || whiteResult.python_result.is_passed
                    };
                } else if (whiteResult.r_result) {
                    testResult = {
                        name: 'White Test (R)',
                        shortName: 'White',
                        statistic: whiteResult.r_result.statistic,
                        p_value: whiteResult.r_result.p_value,
                        is_passed: whiteResult.r_result.is_passed,
                        interpretation: whiteResult.interpretation,
                        guidance: whiteResult.guidance
                    };
                } else if (whiteResult.python_result) {
                    testResult = {
                        name: 'White Test (Python)',
                        shortName: 'White',
                        statistic: whiteResult.python_result.statistic,
                        p_value: whiteResult.python_result.p_value,
                        is_passed: whiteResult.python_result.is_passed,
                        interpretation: whiteResult.interpretation,
                        guidance: whiteResult.guidance
                    };
                } else {
                    testResult = whiteResult;
                }

                diagnostics.heteroscedasticity.push(testResult);
            }
        } catch (e) {
            console.warn('White test failed:', e);
        }

        // Breusch-Pagan Test
        try {
            const bpJson = WasmRegression.breuschPaganTest(yData, xData);
            const bpResult = JSON.parse(bpJson);
            if (!bpResult.error) {
                diagnostics.heteroscedasticity.push({
                    name: 'Breusch-Pagan Test',
                    shortName: 'Breusch-Pagan',
                    ...bpResult
                });
            }
        } catch (e) {
            console.warn('Breusch-Pagan test failed:', e);
        }
    }

    // ========================================================================
    // Normality Tests
    // ========================================================================

    if (shouldRunTest('normality')) {
        // Jarque-Bera Test
        try {
            const jbJson = WasmRegression.jarqueBeraTest(yData, xData);
            const jbResult = JSON.parse(jbJson);
            if (!jbResult.error) {
                diagnostics.normality.push({
                    name: 'Jarque-Bera Test',
                    shortName: 'Jarque-Bera',
                    ...jbResult
                });
            }
        } catch (e) {
            console.warn('Jarque-Bera test failed:', e);
        }

        // Shapiro-Wilk Test
        try {
            const swJson = WasmRegression.shapiroWilkTest(yData, xData);
            const swResult = JSON.parse(swJson);
            if (!swResult.error) {
                diagnostics.normality.push({
                    name: 'Shapiro-Wilk Test',
                    shortName: 'Shapiro-Wilk',
                    ...swResult
                });
            }
        } catch (e) {
            console.warn('Shapiro-Wilk test failed:', e);
        }

        // Anderson-Darling Test
        try {
            const adJson = WasmRegression.andersonDarlingTest(yData, xData);
            const adResult = JSON.parse(adJson);
            if (!adResult.error) {
                diagnostics.normality.push({
                    name: 'Anderson-Darling Test',
                    shortName: 'Anderson-Darling',
                    ...adResult
                });
            }
        } catch (e) {
            console.warn('Anderson-Darling test failed:', e);
        }
    }

    // ========================================================================
    // Autocorrelation Tests
    // ========================================================================

    if (shouldRunTest('autocorrelation')) {
        // Durbin-Watson Test
        try {
            const dwJson = WasmRegression.durbinWatsonTest(yData, xData);
            const dwResult = JSON.parse(dwJson);
            if (!dwResult.error) {
                diagnostics.autocorrelation.push({
                    name: 'Durbin-Watson Test',
                    shortName: 'Durbin-Watson',
                    ...dwResult
                });
            }
        } catch (e) {
            console.warn('Durbin-Watson test failed:', e);
        }

        // Breusch-Godfrey Test (NEW)
        try {
            const bgJson = WasmRegression.breuschGodfreyTest(yData, xData, 2, 'chisq');
            const bgResult = JSON.parse(bgJson);
            if (!bgResult.error) {
                diagnostics.autocorrelation.push({
                    name: 'Breusch-Godfrey Test',
                    shortName: 'Breusch-Godfrey',
                    ...bgResult
                });
            }
        } catch (e) {
            console.warn('Breusch-Godfrey test failed:', e);
        }
    }

    // ========================================================================
    // Influence Tests
    // ========================================================================

    if (shouldRunTest('influence')) {
        // Cook's Distance
        try {
            const cdJson = WasmRegression.cooksDistanceTest(yData, xData);
            const cdResult = JSON.parse(cdJson);
            if (!cdResult.error) {
                // Compute max distance from the distances array
                const maxDistance = cdResult.distances && cdResult.distances.length > 0
                    ? Math.max(...cdResult.distances)
                    : undefined;
                diagnostics.influence.push({
                    name: "Cook's Distance",
                    shortName: "Cook's Distance",
                    ...cdResult,
                    statistic: maxDistance // Add max value as statistic for display
                });
            }
        } catch (e) {
            console.warn("Cook's Distance test failed:", e);
        }

        // DFBETAS Test
        try {
            const dfbetasJson = WasmRegression.dfbetasTest(yData, xData);
            const dfbetasResult = JSON.parse(dfbetasJson);
            if (!dfbetasResult.error) {
                // Coerce result to match expected format - pass if no influential observations
                const isInfluential = dfbetasResult.influential_observations &&
                    Object.keys(dfbetasResult.influential_observations).length > 0;
                // Compute max absolute DFBETAS value
                let maxDfbetas = 0;
                if (dfbetasResult.dfbetas && dfbetasResult.dfbetas.length > 0) {
                    dfbetasResult.dfbetas.forEach(obsDfbetas => {
                        obsDfbetas.forEach(val => {
                            if (Math.abs(val) > maxDfbetas) {
                                maxDfbetas = Math.abs(val);
                            }
                        });
                    });
                }
                diagnostics.influence.push({
                    name: 'DFBETAS',
                    shortName: 'DFBETAS',
                    ...dfbetasResult,
                    is_passed: !isInfluential,
                    statistic: maxDfbetas // Add max value as statistic for display
                });
            }
        } catch (e) {
            console.warn('DFBETAS test failed:', e);
        }

        // DFFITS Test
        try {
            const dffitsJson = WasmRegression.dffitsTest(yData, xData);
            const dffitsResult = JSON.parse(dffitsJson);
            if (!dffitsResult.error) {
                // Coerce result to match expected format - pass if no influential observations
                const isInfluential = dffitsResult.influential_observations &&
                    dffitsResult.influential_observations.length > 0;
                // Compute max absolute DFFITS value
                const maxDffits = dffitsResult.dffits && dffitsResult.dffits.length > 0
                    ? Math.max(...dffitsResult.dffits.map(v => Math.abs(v)))
                    : 0;
                diagnostics.influence.push({
                    name: 'DFFITS',
                    shortName: 'DFFITS',
                    ...dffitsResult,
                    is_passed: !isInfluential,
                    statistic: maxDffits // Add max value as statistic for display
                });
            }
        } catch (e) {
            console.warn('DFFITS test failed:', e);
        }
    }

    // ========================================================================
    // Multicollinearity Tests
    // ========================================================================

    if (shouldRunTest('multicollinearity')) {
        // VIF Test (requires at least 2 predictors)
        if (xData.length >= 2) {
            try {
                const vifJson = WasmRegression.vifTest(yData, xData);
                const vifResult = JSON.parse(vifJson);
                if (!vifResult.error) {
                    const testResult = {
                        name: 'Variance Inflation Factor (VIF)',
                        shortName: 'VIF',
                        max_vif: vifResult.max_vif,
                        vif_results: vifResult.vif_results,
                        interpretation: vifResult.interpretation,
                        guidance: vifResult.guidance,
                        is_passed: vifResult.max_vif <= 10
                    };
                    diagnostics.multicollinearity.push(testResult);
                }
            } catch (e) {
                console.warn('VIF test failed:', e);
            }
        }
    }

    return diagnostics;
}

/**
 * Get a human-readable interpretation of a test result
 * @param {Object} testResult - Test result object
 * @returns {string} Interpretation text
 */
export function interpretTestResult(testResult) {
    if (testResult.p_value === undefined) {
        return 'No p-value available.';
    }

    const alpha = 0.05;
    const passed = testResult.is_passed !== undefined ? testResult.is_passed : testResult.p_value > alpha;

    if (passed) {
        return `p-value = ${testResult.p_value.toFixed(4)} > ${alpha}. No evidence to reject null hypothesis.`;
    } else {
        return `p-value = ${testResult.p_value.toFixed(4)} ≤ ${alpha}. Null hypothesis rejected.`;
    }
}

/**
 * Check if a test passed (p-value > 0.05)
 * @param {Object} testResult - Test result object
 * @returns {boolean} True if test passed
 */
export function testPassed(testResult) {
    const alpha = 0.05;
    return testResult.is_passed !== undefined ? testResult.is_passed : testResult.p_value > alpha;
}

/**
 * Get test category for a given test name
 * @param {string} testName - Name of the test
 * @returns {string} Category name
 */
export function getTestCategory(testName) {
    const categoryMap = {
        'Rainbow': 'linearity',
        'Harvey-Collier': 'linearity',
        'RESET': 'linearity',
        'White': 'heteroscedasticity',
        'Breusch-Pagan': 'heteroscedasticity',
        'Jarque-Bera': 'normality',
        'Shapiro-Wilk': 'normality',
        'Anderson-Darling': 'normality',
        'Durbin-Watson': 'autocorrelation',
        'Breusch-Godfrey': 'autocorrelation',
        "Cook's Distance": 'influence',
        'DFBETAS': 'influence',
        'DFFITS': 'influence',
        'VIF': 'multicollinearity'
    };

    for (const [key, category] of Object.entries(categoryMap)) {
        if (testName.includes(key)) {
            return category;
        }
    }

    return 'other';
}

/**
 * Get available diagnostic test categories
 * @returns {Array<string>} List of category names
 */
export function getDiagnosticCategories() {
    return ['linearity', 'heteroscedasticity', 'normality', 'autocorrelation', 'multicollinearity', 'influence'];
}
