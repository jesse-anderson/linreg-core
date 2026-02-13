// ============================================================================
// utils.js - Shared utility functions
// ============================================================================

/**
 * Escapes HTML special characters to prevent XSS
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
export function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Formats a p-value for display with appropriate notation
 * @param {number} pValue - The p-value to format
 * @returns {string} Formatted p-value string
 */
export function formatPValue(pValue) {
    if (pValue === null || pValue === undefined) {
        return 'N/A';
    }
    if (pValue < 0.0001) {
        return `< 0.0001`;
    }
    return pValue.toFixed(4);
}

/**
 * Shows a toast notification message
 * @param {string} message - Message to display
 * @param {string} type - Type: 'success', 'error', 'warning', 'info'
 * @param {number} duration - Duration in milliseconds (default: 3000)
 */
export function showToast(message, type = 'info', duration = 3000) {
    // Remove existing toast if present
    const existingToast = document.querySelector('.toast-notification');
    if (existingToast) {
        existingToast.remove();
    }

    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;

    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6'
    };

    const icons = {
        success: '✓',
        error: '✕',
        warning: '⚠',
        info: 'ℹ'
    };

    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10000;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        animation: slideIn 0.3s ease-out;
    `;

    toast.innerHTML = `<span style="font-weight: bold;">${icons[type] || icons.info}</span> ${escapeHtml(message)}`;

    // Add animation keyframes if not present
    if (!document.getElementById('toast-animations')) {
        const style = document.createElement('style');
        style.id = 'toast-animations';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Converts hex color to RGBA
 * @param {string} hex - Hex color code (with or without #)
 * @param {number} alpha - Alpha value (0-1)
 * @returns {string} RGBA color string
 */
export function hexToRgba(hex, alpha = 1) {
    let c = hex.replace('#', '');

    // Handle 3-digit hex
    if (c.length === 3) {
        c = c.split('').map(char => char + char).join('');
    }

    const r = parseInt(c.substring(0, 2), 16);
    const g = parseInt(c.substring(2, 4), 16);
    const b = parseInt(c.substring(4, 6), 16);

    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Gets the current theme ('light' or 'dark')
 * @returns {string} Current theme
 */
export function getCurrentTheme() {
    return document.body.getAttribute('data-theme') || 'light';
}

/**
 * Chart colors for different themes
 */
export const ChartColors = {
    light: {
        primary: '#3b82f6',
        secondary: '#10b981',
        accent: '#f59e0b',
        danger: '#ef4444',
        grid: '#e5e7eb',
        text: '#374151',
        background: '#ffffff'
    },
    dark: {
        primary: '#60a5fa',
        secondary: '#34d399',
        accent: '#fbbf24',
        danger: '#f87171',
        grid: '#374151',
        text: '#e5e7eb',
        background: '#1f2937'
    }
};

/**
 * Gets chart colors for current theme
 * @returns {Object} Chart colors object
 */
export function getChartColors() {
    const theme = getCurrentTheme();
    return ChartColors[theme];
}

// ============================================================================
// STATISTICAL UTILITIES
// ============================================================================

/**
 * Statistical functions utility
 */
export const Stats = {
    /**
     * Calculate mean of array
     */
    mean: (arr) => arr.reduce((a, b) => a + b, 0) / arr.length,

    /**
     * Calculate variance of array
     */
    variance: (arr, ddof = 1) => {
        const mu = Stats.mean(arr);
        return arr.reduce((sum, x) => sum + (x - mu) ** 2, 0) / (arr.length - ddof);
    },

    /**
     * Calculate standard deviation of array
     */
    std: (arr, ddof = 1) => Math.sqrt(Stats.variance(arr, ddof)),

    /**
     * Calculate correlation coefficient
     */
    correlation: (x, y) => {
        const n = x.length;
        const meanX = Stats.mean(x);
        const meanY = Stats.mean(y);

        let numerator = 0;
        let sumSqX = 0;
        let sumSqY = 0;

        for (let i = 0; i < n; i++) {
            const dx = x[i] - meanX;
            const dy = y[i] - meanY;
            numerator += dx * dy;
            sumSqX += dx * dx;
            sumSqY += dy * dy;
        }

        return numerator / Math.sqrt(sumSqX * sumSqY);
    }
};

// ============================================================================
// CONSTANTS
// ============================================================================

export const ALLOWED_DOMAINS = [
    'jesse-anderson.net',
    'tools.jesse-anderson.net',
    'linear-regression.jesse-anderson.net',
    'localhost',
    '127.0.0.1'
];

// ============================================================================
// THEME MANAGEMENT
// ============================================================================

export const ThemeManager = {
    init() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(savedTheme);

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (localStorage.getItem('theme') === 'system') {
                this.setTheme(e.matches ? 'dark' : 'light');
            }
        });
    },

    setTheme(theme) {
        // Use data-theme attribute to match shared.css
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);

        // Update charts if they exist and we have regression results
        if (window.updateCharts && STATE.regressionResults) {
            window.updateCharts(STATE.regressionResults);
        }
    },

    toggleTheme() {
        const current = getCurrentTheme();
        this.setTheme(current === 'dark' ? 'light' : 'dark');
    },

    getTheme() {
        return getCurrentTheme();
    }
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

/**
 * Global application state
 */
export const STATE = {
    rawData: [],           // Array of row objects
    headers: [],           // Column names
    numericColumns: [],    // Names of numeric columns
    yVariable: null,       // Selected Y variable
    xVariables: [],        // Selected X variables (array)
    regressionResults: null,
    diagnostics: null,     // Diagnostic test results
    charts: {
        main: null,
        residuals: null,
        qq: null,
        leverage: null,
        cv: null
    },
    cvResults: null,
    pendingWorkbook: null,  // For sheet selector modal
    pendingFileName: null,  // For tracking file name during import
    residualsData: null,    // Residuals table data for sorting/filtering
    residualsSortColumn: null,
    residualsSortDirection: 'desc',
    selectedObservation: null,
    savedModels: null,      // Array of saved models for comparison
    dataSourceName: null    // Name of the data source (file name or example name)
};

/**
 * Updates state and optionally triggers callbacks
 * @param {Object} updates - State updates
 * @param {Function} callback - Optional callback after update
 */
export function updateState(updates, callback) {
    Object.assign(STATE, updates);
    if (callback) {
        callback(STATE);
    }
}
