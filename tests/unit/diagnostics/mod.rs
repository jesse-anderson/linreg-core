// ============================================================================
// Diagnostic Tests Unit Tests Module
// ============================================================================
//
// Comprehensive unit tests for diagnostic test algorithms.
//
// Organization:
// - normality: Tests for normality assumptions (Jarque-Bera, Shapiro-Wilk, Anderson-Darling)
// - heteroscedasticity: Tests for heteroscedasticity detection (Breusch-Pagan, White)
// - autocorrelation: Tests for autocorrelation detection (Durbin-Watson, Breusch-Godfrey)
// - linearity: Tests for linearity assumptions (Rainbow, Harvey-Collier, RESET)
// - influence: Tests for influential observations (Cook's Distance, DFBETAS, DFFITS)

mod autocorrelation;
mod heteroscedasticity;
mod influence;
mod linearity;
mod normality;
