# ============================================================================
# Linear Regression Tests - R Reference Implementation
# ============================================================================
# This script runs the same test cases as the Rust WASM implementation
# to verify correctness. Output can be compared for validation.
#
# IMPORTANT: This script uses NATIVE R library functions for all tests.
# Manual implementations are NOT used as they would not validate correctness.
#
# INSTALL REQUIRED PACKAGES:
#   Run R as administrator and execute:
#   install.packages(c("lmtest", "car", "skedastic", "tseries", "nortest"))
#
# Or install to user library (no admin required):
#   install.packages(c("lmtest", "car", "skedastic", "tseries", "nortest"),
#                   lib = "~/R/win-library/4.4")
#   (Adjust the path for your R version)
# ============================================================================

cat("==============================================================\n")
cat("Linear Regression Tests - R Reference Implementation\n")
cat("==============================================================\n")

# Add user library to path (helps find packages installed via RGUI)
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.4")
if (dir.exists(user_lib)) {
    .libPaths(c(user_lib, .libPaths()))
}

# Safely load libraries - FAIL LOUDLY if missing
# We REQUIRE these packages for established, tested implementations
required_packages <- c("lmtest", "car", "skedastic", "tseries", "nortest")
for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        stop(sprintf("CRITICAL ERROR: Package '%s' is required for validation but not found. 
                     Install it with: install.packages('%s')", pkg, pkg))
    }
}

# ============================================================================ 
# Test Result Storage 
# ============================================================================ 
test_results <- list()

# ============================================================================
# Helper: White Test using NATIVE library function
# ============================================================================
# Uses skedastic::white() - the canonical R implementation
# ============================================================================
perform_white_test <- function(model, data) {
    # Use native library function - CORRECT approach for validation
    # No fallback allowed.
    white_result <- white(model)

    # Extract statistic and p-value
    list(
        statistic = as.numeric(white_result$statistic),
        p_value = as.numeric(white_result$p.value),
        df = white_result$parameter,
        passed = as.numeric(white_result$p.value) > 0.05,
        source = "skedastic::white"
    )
}

# ============================================================================
# Helper: Jarque-Bera Test using NATIVE library function
# ============================================================================
# Uses tseries::jarque.bera.test() - the canonical R implementation
# ============================================================================
perform_jarque_bera_test <- function(model) {
    # Use native library function - CORRECT approach for validation
    # No fallback allowed.
    jb_result <- jarque.bera.test(residuals(model))
    list(
        statistic = as.numeric(jb_result$statistic),
        p_value = as.numeric(jb_result$p.value),
        passed = as.numeric(jb_result$p.value) > 0.05,
        source = "tseries::jarque.bera.test"
    )
}


# Helper function to capture test results
capture_test <- function(name, model, data) {
    # Calculate MSE first
    mse_val <- sum(residuals(model)^2) / model$df.residual
    
    result <- list(
        test_name = name,
        n = nrow(data),
        k = ncol(model$model) - 1,  # number of predictors (excluding intercept)
        df = model$df.residual,

        # Coefficients
        coefficients = as.vector(coef(model)),
        std_errors = as.vector(summary(model)$coefficients[, "Std. Error"]),
        t_stats = as.vector(summary(model)$coefficients[, "t value"]),
        p_values = as.vector(summary(model)$coefficients[, "Pr(>|t|)"]),

        # Model fit
        r_squared = summary(model)$r.squared,
        adj_r_squared = summary(model)$adj.r.squared,
        f_statistic = summary(model)$fstatistic[1],
        f_p_value = pf(summary(model)$fstatistic[1],
                       summary(model)$fstatistic[2],
                       summary(model)$fstatistic[3],
                       lower.tail = FALSE),

        # MSE and Std Error
        mse = mse_val,
        std_error = sqrt(mse_val),

        # Residuals
        residuals = as.vector(residuals(model)),
        standardized_residuals = as.vector(rstandard(model)),

        # VIF (only for multiple regression)
        vif = NULL
    )

    # 95% Confidence Intervals
    ci <- confint(model, level = 0.95)
    result$conf_int_lower <- as.vector(ci[, 1])
    result$conf_int_upper <- as.vector(ci[, 2])

    # VIF calculation (requires at least 2 predictors)
    if (result$k >= 2) {
        vif_result <- vif(model)
        result$vif <- data.frame(
            variable = names(vif_result),
            vif = as.numeric(vif_result),
            rsquared = 1 - 1 / as.numeric(vif_result)
        )
    }

    # Linearity Tests - MANDATORY native library calls
    result$linearity_tests <- list()

    # Rainbow Test
    rainbow <- raintest(model, fraction = 0.5, order.by = NULL)
    result$linearity_tests$rainbow <- list(
        statistic = rainbow$statistic,
        p_value = rainbow$p.value,
        df1 = rainbow$parameter[1],
        df2 = rainbow$parameter[2],
        passed = rainbow$p.value > 0.05
    )

    # Harvey-Collier Test
    hc <- harvtest(model, order.by = NULL)
    result$linearity_tests$harvey_collier <- list(
        statistic = hc$statistic,
        p_value = hc$p.value,
        df = hc$parameter,
        passed = hc$p.value > 0.05
    )

    # Breusch-Pagan Test
    bp <- bptest(model, studentize = TRUE)
    result$linearity_tests$breusch_pagan <- list(
        statistic = bp$statistic,
        p_value = bp$p.value,
        df = bp$parameter,
        passed = bp$p.value > 0.05
    )
    
    # White Test
    white <- perform_white_test(model, data)
    result$linearity_tests$white <- white
    
    # Jarque-Bera Test
    jb <- perform_jarque_bera_test(model)
    result$linearity_tests$jarque_bera <- jb
    
    # Durbin-Watson Test
    dw <- dwtest(model)
    result$linearity_tests$durbin_watson <- list(
        statistic = dw$statistic,
        p_value = dw$p.value,
        passed = dw$p.value > 0.05 # Rough check
    )

    test_results[[name]] <<- result

    return(result)
}

# Helper to print test results
print_test_results <- function(result) {
    cat(sprintf("\n[%s]\n", result$test_name))
    cat(sprintf("  Observations (n): %d\n", result$n))
    cat(sprintf("  Predictors (k): %d\n", result$k))
    cat(sprintf("  Degrees of Freedom: %d\n", result$df))
    cat(sprintf("  R^2: %.6f\n", result$r_squared))
    cat(sprintf("  Adjusted R^2: %.6f\n", result$adj_r_squared))
    cat(sprintf("  F-statistic: %.4f (p = %.6f)\n", result$f_statistic, result$f_p_value))
    cat(sprintf("  MSE: %.6f\n", result$mse))

    # Diagnostic Tests
    cat("\n  Diagnostic Tests (Native Libraries Only):\n")

    cat(sprintf("    Rainbow Test: F = %.4f, p = %.4f [%s] (lmtest::raintest)\n",
        result$linearity_tests$rainbow$statistic,
        result$linearity_tests$rainbow$p_value,
        ifelse(result$linearity_tests$rainbow$passed, "PASS", "FAIL")))

    cat(sprintf("    Harvey-Collier Test: t = %.4f, p = %.4f [%s] (lmtest::harvtest)\n",
        result$linearity_tests$harvey_collier$statistic,
        result$linearity_tests$harvey_collier$p_value,
        ifelse(result$linearity_tests$harvey_collier$passed, "PASS", "FAIL")))

    cat(sprintf("    Breusch-Pagan Test: LM = %.4f, p = %.4f [%s] (lmtest::bptest)\n",
        result$linearity_tests$breusch_pagan$statistic,
        result$linearity_tests$breusch_pagan$p_value,
        ifelse(result$linearity_tests$breusch_pagan$passed, "PASS", "FAIL")))

    source_label <- if (!is.null(result$linearity_tests$white$source)) result$linearity_tests$white$source else "unknown"
    cat(sprintf("    White Test: LM = %.4f, p = %.4f [%s] (%s)\n",
        result$linearity_tests$white$statistic,
        result$linearity_tests$white$p_value,
        ifelse(result$linearity_tests$white$passed, "PASS", "FAIL"),
        source_label))

    source_label <- if (!is.null(result$linearity_tests$jarque_bera$source)) result$linearity_tests$jarque_bera$source else "unknown"
    cat(sprintf("    Jarque-Bera Test: JB = %.4f, p = %.4f [%s] (%s)\n",
        result$linearity_tests$jarque_bera$statistic,
        result$linearity_tests$jarque_bera$p_value,
        ifelse(result$linearity_tests$jarque_bera$passed, "PASS", "FAIL"),
        source_label))

    cat(sprintf("    Durbin-Watson Test: d = %.4f (lmtest::dwtest)\n",
        result$linearity_tests$durbin_watson$statistic))
}

# ============================================================================ 
# TEST 1: Simple Linear Regression (Study Hours vs Test Scores)
# ============================================================================ 
cat("\n--- Test 1: Simple Linear Regression ---\n")

df_simple <- data.frame(
    Study_Hours = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     2.5, 4.5, 6.5, 1.5, 7.5, 3.5, 5.5, 8.5, 9.5, 0.5),
    Test_Score = c(52, 58, 62, 68, 75, 79, 85, 88, 92, 95,
                  60, 72, 82, 55, 86, 66, 77, 90, 94, 48)
)

model_simple <- lm(Test_Score ~ Study_Hours, data = df_simple)
result_simple <- capture_test("simple_regression", model_simple, df_simple)
print_test_results(result_simple)

# ============================================================================ 
# TEST 2: Multiple Linear Regression (Housing Prices)
# ============================================================================ 
cat("\n--- Test 2: Multiple Linear Regression ---\n")

df_housing <- data.frame(
    Price = c(245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
              445.8, 167.9, 367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9,
              223.4, 312.5, 156.8, 423.7, 267.9),
    Square_Feet = c(1200, 1800, 950, 2400, 1450, 2000, 1100, 2800, 1350, 1650,
                    2200, 900, 1950, 1500, 1050, 2600, 1300, 1850, 1000, 2100,
                    1250, 1700, 850, 2350, 1400),
    Bedrooms = c(3, 4, 2, 4, 3, 4, 2, 5, 3, 3,
                 4, 2, 4, 3, 2, 5, 3, 4, 2, 4,
                 3, 3, 2, 4, 3),
    Age = c(15, 10, 25, 5, 8, 12, 20, 2, 18, 7,
           3, 30, 6, 14, 22, 1, 16, 9, 28, 4,
           19, 11, 35, 3, 13)
)

model_housing <- lm(Price ~ Square_Feet + Bedrooms + Age, data = df_housing)
result_housing <- capture_test("housing_regression", model_housing, df_housing)
print_test_results(result_housing)

# ============================================================================ 
# TEST 3: Singular Matrix (Perfect Multicollinearity)
# ============================================================================ 
cat("\n--- Test 3: Singular Matrix (Should Fail) ---\n")

df_singular <- data.frame(
    Y = c(10, 15, 20, 25, 30, 35, 40, 45, 50, 55),
    X1 = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    X2 = c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    X3 = c(3, 5, 7, 9, 11, 13, 15, 17, 19, 21)  # X3 = X1 + X2 exactly
)

# This should produce a singular matrix error
tryCatch({
    model_singular <- lm(Y ~ X1 + X2 + X3, data = df_singular)
    cat("  ERROR: Model should have failed due to singular matrix!\n")
}, error = function(e) {
    cat(sprintf("  Expected error: %s\n", conditionMessage(e)))
})

# ============================================================================ 
# TEST 4: Messy Data (Store Tier Clustering - Poor Linearity)
# ============================================================================ 
cat("\n--- Test 4: Messy Data (Clustering, Non-linear) ---\n")

df_messy <- data.frame(
    Store_Tier = rep(c("Budget", "Mid_range", "Luxury"), each = 10),
    Marketing_Spend = c(
        # Budget tier
        1000, 1200, 800, 1500, 900, 1100, 1300, 700, 1400, 850,
        # Mid_range tier
        3000, 3500, 2800, 4000, 3200, 3800, 2500, 4200, 2900, 3600,
        # Luxury tier
        8000, 9000, 7500, 10000, 8500, 9500, 7000, 10500, 7800, 9200
    ),
    Sales = c(
        # Budget tier
        15000, 16500, 14000, 18000, 15500, 16200, 17000, 13500, 17500, 14800,
        # Mid_range tier
        35000, 38000, 33000, 42000, 36000, 40000, 31000, 43500, 34000, 39000,
        # Luxury tier
        55000, 58000, 53000, 62000, 56000, 60000, 51000, 64000, 54000, 59000
    )
)

model_messy <- lm(Sales ~ Marketing_Spend, data = df_messy)
result_messy <- capture_test("messy_clustering", model_messy, df_messy)
print_test_results(result_messy)

# ============================================================================ 
# Generate JSON Output for Rust Validation
# ============================================================================ 
# Manual JSON construction to avoid external dependencies like jsonlite

format_vector <- function(vec) {
    if (is.null(vec) || length(vec) == 0) return("[]")
    return(sprintf("[%s]", paste(vec, collapse = ", ")))
}

format_string_vector <- function(vec) {
    if (is.null(vec) || length(vec) == 0) return("[]")
    s <- sprintf("\"%s\"", vec)
    p <- paste(s, collapse = ", ")
    return(sprintf("[%s]", p))
}

# Focus on Housing Regression for validation
res <- result_housing

# Helper to format Rainbow result
rainbow_json <- "null"
if (!is.null(res$linearity_tests$rainbow) && is.null(res$linearity_tests$rainbow$error)) {
    rainbow_json = sprintf('{"statistic": %.10f, "p_value": %.10f, "passed": %s}',
        res$linearity_tests$rainbow$statistic,
        res$linearity_tests$rainbow$p_value,
        ifelse(res$linearity_tests$rainbow$passed, "true", "false"))
}

# Helper to format Harvey-Collier result
harvey_collier_json <- "null"
if (!is.null(res$linearity_tests$harvey_collier) && is.null(res$linearity_tests$harvey_collier$error)) {
    harvey_collier_json = sprintf('{"statistic": %.10f, "p_value": %.10f, "passed": %s}',
        res$linearity_tests$harvey_collier$statistic,
        res$linearity_tests$harvey_collier$p_value,
        ifelse(res$linearity_tests$harvey_collier$passed, "true", "false"))
}

# Helper to format Breusch-Pagan result
bp_json <- "null"
if (!is.null(res$linearity_tests$breusch_pagan) && is.null(res$linearity_tests$breusch_pagan$error)) {
    bp_json = sprintf('{"statistic": %.10f, "p_value": %.10f, "passed": %s}',
        res$linearity_tests$breusch_pagan$statistic,
        res$linearity_tests$breusch_pagan$p_value,
        ifelse(res$linearity_tests$breusch_pagan$passed, "true", "false"))
}

# Helper to format White Test result
white_json <- "null"
if (!is.null(res$linearity_tests$white) && is.null(res$linearity_tests$white$error)) {
    white_json = sprintf('{"statistic": %.10f, "p_value": %.10f, "passed": %s}',
        res$linearity_tests$white$statistic,
        res$linearity_tests$white$p_value,
        ifelse(res$linearity_tests$white$passed, "true", "false"))
}

# Helper to format Jarque-Bera result
jb_json <- "null"
if (!is.null(res$linearity_tests$jarque_bera) && is.null(res$linearity_tests$jarque_bera$error)) {
    jb_json = sprintf('{"statistic": %.10f, "p_value": %.10f, "passed": %s}',
        res$linearity_tests$jarque_bera$statistic,
        res$linearity_tests$jarque_bera$p_value,
        ifelse(res$linearity_tests$jarque_bera$passed, "true", "false"))
}

# Helper to format Durbin-Watson result
dw_json <- "null"
if (!is.null(res$linearity_tests$durbin_watson) && is.null(res$linearity_tests$durbin_watson$error)) {
    dw_json = sprintf('{"statistic": %.10f, "p_value": %.10f, "passed": %s}',
        res$linearity_tests$durbin_watson$statistic,
        # p-value might be NaN in Rust, so we won't strictly validate it there, but we output it here
        res$linearity_tests$durbin_watson$p_value,
        ifelse(res$linearity_tests$durbin_watson$passed, "true", "false"))
}

json_output <- sprintf('{
  "housing_regression": {
    "coefficients": %s,
    "std_errors": %s,
    "t_stats": %s,
    "p_values": %s,
    "r_squared": %.10f,
    "adj_r_squared": %.10f,
    "f_statistic": %.10f,
    "f_p_value": %.20f,
    "mse": %.10f,
    "std_error": %.10f,
    "conf_int_lower": %s,
    "conf_int_upper": %s,
    "residuals": %s,
    "standardized_residuals": %s,
    "leverage": [],
    "vif": %s,
    "rainbow": %s,
    "harvey_collier": %s,
    "breusch_pagan": %s,
    "white": %s,
    "jarque_bera": %s,
    "durbin_watson": %s,
    "n": %d,
    "k": %d,
    "df": %d
  }
}',
    format_vector(res$coefficients),
    format_vector(res$std_errors),
    format_vector(res$t_stats),
    format_vector(res$p_values),
    res$r_squared,
    res$adj_r_squared,
    res$f_statistic,
    res$f_p_value,
    res$mse,
    res$std_error,
    format_vector(res$conf_int_lower),
    format_vector(res$conf_int_upper),
    format_vector(res$residuals),
    format_vector(res$standardized_residuals),
    if (!is.null(res$vif)) {
        # Construct VIF array of objects
        vif_entries <- c()
        for (i in 1:nrow(res$vif)) {
            vif_entries <- c(vif_entries, sprintf('{"variable": "%s", "vif": %.10f, "rsquared": %.10f}',
                res$vif$variable[i], res$vif$vif[i], res$vif$rsquared[i]))
        }
        sprintf("[%s]", paste(vif_entries, collapse = ", "))
    } else { "[]" },
    rainbow_json,
    harvey_collier_json,
    bp_json,
    white_json,
    jb_json,
    dw_json,
    res$n,
    res$k,
    res$df
)

# Write to file - ALWAYS OVERWRITE to ensure fresh validation data
# Output to verification/results/r/
script_dir <- dirname(sys.frame(1)$filename)
output_path <- file.path(script_dir, "..", "..", "..", "results", "r", "R_results.json")
writeLines(json_output, output_path)

cat(sprintf("\n[SUCCESS] Wrote fresh validation data to: %s\n", output_path))

# ============================================================================
# R Library Function Summary
# ============================================================================
cat("\n==============================================================\n")
cat("R Library Functions Used for Validation\n")
cat("==============================================================\n")

cat("\nCore Regression:\n")
cat("  lm()           - Base R (Ordinary Least Squares)\n")
cat("  summary()      - Base R (Model statistics)\n")
cat("  confint()      - Base R (Confidence intervals)\n")
cat("  rstandard()    - Base R (Standardized residuals)\n")

cat("\nDiagnostic Tests (MANDATORY):\n")
cat("  raintest()     - lmtest::raintest (Rainbow test for linearity)\n")
cat("  harvtest()     - lmtest::harvtest (Harvey-Collier test for linearity)\n")
cat("  bptest()       - lmtest::bptest (Breusch-Pagan test for heteroscedasticity)\n")
cat("  dwtest()       - lmtest::dwtest (Durbin-Watson test for autocorrelation)\n")
cat("  white()        - skedastic::white (White test for heteroscedasticity)\n")
cat("  jarque.bera.test() - tseries::jarque.bera.test (Jarque-Bera test for normality)\n")

cat("\nMulticollinearity (MANDATORY):\n")
cat("  vif()          - car::vif (Variance Inflation Factor)\n")

cat("\nDistributions:\n")
cat("  pf()           - Base R (F-distribution CDF)\n")
cat("  pt()           - Base R (Student's t-distribution CDF)\n")
cat("  pchisq()       - Base R (Chi-squared distribution)\n")
cat("  qt()           - Base R (Quantile function for t-distribution)\n")
cat("  qnorm()        - Base R (Quantile function for normal)\n")

cat("\n==============================================================\n")
