# ============================================================================
# Extended Datasets Validation - R Reference Implementation
# ============================================================================
# This script validates the Rust OLS implementation against R using synthetic
# and real-world datasets designed to test edge cases and stress conditions.
#
# Output: verification/datasets/results/r/*.json
# ============================================================================

cat("==============================================================\n")
cat("Extended Datasets Validation - R Reference Implementation\n")
cat("==============================================================\n")

# Add user library to path (helps find packages installed via RGUI)
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.4")
if (dir.exists(user_lib)) {
    .libPaths(c(user_lib, .libPaths()))
}

# Load required packages
required_packages <- c("lmtest", "car", "tseries")
for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        stop(sprintf("Package '%s' is required. Install with: install.packages('%s')", pkg, pkg))
    }
}

# Set up output directory (relative to script location)
script_dir <- dirname(sys.frame(1)$filename)
output_dir <- file.path(script_dir, "..", "..", "..", "results", "r")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Set path to datasets (relative to script location)
datasets_dir <- file.path(script_dir, "..", "..", "..", "datasets", "csv")

# ============================================================================
# Helper Functions
# ============================================================================

format_vector <- function(vec) {
    if (is.null(vec) || length(vec) == 0) return("[]")
    return(sprintf("[%s]", paste(vec, collapse = ", ")))
}

run_regression <- function(dataset_name, df, y_var, x_vars = NULL) {
    cat(sprintf("\n=== %s ===\n", dataset_name))

    # Build formula
    if (is.null(x_vars)) {
        x_vars <- setdiff(names(df), y_var)
    }
    formula_str <- paste(y_var, "~", paste(x_vars, collapse = " + "))
    formula <- as.formula(formula_str)

    # Fit model (with error handling)
    model <- tryCatch(lm(formula, data = df), error = function(e) {
        cat(sprintf("  Error: %s\n", conditionMessage(e)))
        return(NULL)
    })

    if (is.null(model)) return(NULL)

    # Extract results
    n <- nrow(df)
    k <- length(x_vars)
    df_residual <- model$df.residual

    coefficients <- as.vector(coef(model))
    std_errors <- as.vector(summary(model)$coefficients[, "Std. Error"])
    t_stats <- as.vector(summary(model)$coefficients[, "t value"])
    p_values <- as.vector(summary(model)$coefficients[, "Pr(>|t|)"])

    r_squared <- summary(model)$r.squared
    adj_r_squared <- summary(model)$adj.r.squared

    # Handle cases where F-statistic might be NA (e.g., perfect fit)
    f_stat <- tryCatch(summary(model)$fstatistic, error = function(e) NULL)
    if (!is.null(f_stat)) {
        f_statistic <- f_stat[1]
        f_p_value <- pf(f_statistic[1], f_statistic[2], f_statistic[3], lower.tail = FALSE)
    } else {
        f_statistic <- NA
        f_p_value <- NA
    }

    # Confidence intervals
    ci <- confint(model, level = 0.95)
    ci_lower <- as.vector(ci[, 1])
    ci_upper <- as.vector(ci[, 2])

    # VIF for multiple regression
    vif_results <- NULL
    if (k >= 2) {
        vif_values <- tryCatch(car::vif(model), error = function(e) NULL)
        if (!is.null(vif_values)) {
            vif_results <- list(
                variables = names(vif_values),
                vif = as.numeric(vif_values)
            )
        }
    }

    # Output JSON
    json_output <- sprintf('{
  "dataset": "%s",
  "n": %d,
  "k": %d,
  "df_residual": %d,
  "coefficients": %s,
  "std_errors": %s,
  "t_stats": %s,
  "p_values": %s,
  "r_squared": %.15f,
  "adj_r_squared": %.15f,
  "f_statistic": %s,
  "f_p_value": %s,
  "ci_lower": %s,
  "ci_upper": %s,
  "vif": %s
}',
        dataset_name,
        n, k, df_residual,
        format_vector(coefficients),
        format_vector(std_errors),
        format_vector(t_stats),
        format_vector(p_values),
        r_squared, adj_r_squared,
        ifelse(is.na(f_statistic), "null", as.character(f_statistic)),
        ifelse(is.na(f_p_value), "null", as.character(f_p_value)),
        format_vector(ci_lower),
        format_vector(ci_upper),
        if (!is.null(vif_results)) {
            sprintf('{"variables": ["%s"], "vif": %s}',
                    paste(vif_results$variables, collapse = '", "'),
                    paste(vif_results$vif, collapse = ", "))
        } else {
            "null"
        }
    )

    # Write to file
    output_file <- file.path(output_dir, paste0(gsub(" ", "_", tolower(dataset_name)), ".json"))
    writeLines(json_output, output_file)
    cat(sprintf("  Wrote: %s\n", basename(output_file)))

    # Print summary
    cat(sprintf("    n = %d, k = %d\n", n, k))
    cat(sprintf("    R² = %.6f, Adj R² = %.6f\n", r_squared, adj_r_squared))
    cat(sprintf("    F(%d, %d) = %.4f, p = %.6f\n", k, df_residual, f_statistic, f_p_value))
    if (!is.null(vif_results)) {
        cat(sprintf("    VIF: %s\n", paste(round(vif_results$vif, 2), collapse = ", ")))
    }

    invisible(list(
        dataset = dataset_name,
        n = n,
        k = k,
        r_squared = r_squared,
        vif = if (!is.null(vif_results)) vif_results$vif else NULL
    ))
}

# ============================================================================
# Load and Test Datasets
# ============================================================================

# --- Synthetic Datasets ---

# 1. Simple Linear
synthetic_simple <- read.csv(file.path(datasets_dir, "synthetic_simple_linear.csv"))
run_regression("Synthetic Simple Linear", synthetic_simple, "y", c("x"))

# 2. Multiple Regression
synthetic_multiple <- read.csv(file.path(datasets_dir, "synthetic_multiple.csv"))
run_regression("Synthetic Multiple", synthetic_multiple, "y", c("x1", "x2", "x3"))

# 3. Collinear (should fail or show high VIF)
synthetic_collinear <- read.csv(file.path(datasets_dir, "synthetic_collinear.csv"))
cat("\n--- Testing Collinear Dataset (expecting singular matrix or extreme VIF) ---\n")
tryCatch({
    run_regression("Synthetic Collinear", synthetic_collinear, "y", c("x1", "x2", "x3"))
}, error = function(e) {
    cat(sprintf("  Expected error: %s\n", conditionMessage(e)))
})

# 4. Heteroscedastic
synthetic_hetero <- read.csv(file.path(datasets_dir, "synthetic_heteroscedastic.csv"))
run_regression("Synthetic Heteroscedastic", synthetic_hetero, "y", c("x"))

# 5. Nonlinear
synthetic_nonlinear <- read.csv(file.path(datasets_dir, "synthetic_nonlinear.csv"))
run_regression("Synthetic Nonlinear", synthetic_nonlinear, "y", c("x"))

# 6. Nonnormal
synthetic_nonnormal <- read.csv(file.path(datasets_dir, "synthetic_nonnormal.csv"))
run_regression("Synthetic Nonnormal", synthetic_nonnormal, "y", c("x"))

# 7. Autocorrelated
synthetic_auto <- read.csv(file.path(datasets_dir, "synthetic_autocorrelated.csv"))
run_regression("Synthetic Autocorrelated", synthetic_auto, "y", c("x"))

# 8. High VIF
synthetic_high_vif <- read.csv(file.path(datasets_dir, "synthetic_high_vif.csv"))
run_regression("Synthetic High VIF", synthetic_high_vif, "y", NULL)

# 9. Outliers
synthetic_outliers <- read.csv(file.path(datasets_dir, "synthetic_outliers.csv"))
run_regression("Synthetic Outliers", synthetic_outliers, "y", NULL)

# 10. Small Sample
synthetic_small <- read.csv(file.path(datasets_dir, "synthetic_small.csv"))
run_regression("Synthetic Small", synthetic_small, "y", NULL)

# --- Real-World Datasets ---

# Longley (famous multicollinearity test case)
longley <- read.csv(file.path(datasets_dir, "longley.csv"))
# Use Employed as dependent, other columns as predictors
longley_y <- "Employed"
longley_x <- setdiff(names(longley), c(longley_y, "Year", "Unnamed..0"))
run_regression("Longley", longley, longley_y, longley_x)

# Mtcars
mtcars <- read.csv(file.path(datasets_dir, "mtcars.csv"))
run_regression("Mtcars", mtcars, "mpg", c("cyl", "disp", "hp", "wt", "qsec"))

# Bodyfat
bodyfat <- read.csv(file.path(datasets_dir, "bodyfat.csv"))
run_regression("Bodyfat", bodyfat, names(bodyfat)[1], NULL)

# Prostate
prostate <- read.csv(file.path(datasets_dir, "prostate.csv"))
run_regression("Prostate", prostate, names(prostate)[ncol(prostate)], NULL)

cat("\n==============================================================\n")
cat("Extended validation complete!\n")
cat(sprintf("Results saved to: %s\n", output_dir))
cat("==============================================================\n")
