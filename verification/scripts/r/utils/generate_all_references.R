#!/usr/bin/env Rscript
# ============================================================================
# Master Reference Generation Script (R)
# ============================================================================
#
# This script generates reference outputs for all test datasets using R's
# native statistical libraries. The outputs are saved as JSON files for
# cross-validation with the Rust WASM implementation.
#
# Usage: Rscript generate_all_references.R
#
# Output: JSON files in verification/datasets/references/expanded/
# ============================================================================

library(jsonlite)

# Check for required packages
required_packages <- c("lmtest", "car", "skedastic", "tseries")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if (length(missing_packages) > 0) {
  message("Installing missing packages: ", paste(missing_packages, collapse = ", "))
  install.packages(missing_packages)
}

library(lmtest)
library(car)

# Try to load skedastic, provide fallback if not available
use_skedastic <- requireNamespace("skedastic", quietly = TRUE)
if (use_skedastic) {
  library(skedastic)
} else {
  message("Note: skedastic package not available. White test will use manual implementation.")
}

library(tseries)

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR <- "verification/datasets/references/expanded"
ALPHA <- 0.05

# Create output directory if it doesn't exist
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# ============================================================================
# Dataset Definitions
# ============================================================================

# Each dataset has: name, y vector, x_vars list, variable names
datasets <- list()

# 1. Housing dataset (synthetic)
datasets$housing <- list(
  y = c(245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
        445.8, 167.9, 367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9,
        223.4, 312.5, 156.8, 423.7, 267.9),
  x_vars = list(
    Square_Feet = c(1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
                   2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
                   1250.0, 1700.0, 850.0, 2350.0, 1400.0),
    Bedrooms = c(3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0,
                4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 4.0, 2.0, 4.0,
                3.0, 3.0, 2.0, 4.0, 3.0),
    Age = c(15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0,
           3.0, 30.0, 6.0, 14.0, 22.0, 1.0, 16.0, 9.0, 28.0, 4.0,
           19.0, 11.0, 35.0, 3.0, 13.0)
  ),
  variable_names = c("Intercept", "Square_Feet", "Bedrooms", "Age")
)

# 2. Perfect fit dataset (synthetic)
datasets$perfect_fit <- list(
  y = c(5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0),
  x_vars = list(
    x1 = c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    x2 = c(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
  ),
  variable_names = c("Intercept", "x1", "x2")
)

# 3. Single predictor dataset
datasets$single_predictor <- list(
  y = c(3.1, 5.0, 6.9, 9.0, 11.1, 12.8, 15.0, 17.1, 18.9, 21.0),
  x_vars = list(
    x = c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
  ),
  variable_names = c("Intercept", "x")
)

# 4. High multicollinearity dataset
datasets$high_multicollinearity <- list(
  y = c(5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0),
  x_vars = list(
    x1 = c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    x2 = c(2.02, 4.01, 5.99, 8.01, 9.98, 12.02, 13.99, 16.01, 17.98, 20.02),
    x3 = c(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
  ),
  variable_names = c("Intercept", "x1", "x2", "x3")
)

# 5. Small n dataset (boundary case)
datasets$small_n <- list(
  y = c(3.1, 5.0, 6.9, 9.0, 11.1),
  x_vars = list(
    x1 = c(1.0, 2.0, 3.0, 4.0, 5.0),
    x2 = c(2.0, 3.0, 4.0, 5.0, 6.0)
  ),
  variable_names = c("Intercept", "x1", "x2")
)

# ============================================================================
# White Test Implementation (Fallback)
# ============================================================================

white_test_fallback <- function(model) {
  # Manual implementation of White test if skedastic is not available
  # Z = [1, X, X^2]
  x <- model.matrix(model)
  n <- nrow(x)
  k <- ncol(x) - 1  # exclude intercept

  # Create Z matrix: intercept, X, X^2 (no interactions)
  Z <- cbind(1, x[, -1, drop=FALSE])  # intercept + linear terms

  # Add squared terms
  for (j in 2:ncol(x)) {
    squared <- x[, j]^2
    # Check if redundant
    if (!all(abs(squared - Z[, 1]) < 1e-10)) {  # Not constant
      Z <- cbind(Z, squared)
    }
  }

  residuals <- residuals(model)
  residuals_sq <- residuals^2

  # Auxiliary regression
  aux_model <- lm(residuals_sq ~ Z - 1)
  r_squared <- summary(aux_model)$r.squared

  # LM statistic
  lm_stat <- n * r_squared
  df <- ncol(Z) - 1

  # p-value (chi-squared)
  p_value <- 1 - pchisq(lm_stat, df)

  list(statistic = lm_stat, p_value = p_value)
}

# ============================================================================
# Main Validation Function
# ============================================================================

generate_reference <- function(dataset_name, dataset) {
  cat(sprintf("\n=== Generating R reference for: %s ===\n", dataset_name))

  y <- dataset$y
  x_vars <- dataset$x_vars
  variable_names <- dataset$variable_names

  # Create data frame
  df <- as.data.frame(x_vars)
  df$y <- y

  # Create formula
  formula_str <- paste("y ~", paste(names(x_vars), collapse = " + "))
  formula <- as.formula(formula_str)

  # Fit model
  model <- lm(formula, data = df)

  # Get summary
  smry <- summary(model)

  # Basic results
  n <- length(y)
  k <- length(x_vars)
  df_residual <- n - k - 1

  # Coefficients
  coefs <- coef(model)
  std_errors <- smry$coefficients[, "Std. Error"]
  t_stats <- smry$coefficients[, "t value"]
  p_values <- smry$coefficients[, "Pr(>|t|)"]

  # Confidence intervals
  conf_int <- confint(model, level = 1 - ALPHA)

  # Model fit statistics
  r_squared <- smry$r.squared
  adj_r_squared <- smry$adj.r.squared
  f_statistic <- smry$fstatistic[1]
  f_p_value <- pf(f_statistic, k, df_residual, lower.tail = FALSE)

  # Residuals
  residuals_val <- residuals(model)
  mse <- sum(residuals_val^2) / df_residual
  std_error <- sqrt(mse)

  # Standardized residuals
  leverage <- hatvalues(model)
  standardized_residuals <- rstandard(model)

  # Predictions
  predictions <- fitted(model)

  # VIF
  vif_results <- tryCatch({
    vif_values <- vif(model)
    lapply(seq_along(vif_values), function(i) {
      list(
        variable = names(vif_values)[i],
        vif = vif_values[i],
        rsquared = 1 - 1 / vif_values[i]
      )
    })
  }, error = function(e) {
    list()
  })

  # Rainbow test
  rainbow_result <- tryCatch({
    rt <- raintest(model, fraction = 0.5, order.by = NULL)
    list(
      statistic = as.numeric(rt$statistic),
      p_value = as.numeric(rt$p.value),
      passed = as.numeric(rt$p.value) > ALPHA
    )
  }, error = function(e) {
    list(statistic = NA, p_value = NA, passed = FALSE)
  })

  # Harvey-Collier test
  hc_result <- tryCatch({
    hct <- harvtest(model, order.by = fitted(model))
    list(
      statistic = as.numeric(hct$statistic),
      p_value = as.numeric(hct$p.value),
      passed = as.numeric(hct$p.value) > ALPHA
    )
  }, error = function(e) {
    list(statistic = NA, p_value = NA, passed = FALSE)
  })

  # Breusch-Pagan test
  bp_result <- tryCatch({
    bpt <- bptest(model, studentize = TRUE)
    list(
      statistic = as.numeric(bpt$statistic),
      p_value = as.numeric(bpt$p.value),
      passed = as.numeric(bpt$p.value) > ALPHA
    )
  }, error = function(e) {
    list(statistic = NA, p_value = NA, passed = FALSE)
  })

  # White test
  white_result <- tryCatch({
    if (use_skedastic) {
      wt <- white(model, interactions = FALSE)
      list(
        statistic = as.numeric(wt$statistic),
        p_value = as.numeric(wt$p.value),
        passed = as.numeric(wt$p.value) > ALPHA
      )
    } else {
      white_test_fallback(model)
    }
  }, error = function(e) {
    # Fallback to manual implementation
    tryCatch(white_test_fallback(model), error = function(e2) {
      list(statistic = NA, p_value = NA, passed = FALSE)
    })
  })

  # Jarque-Bera test
  jb_result <- tryCatch({
    jbt <- jarque.bera.test(residuals(model))
    list(
      statistic = as.numeric(jbt$statistic),
      p_value = as.numeric(jbt$p.value),
      passed = as.numeric(jbt$p.value) > ALPHA
    )
  }, error = function(e) {
    list(statistic = NA, p_value = NA, passed = FALSE)
  })

  # Durbin-Watson test
  dw_result <- tryCatch({
    dwt <- dwtest(model)
    list(
      statistic = as.numeric(dwt$statistic),
      p_value = as.numeric(dwt$p.value),
      passed = TRUE  # DW test interpretation is context-dependent
    )
  }, error = function(e) {
    list(statistic = NA, p_value = NA, passed = FALSE)
  })

  # Breusch-Godfrey test (order = 1, Chi-squared)
  bg_result <- tryCatch({
    bgt <- bgtest(model, order = 1, type = "Chisq")
    list(
      statistic = as.numeric(bgt$statistic),
      p_value = as.numeric(bgt$p.value),
      passed = as.numeric(bgt$p.value) > ALPHA
    )
  }, error = function(e) {
    list(statistic = NA, p_value = NA, passed = FALSE)
  })

  # Build output structure
  output <- list(
    dataset_name = dataset_name,
    coefficients = as.numeric(coefs),
    std_errors = as.numeric(std_errors),
    t_stats = as.numeric(t_stats),
    p_values = as.numeric(p_values),
    r_squared = r_squared,
    adj_r_squared = adj_r_squared,
    f_statistic = as.numeric(f_statistic),
    f_p_value = as.numeric(f_p_value),
    mse = mse,
    std_error = std_error,
    conf_int_lower = as.numeric(conf_int[, 1]),
    conf_int_upper = as.numeric(conf_int[, 2]),
    residuals = as.numeric(residuals_val),
    standardized_residuals = as.numeric(standardized_residuals),
    leverage = as.numeric(leverage),
    vif = vif_results,
    rainbow = rainbow_result,
    harvey_collier = hc_result,
    breusch_pagan = bp_result,
    white = white_result,
    jarque_bera = jb_result,
    durbin_watson = dw_result,
    breusch_godfrey = bg_result,
    n = n,
    k = k,
    df = df_residual,
    variable_names = variable_names
  )

  # Write to JSON
  output_file <- file.path(OUTPUT_DIR, paste0(dataset_name, "_r.json"))
  write_json(output, output_file, pretty = TRUE, auto_unbox = TRUE)

  cat(sprintf("  -> Wrote: %s\n", output_file))
  cat(sprintf("  R² = %.4f, F = %.2f\n", r_squared, f_statistic))

  return(output)
}

# ============================================================================
# Main Execution
# ============================================================================

main <- function() {
  cat("======================================================================\n")
  cat("  R Reference Generation Script\n")
  cat("======================================================================\n")
  cat(sprintf("Output directory: %s\n", OUTPUT_DIR))
  cat(sprintf("Number of datasets: %d\n", length(datasets)))
  cat(sprintf("Using skedastic for White test: %s\n", use_skedastic))

  # Generate references for each dataset
  results <- list()
  for (name in names(datasets)) {
    result <- tryCatch({
      generate_reference(name, datasets[[name]])
    }, error = function(e) {
      cat(sprintf("ERROR generating %s: %s\n", name, e$message))
      NULL
    })
    if (!is.null(result)) {
      results[[name]] <- result
    }
  }

  cat("\n======================================================================\n")
  cat("Summary\n")
  cat("======================================================================\n")
  cat(sprintf("Successfully generated: %d / %d datasets\n",
              length(results), length(datasets)))

  # Print table of results
  if (length(results) > 0) {
    cat("\nDataset           R²       F-stat   Rainbow  HC       BP       White    JB       DW       BG\n")
    cat("--------------------------------------------------------------------------------------------------\n")
    for (name in names(results)) {
      r <- results[[name]]
      cat(sprintf("%-16s  %.4f   %6.2f   %s  %s  %s  %s  %s  %s  %s\n",
                  name,
                  r$r_squared,
                  r$f_statistic,
                  ifelse(is.na(r$rainbow$p_value), "N/A",
                         ifelse(r$rainbow$p_value > ALPHA, "PASS", "FAIL")),
                  ifelse(is.na(r$harvey_collier$p_value), "N/A",
                         ifelse(r$harvey_collier$p_value > ALPHA, "PASS", "FAIL")),
                  ifelse(is.na(r$breusch_pagan$p_value), "N/A",
                         ifelse(r$breusch_pagan$p_value > ALPHA, "PASS", "FAIL")),
                  ifelse(is.na(r$white$p_value), "N/A",
                         ifelse(r$white$p_value > ALPHA, "PASS", "FAIL")),
                  ifelse(is.na(r$jarque_bera$p_value), "N/A",
                         ifelse(r$jarque_bera$p_value > ALPHA, "PASS", "FAIL")),
                  ifelse(is.na(r$durbin_watson$statistic), "N/A",
                         sprintf("%.2f", r$durbin_watson$statistic)),
                  ifelse(is.na(r$breusch_godfrey$p_value), "N/A",
                         ifelse(r$breusch_godfrey$p_value > ALPHA, "PASS", "FAIL"))
      ))
    }
  }

  cat("\n")
}

# Run main function
main()
