# ============================================================================
# Feature Importance Validation - R Reference Implementation
# ============================================================================
# This script computes feature importance metrics using NATIVE R libraries
# (stats, car, lmtest) to generate reference values for validation against
# the Rust implementation.
#
# Required packages:
#   install.packages(c("stats", "car", "lmtest", "caret"))
# ============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(stats)
  library(car)
  library(lmtest)
})

# =============================================================================
# Data Helper Functions
# =============================================================================

convert_categorical_to_numeric <- function(data, dataset_name) {
  non_numeric_cols <- names(data)[sapply(data, function(x) !is.numeric(x))]

  if (length(non_numeric_cols) > 0) {
    cat(paste0("INFO: Dataset '", dataset_name, "' contains non-numeric columns: ",
               paste(non_numeric_cols, collapse = ", "), "\n"))
    cat("Converting categorical variables to numeric representations...\n")

    for (col in non_numeric_cols) {
      if (is.factor(data[[col]])) {
        data[[col]] <- as.numeric(data[[col]])
      } else if (is.character(data[[col]])) {
        temp_numeric <- as.numeric(data[[col]])
        if (any(is.na(temp_numeric))) {
          data[[col]] <- as.numeric(as.factor(data[[col]]))
        } else {
          data[[col]] <- temp_numeric
        }
      }
    }
  }

  return(data)
}

load_dataset <- function(dataset_name) {
  base_path <- file.path("..", "..", "..", "datasets", "csv")

  datasets <- list(
    mtcars = file.path(base_path, "mtcars.csv"),
    iris = file.path(base_path, "iris.csv"),
    faithful = file.path(base_path, "faithful.csv"),
    cars_stopping = file.path(base_path, "cars_stopping.csv")
  )

  if (!(dataset_name %in% names(datasets))) {
    stop(paste("Unknown dataset:", dataset_name))
  }

  df <- read.csv(datasets[[dataset_name]])
  df <- convert_categorical_to_numeric(df, dataset_name)
  return(df)
}

# =============================================================================
# Standardized Coefficients (using base R)
# =============================================================================

validate_standardized_coefficients <- function(df, predictors, response) {
  # Standardize coefficients: beta* = beta * (sigma_x / sigma_y)
  # Using base R's sd() function

  formula_str <- paste(response, "~", paste(predictors, collapse = " + "))
  formula <- as.formula(formula_str)

  model <- lm(formula, data = df)

  # Get coefficients (excluding intercept)
  coefficients <- coef(model)[-1]  # Remove intercept
  variable_names <- names(coefficients)

  # Compute standard deviations using base R
  x_stds <- sapply(df[, predictors], sd, na.rm = TRUE)
  y_std <- sd(df[[response]], na.rm = TRUE)

  # Standardized coefficients
  std_coefs <- coefficients * (x_stds / y_std)

  list(
    variable_names = variable_names,
    standardized_coefficients = as.numeric(std_coefs),
    y_std = y_std,
    raw_coefficients = as.numeric(coefficients)
  )
}

# =============================================================================
# VIF Ranking (using car package)
# =============================================================================

validate_vif_ranking <- function(df, predictors, response) {
  formula_str <- paste(response, "~", paste(predictors, collapse = " + "))
  formula <- as.formula(formula_str)

  model <- lm(formula, data = df)

  # VIF using car::vif - this is the REFERENCE implementation
  vif_values <- car::vif(model)

  # Add interpretation
  interpretations <- sapply(vif_values, function(v) {
    if (v < 5) "Low multicollinearity"
    else if (v < 10) "Moderate multicollinearity"
    else "High multicollinearity"
  })

  vif_data <- data.frame(
    variable = names(vif_values),
    vif = as.numeric(vif_values),
    rsquared = 1 - 1/vif_values,
    interpretation = interpretations,
    stringsAsFactors = FALSE
  )

  # Sort by VIF ascending (least redundant first)
  vif_sorted <- vif_data[order(vif_data$vif), ]

  list(
    variable_names = vif_sorted$variable,
    vif_values = vif_sorted$vif,
    ranking = vif_sorted
  )
}

# =============================================================================
# Linear SHAP Values (exact, closed-form)
# =============================================================================

validate_linear_shap <- function(df, predictors, response) {
  formula_str <- paste(response, "~", paste(predictors, collapse = " + "))
  formula <- as.formula(formula_str)

  model <- lm(formula, data = df)

  # Get coefficients
  coefficients <- coef(model)[-1]  # Exclude intercept
  intercept <- coef(model)[1]

  # Compute means of predictors
  X <- as.matrix(df[, predictors])
  X_means <- colMeans(X)

  # Compute SHAP values: SHAP_i = coef_i * (x_i - mean(x_i))
  shap_values <- apply(X, 1, function(row) {
    coefficients * (row - X_means)
  })

  # Mean absolute SHAP per feature
  mean_abs_shap <- colMeans(abs(shap_values))

  list(
    variable_names = paste0("X", 1:length(predictors)),
    shap_values = t(shap_values),  # Transpose to match Python format (obs x features)
    base_value = as.numeric(intercept),
    mean_abs_shap = as.numeric(mean_abs_shap)
  )
}

# =============================================================================
# Permutation Importance (using base R and caret)
# =============================================================================

validate_permutation_importance <- function(df, predictors, response,
                                                   n_permutations = 50,
                                                   random_seed = 42) {
  set.seed(random_seed)

  X <- as.matrix(df[, predictors])
  y <- df[[response]]

  # Fit baseline model
  model <- lm(y ~ X)
  baseline_r2 <- summary(model)$r.squared

  n_obs <- nrow(X)
  n_features <- ncol(X)
  importance_scores <- numeric(n_features)

  for (j in 1:n_features) {
    perm_r2_sum <- 0

    for (iter in 1:n_permutations) {
      X_permuted <- X
      # Shuffle column j
      perm_col <- X_permuted[, j]
      X_permuted[, j] <- sample(perm_col)

      # Fit on permuted data
      perm_model <- lm(y ~ X_permuted)
      perm_r2 <- summary(perm_model)$r.squared

      perm_r2_sum <- perm_r2_sum + (baseline_r2 - perm_r2)
    }

    importance_scores[j] <- perm_r2_sum / n_permutations
  }

  list(
    variable_names = predictors,
    importance = as.numeric(importance_scores),
    baseline_score = baseline_r2,
    n_permutations = n_permutations,
    seed = random_seed
  )
}

# =============================================================================
# Main Validation Runner
# =============================================================================

main <- function() {
  cat("=", rep(70), "\n")
  cat("Feature Importance Validation - R Reference Implementation\n")
  cat("Using NATIVE libraries: stats, car, lmtest\n")
  cat("=", rep(70), "\n")

  # Datasets to validate
  datasets <- c("mtcars", "iris", "faithful", "cars_stopping")

  results <- list()

  for (dataset_name in datasets) {
    cat("\n", paste(rep("=", 70), collapse = ""), "\n")
    cat("Dataset:", dataset_name, "\n")
    cat(paste(rep("=", 70), collapse = ""), "\n")

    tryCatch({
      df <- load_dataset(dataset_name)

      # Use first column as response, rest as predictors
      response <- names(df)[1]
      predictors <- setdiff(names(df), response)

      cat("Predictors:", paste(predictors, collapse = ", "), "\n")
      cat("Response:", response, "\n")
      cat("Observations:", nrow(df), "\n")

      if (length(predictors) < 1) {
        cat("Skipping: insufficient predictors\n")
        next
      }

      dataset_results <- list()

      # 1. Standardized Coefficients (base R)
      cat("\n[1/4] Standardized Coefficients (base R stats)...\n")
      tryCatch({
        std_coefs <- validate_standardized_coefficients(df, predictors, response)
        dataset_results$standardized_coefficients <- std_coefs
        cat("   Variables:", paste(std_coefs$variable_names, collapse=", "), "\n")
        cat("   Standardized Coefs:", paste(round(std_coefs$standardized_coefficients, 4), collapse=", "), "\n")
      }, error = function(e) {
        cat("   Error:", conditionMessage(e), "\n")
      })

      # 2. VIF Ranking (car::vif)
      cat("\n[2/4] VIF Ranking (car package)...\n")
      tryCatch({
        vif_result <- validate_vif_ranking(df, predictors, response)
        dataset_results$vif_ranking <- vif_result
        for (i in 1:min(5, nrow(vif_result$ranking))) {
          v <- vif_result$ranking[i, ]
          cat("   ", v$variable, ": VIF=", round(v$vif, 2), "(", v$interpretation, ")\n", sep="")
        }
      }, error = function(e) {
        cat("   Error:", conditionMessage(e), "\n")
      })

      # 3. Linear SHAP (base R)
      cat("\n[3/4] Linear SHAP (closed-form)...\n")
      tryCatch({
        shap_result <- validate_linear_shap(df, predictors, response)
        dataset_results$shap <- shap_result
        cat("   Mean |SHAP|:", paste(round(shap_result$mean_abs_shap, 4), collapse=", "), "\n")
      }, error = function(e) {
        cat("   Error:", conditionMessage(e), "\n")
      })

      # 4. Permutation Importance
      cat("\n[4/4] Permutation Importance (n=10 for speed)...\n")
      tryCatch({
        perm_result <- validate_permutation_importance(df, predictors, response, n_permutations=10)
        dataset_results$permutation_importance <- perm_result
        cat("   Importance scores:", paste(round(perm_result$importance, 4), collapse=", "), "\n")
      }, error = function(e) {
        cat("   Error:", conditionMessage(e), "\n")
      })

      results[[dataset_name]] <- dataset_results

    }, error = function(e) {
      cat("Error processing", dataset_name, ":", conditionMessage(e), "\n")
    })
  }

  # Save results to JSON
  output_dir <- file.path("..", "..", "..", "results", "r", "feature_importance")
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  output_file <- file.path(output_dir, "feature_importance_reference.json")
  jsonlite::write_json(results, output_file, pretty = TRUE)

  cat("\n", paste(rep("=", 70), collapse = ""), "\n")
  cat("Results saved to:", output_file, "\n")
  cat(paste(rep("=", 70), collapse = ""), "\n")

  # Print summary
  cat("\nSUMMARY:\n")
  cat("-------\n")
  for (dataset_name in names(results)) {
    cat("\n", dataset_name, ":\n")
    metrics <- results[[dataset_name]]

    if (!is.null(metrics$standardized_coefficients)) {
      sc <- metrics$standardized_coefficients$standardized_coefficients
      cat("  Standardized Coefs:", paste(round(sc, 4), collapse=", "), "\n")
    }
    if (!is.null(metrics$vif_ranking)) {
      v <- metrics$vif_ranking$vif_values
      cat("  VIF values:", paste(round(v, 2), collapse=", "), "\n")
    }
    if (!is.null(metrics$shap)) {
      s <- metrics$shap$mean_abs_shap
      cat("  Mean |SHAP|:", paste(round(s, 4), collapse=", "), "\n")
    }
    if (!is.null(metrics$permutation_importance)) {
      p <- metrics$permutation_importance$importance
      cat("  Permutation Importance:", paste(round(p, 4), collapse=", "), "\n")
    }
  }
}

# Run validation if called directly
if (exists("commandArgs") && length(commandArgs()) > 0 && "--cli" %in% commandArgs()) {
  main()
}
