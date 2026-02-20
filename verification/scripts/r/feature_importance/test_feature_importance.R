# ============================================================================
# Feature Importance Test Reference Implementation (R)
# ============================================================================
# This script generates reference values for feature importance using R's
# native libraries (stats, car). The test validates that the Rust
# implementation matches R's behavior for feature importance metrics.
#
# Required packages:
#   install.packages(c("stats", "car", "jsonlite"))
#
# Usage:
#   Rscript test_feature_importance.R [csv_path] [output_dir]
#   Args:
#     csv_path    - Path to CSV file (first col = response, rest = predictors)
#                   Default: ../../../datasets/csv/mtcars.csv
#     output_dir  - Path to output directory
#                   Default: ../../../results/r
# ============================================================================

# Get script directory and resolve relative paths
args_all <- commandArgs(trailingOnly = FALSE)
script_arg <- args_all[grep("^--file=", args_all)]
if (length(script_arg) > 0) {
  script_path <- dirname(sub("--file=", "", script_arg))
  original_wd <- getwd()
  setwd(script_path)
} else {
  original_wd <- getwd()
}

# Add user library to path
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.4")
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

# Load required packages
suppressPackageStartupMessages({
  library(stats)
  library(car)
  library(jsonlite)
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

# Resolve paths relative to original working directory
resolve_path <- function(path) {
  is_absolute <- grepl("^(/|[A-Za-z]:)", path)
  if (!is_absolute) {
    file.path(original_wd, path)
  } else {
    path
  }
}

# =============================================================================
# Standardized Coefficients (using base R)
# =============================================================================

validate_standardized_coefficients <- function(df, predictors, response) {
  formula_str <- paste(response, "~", paste(predictors, collapse = " + "))
  formula <- as.formula(formula_str)

  model <- lm(formula, data = df)

  # Get coefficients (excluding intercept)
  coefficients <- coef(model)[-1]
  variable_names <- names(coefficients)

  # Compute standard deviations using base R
  # Use drop=FALSE to ensure we always get a data frame (even for single column)
  x_stds <- sapply(df[, predictors, drop = FALSE], sd, na.rm = TRUE)
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

  # VIF using car::vif
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
  coefficients <- coef(model)[-1]
  intercept <- coef(model)[1]

  # Compute means of predictors
  X <- as.matrix(df[, predictors])
  X_means <- colMeans(X)

  # Compute SHAP values: SHAP_i = coef_i * (x_i - mean(x_i))
  # Handle single-predictor case
  if (length(predictors) == 1) {
    shap_values <- matrix(coefficients * (X - X_means), nrow = 1)
  } else {
    shap_values <- apply(X, 1, function(row) {
      coefficients * (row - X_means)
    })
  }

  # shap_values is a k × n matrix (k features × n observations)
  # Mean absolute SHAP per feature: mean across observations (rows)
  mean_abs_shap <- rowMeans(abs(shap_values))

  list(
    variable_names = predictors,
    shap_values = t(shap_values),
    base_value = as.numeric(intercept),
    mean_abs_shap = as.numeric(mean_abs_shap)
  )
}

# =============================================================================
# Permutation Importance (using base R)
# =============================================================================

validate_permutation_importance <- function(df, predictors, response,
                                            n_permutations = 10,
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
# Main
# =============================================================================

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults
default_csv    <- "../../../datasets/csv/mtcars.csv"
default_output <- "../../../results/r"

# Parse arguments
csv_path_raw   <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir_raw <- ifelse(length(args) >= 2, args[2], default_output)

# Resolve paths
csv_path   <- resolve_path(csv_path_raw)
output_dir <- resolve_path(output_dir_raw)

# Validate CSV path
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

# Extract dataset name from filename
dataset_name <- tools::file_path_sans_ext(basename(csv_path))

cat(sprintf("Running feature importance test on dataset: %s\n", dataset_name))

# Load data
data <- read.csv(csv_path)
data <- convert_categorical_to_numeric(data, dataset_name)

# Use first column as response, rest as predictors
response <- names(data)[1]  # Get FIRST column (index 0 in 0-based, but R is 1-based)
predictors <- setdiff(names(data), response)

cat("Predictors:", paste(predictors, collapse = ", "), "\n")
cat("Response:", response, "\n")
cat("Observations:", nrow(data), "\n")

# Skip if too few predictors
if (length(predictors) < 1) {
  cat("Skipping: insufficient predictors\n")
  quit(status = 1)
}

n_features <- length(predictors)
has_vif <- n_features >= 2

# Build result object
result <- list(
  test = "feature_importance",
  method = "lm",
  dataset = dataset_name,
  n = nrow(data),
  k = n_features,
  variable_names = predictors,
  response = response
)

# 1. Standardized Coefficients
cat("\n[1/4] Standardized Coefficients...\n")
tryCatch({
  std_coefs <- validate_standardized_coefficients(data, predictors, response)
  result$standardized_coefficients <- std_coefs
  cat("   Variables:", paste(std_coefs$variable_names, collapse = ", "), "\n")
  cat("   Standardized Coefs:", paste(round(std_coefs$standardized_coefficients, 4), collapse = ", "), "\n")
}, error = function(e) {
  cat("   Error:", conditionMessage(e), "\n")
})

# 2. VIF Ranking (skip for single predictor)
if (has_vif) {
  cat("\n[2/4] VIF Ranking...\n")
  tryCatch({
    vif_result <- validate_vif_ranking(data, predictors, response)
    result$vif_ranking <- vif_result
    for (i in 1:min(5, nrow(vif_result$ranking))) {
      v <- vif_result$ranking[i, ]
      cat("   ", v$variable, ": VIF=", round(v$vif, 2), "(", v$interpretation, ")\n", sep="")
    }
  }, error = function(e) {
    cat("   Error:", conditionMessage(e), "\n")
  })
} else {
  cat("\n[2/4] VIF Ranking... Skipped (only 1 predictor)\n")
}

# 3. Linear SHAP
cat("\n[3/4] Linear SHAP...\n")
tryCatch({
  shap_result <- validate_linear_shap(data, predictors, response)
  result$shap <- shap_result
  cat("   Mean |SHAP|:", paste(round(shap_result$mean_abs_shap, 4), collapse = ", "), "\n")
}, error = function(e) {
  cat("   Error:", conditionMessage(e), "\n")
})

# 4. Permutation Importance
cat("\n[4/4] Permutation Importance...\n")
tryCatch({
  perm_result <- validate_permutation_importance(data, predictors, response, n_permutations = 10)
  result$permutation_importance <- perm_result
  cat("   Importance scores:", paste(round(perm_result$importance, 4), collapse = ", "), "\n")
}, error = function(e) {
  cat("   Error:", conditionMessage(e), "\n")
})

# Ensure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Write output
output_file <- file.path(output_dir, paste0(dataset_name, "_feature_importance.json"))
write_json(result, output_file, pretty = TRUE, auto_unbox = FALSE, digits = 22)

cat(sprintf("\nWrote: %s\n", normalizePath(output_file)))
