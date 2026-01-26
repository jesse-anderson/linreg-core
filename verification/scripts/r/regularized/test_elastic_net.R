# ============================================================================
# Elastic Net Regression Validation Script for R
# ============================================================================
#
# This script generates reference elastic net results using glmnet for
# comparison with the Rust implementation.
#
# Usage:
#   Rscript test_elastic_net.R
#
# Output:
#   verification/results/r/{dataset}_elastic_net_alpha_{alpha}.json

library(glmnet)
library(jsonlite)

# Set working directory to script location
script_dir <- dirname(sys.frame(1)$ofile)
setwd(script_dir)

# Path configuration
datasets_dir <- file.path("..", "..", "..", "datasets", "csv")
results_dir <- file.path("..", "..", "..", "results", "r")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# Datasets to test
test_datasets <- c(
  "mtcars",
  "bodyfat",
  "prostate",
  "longley",
  "synthetic_collinear",
  "synthetic_high_vif"
)

# Alpha values to test
test_alphas <- c(0.0, 0.25, 0.5, 0.75, 0.9, 1.0)

# Number of lambdas in the path
nlambda <- 100

# Lambda min ratio (default in glmnet)
lambda_min_ratio <- ifelse(nrow(x) < ncol(x), 0.01, 0.0001)

# ============================================================================
# Helper Functions
# ============================================================================

load_dataset <- function(dataset_name) {
  csv_path <- file.path(datasets_dir, paste0(dataset_name, ".csv"))

  if (!file.exists(csv_path)) {
    stop(paste("Dataset file not found:", csv_path))
  }

  data <- read.csv(csv_path, stringsAsFactors = FALSE)

  # First column is y (dependent), rest are X (predictors)
  y <- data[, 1]
  X <- as.matrix(data[, -1])

  # Add dummy predictor if only 1 predictor (glmnet requires 2+ columns)
  # Use a column of zeros as dummy (doesn't affect regularization test)
  if (ncol(X) < 2) {
    cat(sprintf("  INFO: Adding dummy predictor (zeros) for %s\n", dataset_name))
    X <- cbind(X, rep(0, length(y)))
  }

  list(y = y, X = X, n = length(y), p = ncol(X))
}

generate_elastic_net_result <- function(dataset_name, alpha) {
  cat(sprintf("\n=== Processing: %s (alpha = %.2f) ===\n", dataset_name, alpha))

  # Load dataset
  data <- load_dataset(dataset_name)
  X <- data$X
  y <- data$y

  # Fit glmnet
  t_start <- Sys.time()
  fit <- glmnet(
    X, y,
    alpha = alpha,
    nlambda = nlambda,
    standardize = TRUE,
    intercept = TRUE
  )
  t_elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

  # Extract lambda sequence
  lambda_sequence <- fit$lambda

  # Extract coefficient matrix (including intercept)
  # glmnet returns a matrix where each column is coefficients for a lambda
  coef_matrix <- as.matrix(coef(fit))
  # coef_matrix has p+1 rows (intercept + p predictors) and nlambda columns

  # Transpose so coefficients[[]] is a list of coefficient vectors
  # Each element is a vector: (intercept, beta_1, ..., beta_p)
  coefficients <- lapply(1:nlambda, function(i) {
    as.numeric(coef_matrix[, i])
  })

  # Calculate non-zero counts (excluding intercept)
  nonzero_counts <- sapply(coefficients, function(coef) {
    sum(abs(coef[-1]) > 1e-10)  # Exclude intercept from count
  })

  # Get fitted values and residuals at the final lambda
  y_pred_final <- as.numeric(predict(fit, X, s = lambda_sequence[nlambda]))
  residuals_final <- y - y_pred_final

  # Test predictions at specific lambdas (first, middle, last)
  test_indices <- c(1, floor(nlambda / 2), nlambda)
  test_predictions <- lapply(test_indices, function(idx) {
    as.numeric(predict(fit, X, s = lambda_sequence[idx]))
  })

  # Build result object
  result <- list(
    test = "elastic_net",
    method = "glmnet",
    alpha = alpha,
    n = data$n,
    p = data$p,
    nlambda = nlambda,
    lambda_sequence = as.numeric(lambda_sequence),
    coefficients = coefficients,
    nonzero_counts = as.integer(nonzero_counts),
    fitted_values = as.numeric(y_pred_final),
    residuals = as.numeric(residuals_final),
    test_predictions = test_predictions,
    glmnet_version = as.character(packageVersion("glmnet")),
    fit_time_seconds = t_elapsed
  )

  result
}

save_result <- function(result, dataset_name, alpha) {
  # Create filename with alpha value
  alpha_str <- gsub("\\.", "_", as.character(alpha))
  filename <- sprintf("%s_elastic_net_alpha_%s.json", dataset_name, alpha_str)
  output_path <- file.path(results_dir, filename)

  # Write to JSON
  write toJSON(result, auto_unbox = TRUE, pretty = TRUE), output_path

  cat(sprintf("  Saved: %s\n", filename))
}

# ============================================================================
# Main Execution
# ============================================================================

cat("\n")
cat("╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║  Elastic Net Reference Generation (glmnet)                         ║\n")
cat("╚══════════════════════════════════════════════════════════════════════╝\n")
cat("\n")

total_tests <- length(test_datasets) * length(test_alphas)
current_test <- 0

for (dataset_name in test_datasets) {
  for (alpha in test_alphas) {
    current_test <- current_test + 1
    cat(sprintf("\n[Test %d/%d] ", current_test, total_tests))

    tryCatch({
      result <- generate_elastic_net_result(dataset_name, alpha)
      save_result(result, dataset_name, alpha)
    }, error = function(e) {
      cat(sprintf("\n  ERROR processing %s (alpha=%.2f): %s\n", dataset_name, alpha, e$message))
    })
  }
}

cat("\n")
cat("╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║  Summary                                                           ║\n")
cat("╚══════════════════════════════════════════════════════════════════════╝\n")
cat("\n")
cat(sprintf("  Datasets processed: %d\n", length(test_datasets)))
cat(sprintf("  Alpha values tested: %s\n", paste(test_alphas, collapse = ", ")))
cat(sprintf("  Total reference files: %d\n", total_tests))
cat("\n")
cat("  Results saved to:", results_dir, "\n")
cat("\n")
