# ============================================================================
# Ridge Regression Test Reference Implementation (R with glmnet)
# ============================================================================
# This script generates reference values for ridge regression using R's
# glmnet package with alpha=0. The test validates that the Rust implementation
# matches glmnet's behavior for coefficient paths and predictions.
#
# Source: glmnet package, glmnet function with alpha=0
# Reference: Friedman, Hastie, Tibshirani (2010), "Regularization Paths for
#            Generalized Linear Models via Coordinate Descent"
#
# Usage:
#   Rscript test_ridge.R [csv_path] [output_dir] [lambda_count]
#   Args:
#     csv_path    - Path to CSV file (first col = response, rest = predictors)
#                   Default: ../../../datasets/csv/mtcars.csv
#     output_dir  - Path to output directory
#                   Default: ../../../results/r
#     lambda_count- Number of lambda values to generate
#                   Default: 20
# ============================================================================

# Get script directory and resolve relative paths
# Get the script path from commandArgs (works with Rscript)
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
if (!require("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required. Install with: install.packages('glmnet')")
}
if (!require("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with: install.packages('jsonlite')")
}

# Helper function to convert categorical columns to numeric
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

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults (relative to original working directory)
default_csv <- "../../../datasets/csv/mtcars.csv"
default_output <- "../../../results/r"
default_lambda_count <- 20

# Resolve paths relative to original working directory (before setwd)
resolve_path <- function(path) {
  # Check if path is absolute (starts with / on Unix, or drive letter on Windows)
  is_absolute <- grepl("^(/|[A-Za-z]:)", path)
  if (!is_absolute) {
    # Relative path - resolve from original working directory
    file.path(original_wd, path)
  } else {
    path
  }
}

# Parse arguments
csv_path_raw <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir_raw <- ifelse(length(args) >= 2, args[2], default_output)
lambda_count <- as.integer(ifelse(length(args) >= 3, args[3], default_lambda_count))

# Resolve paths
csv_path <- resolve_path(csv_path_raw)
output_dir <- resolve_path(output_dir_raw)

# Validate CSV path
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

# Extract dataset name from filename
dataset_name <- tools::file_path_sans_ext(basename(csv_path))

cat(sprintf("Running ridge regression test on dataset: %s\n", dataset_name))

# Load data
data <- read.csv(csv_path)
data <- convert_categorical_to_numeric(data, dataset_name)

# Extract response (first column) and predictors (remaining columns)
y <- data[, 1]
X <- as.matrix(data[, -1, drop = FALSE])

n <- nrow(X)
p <- ncol(X)

cat(sprintf("  n = %d observations, p = %d predictors\n", n, p))

# Run glmnet for ridge regression (alpha = 0)
set.seed(42)  # For reproducibility
fit <- glmnet(X, y, family = "gaussian", alpha = 0,
             standardize = TRUE, intercept = TRUE,
             nlambda = lambda_count)

# Extract coefficient matrix (includes intercept in first row)
coef_matrix <- as.matrix(coef(fit))
# coef_matrix is (p+1) x lambda_count, with intercept in row 1

# Convert to list format for JSON serialization
# Each lambda gets a list of coefficients (including intercept as first element)
coef_list <- split(coef_matrix, rep(1:ncol(coef_matrix), each = nrow(coef_matrix)))
coef_list <- lapply(coef_list, as.vector)

# Get lambda sequence
lambda_seq <- fit$lambda

# Get degrees of freedom for each lambda
df <- fit$df

# Get predictions at a few representative lambdas
# Use actual length of lambda sequence (glmnet may generate fewer than requested)
actual_lambda_count <- length(lambda_seq)
test_indices <- c(1, ceiling(actual_lambda_count/2), actual_lambda_count)
test_lambdas <- lambda_seq[test_indices]

# Create test data (use first min(5, n) rows for prediction tests)
n_test <- min(5, n)
X_test <- X[1:n_test, , drop = FALSE]

predictions <- list()
for (i in test_indices) {
  pred <- predict(fit, newx = X_test, s = lambda_seq[i])
  predictions[[length(predictions) + 1]] <- as.vector(pred)
}

# Compute fitted values at the final lambda
fitted_values <- as.vector(predict(fit, newx = X, s = lambda_seq[actual_lambda_count]))

# Compute residuals at the final lambda
residuals <- as.vector(y - fitted_values)

# Build result object
result <- list(
  test = "ridge",
  method = "glmnet",
  alpha = 0,
  n = n,
  p = p,
  lambda_sequence = as.vector(lambda_seq),
  coefficients = coef_list,
  degrees_of_freedom = as.vector(df),
  test_lambdas = as.vector(test_lambdas),
  test_predictions = predictions,
  fitted_values = fitted_values,
  residuals = residuals,
  glmnet_version = as.character(packageVersion("glmnet"))
)

# Write output
output_file <- file.path(output_dir, paste0(dataset_name, "_ridge.json"))
write_json(result, output_file, pretty = TRUE, auto_unbox = TRUE)

cat(sprintf("Wrote: %s\n", normalizePath(output_file)))
