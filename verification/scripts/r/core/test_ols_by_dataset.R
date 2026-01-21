# ============================================================================
# OLS Regression Test Reference Implementation (R with lm)
# ============================================================================
# This script generates reference values for OLS regression using R's
# lm() function. The test validates that the Rust implementation matches
# R's behavior for coefficient estimates and statistics.
#
# Source: stats::lm (R base)
# Reference: "Applied Linear Statistical Models" (Kutner et al.)
#
# Usage:
#   Rscript test_ols_by_dataset.R [csv_path] [output_dir]
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

# Resolve paths relative to original working directory
resolve_path <- function(path) {
  is_absolute <- grepl("^(/|[A-Za-z]:)", path)
  if (!is_absolute) {
    file.path(original_wd, path)
  } else {
    path
  }
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults
default_csv <- "../../../datasets/csv/mtcars.csv"
default_output <- "../../../results/r"

# Parse arguments
csv_path_raw <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir_raw <- ifelse(length(args) >= 2, args[2], default_output)

# Resolve paths
csv_path <- resolve_path(csv_path_raw)
output_dir <- resolve_path(output_dir_raw)

# Validate CSV path
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

# Extract dataset name from filename
dataset_name <- tools::file_path_sans_ext(basename(csv_path))

cat(sprintf("Running OLS regression test on dataset: %s\n", dataset_name))

# Load data
data <- read.csv(csv_path)
data <- convert_categorical_to_numeric(data, dataset_name)

# Extract response (first column) and predictors (remaining columns)
response_col <- names(data)[1]
predictor_cols <- names(data)[-1]

# Build formula
formula_str <- paste(response_col, "~", paste(predictor_cols, collapse = " + "))
formula <- as.formula(formula_str)

# Fit OLS model
fit <- lm(formula, data = data)

# Extract model summary
summary_fit <- summary(fit)

# Extract coefficients
coefs <- summary_fit$coefficients
n_coef <- nrow(coefs)

# Extract statistics
coefficients <- as.vector(coefs[, 1])  # Estimates
std_errors <- as.vector(coefs[, 2])    # Standard errors
t_stats <- as.vector(coefs[, 3])       # t-statistics
p_values <- as.vector(coefs[, 4])      # p-values

# Variable names (including intercept)
variable_names <- c("Intercept", rownames(coefs)[-1])

# Model fit statistics
n <- length(fitted(fit))
k <- length(coefficients) - 1  # Number of predictors (excluding intercept)
df_residual <- fit$df.residual

r_squared <- summary_fit$r.squared
adj_r_squared <- summary_fit$adj.r.squared
f_statistic <- summary_fit$fstatistic[1]
f_p_value <- pf(f_statistic, k, df_residual, lower.tail = FALSE)

# Residuals and MSE
residuals <- as.vector(residuals(fit))
mse <- sum(residuals^2) / df_residual
std_error <- sqrt(mse)

# Confidence intervals (95%)
ci <- confint(fit, level = 0.95)
conf_int_lower <- as.vector(ci[, 1])
conf_int_upper <- as.vector(ci[, 2])

# VIF calculation (exclude intercept)
if (require("car", quietly = TRUE)) {
  # Only compute VIF if we have multiple predictors
  if (k > 1) {
    tryCatch({
      vif_values <- vif(fit)
      vif <- mapply(function(var, val) {
        list(variable = var, vif = val, rsquared = 1 - 1/val)
      }, names(vif_values), vif_values, SIMPLIFY = FALSE)
    }, error = function(e) {
      # VIF failed, return empty list
      vif <<- list()
    })
  } else {
    vif <- list()
  }
} else {
  vif <- list()
}

# Build result object
result <- list(
  test = "ols",
  method = "lm",
  dataset = dataset_name,
  formula = formula_str,
  n = n,
  k = k,
  df_residual = as.integer(df_residual),
  variable_names = variable_names,
  coefficients = coefficients,
  std_errors = std_errors,
  t_stats = t_stats,
  p_values = p_values,
  r_squared = r_squared,
  adj_r_squared = adj_r_squared,
  f_statistic = f_statistic,
  f_p_value = f_p_value,
  mse = mse,
  std_error = std_error,
  conf_int_lower = conf_int_lower,
  conf_int_upper = conf_int_upper,
  residuals = residuals,
  vif = vif
)

# Ensure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Write output
output_file <- file.path(output_dir, paste0(dataset_name, "_ols.json"))
write_json(result, output_file, pretty = TRUE, auto_unbox = TRUE)

cat(sprintf("Wrote: %s\n", normalizePath(output_file)))
