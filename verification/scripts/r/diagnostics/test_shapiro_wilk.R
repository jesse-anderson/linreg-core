#!/usr/bin/env Rscript
# ============================================================================
# Shapiro-Wilk Test Reference Implementation (R)
# ============================================================================
# This script generates reference values for the Shapiro-Wilk test using R's
# native shapiro.test function. The Shapiro-Wilk test checks whether residuals
# are normally distributed.
#
# Source: stats package, shapiro.test function
# Reference: Shapiro & Wilk (1965), "An analysis of variance test for normality
#            (complete samples)", Biometrika, Vol. 52, pp. 591-611
#
# Usage:
#   Rscript test_shapiro_wilk.R [csv_path] [output_dir]
#   Args:
#     csv_path  - Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     output_dir- Path to output directory
#                 Default: ../../results/r
# ============================================================================

library(stats)
library(jsonlite)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults
default_csv <- "../../datasets/csv/mtcars.csv"
default_output <- "../../results/r"

# Parse arguments
csv_path <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir <- ifelse(length(args) >= 2, args[2], default_output)

# Validate CSV path
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

# Extract dataset name from filename
dataset_name <- tools::file_path_sans_ext(basename(csv_path))

# Read CSV data
data <- read.csv(csv_path)

# Get the column names
cols <- colnames(data)

# First column is y (dependent variable), rest are x variables (predictors)
y <- data[[1]]
if (ncol(data) > 1) {
  x_vars <- data[, -1, drop = FALSE]
} else {
  # No predictors - simple mean model
  x_vars <- NULL
}

# Fit OLS model and compute residuals
if (is.null(x_vars)) {
  # Simple mean model
  residuals <- y - mean(y)
} else {
  # OLS regression
  model <- lm(y ~ ., data = x_vars)
  residuals <- resid(model)
}

# Run Shapiro-Wilk test using R's native function
# H0: Residuals are normally distributed
# H1: Residuals are not normally distributed
sw_result <- shapiro.test(residuals)

# Extract results
w_statistic <- sw_result$statistic
p_value <- sw_result$p.value

# Determine if test passed (null hypothesis not rejected at alpha = 0.05)
alpha <- 0.05
passed <- p_value > alpha

# Create interpretation text
if (passed) {
  interpretation <- sprintf(
    "p-value = %.4f is greater than %.2f. Cannot reject H0. No significant evidence that residuals deviate from normality.",
    p_value, alpha
  )
  guidance <- "The normality assumption appears to be met. Shapiro-Wilk test does not detect significant deviation from normal distribution."
} else {
  interpretation <- sprintf(
    "p-value = %.4f is less than or equal to %.2f. Reject H0. Significant evidence that residuals deviate from normality.",
    p_value, alpha
  )
  guidance <- "Consider transforming the dependent variable (e.g., log, Box-Cox transformation), using robust standard errors, or applying a different estimation method."
}

# Create output JSON object (using array format like other diagnostic tests)
output <- list(
  test_name = "Shapiro-Wilk Test for Normality",
  statistic = c(w_statistic),
  p_value = c(p_value),
  passed = c(passed),
  interpretation = interpretation,
  guidance = guidance
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_shapiro_wilk.json
output_file <- file.path(output_dir, paste0(dataset_name, "_shapiro_wilk.json"))

# Write output to JSON file
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
