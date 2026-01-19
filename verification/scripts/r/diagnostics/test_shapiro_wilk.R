#!/usr/bin/env Rscript
# ============================================================================
# Shapiro-Wilk Test Validation Script (R)
# ============================================================================
#
# This script runs the Shapiro-Wilk test on regression residuals using R's
# native shapiro.test function and outputs the results to JSON for validation
# against the Rust implementation.
#
# Usage: Rscript test_shapiro_wilk.R <csv_file>
# Example: Rscript test_shapiro_wilk.R ../../datasets/csv/mtcars.csv
#
# Output: JSON file with test results

library(stats)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: Rscript test_shapiro_wilk.R <csv_file>\n")
  quit(status = 1)
}

csv_file <- args[1]

# Check if file exists
if (!file.exists(csv_file)) {
  cat(paste("Error: File not found:", csv_file, "\n"))
  quit(status = 1)
}

# Read CSV data
data <- read.csv(csv_file)

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

# Get output file name from CSV file name
basename <- tools::file_path_sans_ext(basename(csv_file))
output_file <- paste0("../../results/r/", basename, "_shapiro_wilk.json")

# Create output directory if it doesn't exist
output_dir <- dirname(output_file)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Write output to JSON file
library(jsonlite)
write_json(output, output_file, pretty = TRUE, auto_unbox = FALSE)

cat(paste("Results written to:", output_file, "\n"))
cat(paste("W statistic:", w_statistic, "\n"))
cat(paste("p-value:", p_value, "\n"))
