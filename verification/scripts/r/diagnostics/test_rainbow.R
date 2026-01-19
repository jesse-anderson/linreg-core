# ============================================================================
# Rainbow Test Reference Implementation (R)
# ============================================================================
# This script generates reference values for the Rainbow test using R's
# lmtest::raintest function. The Rainbow test checks for linearity by comparing
# the fit on a central subset of observations against the fit on all observations.
#
# Source: lmtest package, raintest function
# Reference: Utts (1982), "The Rainbow Test for Lack of Fit in Regression"
#
# Usage:
#   Rscript test_rainbow.R [csv_path] [output_dir] [fraction]
#   Args:
#     csv_path  - Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     output_dir- Path to output directory
#                 Default: ../../results/r
#     fraction  - Fraction of data for central subset
#                 Default: 0.5
# ============================================================================

library(lmtest)
library(jsonlite)

# Helper function to convert categorical columns to numeric
convert_categorical_to_numeric <- function(data, dataset_name) {
  # Check for non-numeric columns
  non_numeric_cols <- names(data)[sapply(data, function(x) !is.numeric(x))]

  if (length(non_numeric_cols) > 0) {
    cat(paste0("INFO: Dataset '", dataset_name, "' contains non-numeric columns: ",
               paste(non_numeric_cols, collapse = ", "), "\n"))
    cat("Converting categorical variables to numeric representations...\n")

    for (col in non_numeric_cols) {
      if (is.factor(data[[col]])) {
        # For factors, use integer level encoding
        unique_vals <- length(unique(data[[col]]))
        data[[col]] <- as.numeric(data[[col]])
        cat(paste0("  ", col, ": ", unique_vals, " unique values -> integer level encoding\n"))
      } else if (is.character(data[[col]])) {
        # For character columns, first try to convert to numeric directly
        # If that produces NAs, convert to factor and use integer encoding
        unique_vals <- length(unique(data[[col]]))
        temp_numeric <- as.numeric(data[[col]])
        if (any(is.na(temp_numeric))) {
          # Contains non-numeric strings, use factor encoding
          data[[col]] <- as.numeric(as.factor(data[[col]]))
          cat(paste0("  ", col, ": ", unique_vals, " unique values -> integer level encoding\n"))
        } else {
          # Successfully converted to numeric
          data[[col]] <- temp_numeric
          cat(paste0("  ", col, ": ", unique_vals, " unique values -> numeric encoding\n"))
        }
      }
    }
  }

  return(data)
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults
default_csv <- "../../datasets/csv/mtcars.csv"
default_output <- "../../results/r"
default_fraction <- 0.5

# Parse arguments
csv_path <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir <- ifelse(length(args) >= 2, args[2], default_output)
fraction <- as.numeric(ifelse(length(args) >= 3, args[3], default_fraction))

# Validate CSV path
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

# Extract dataset name from filename
dataset_name <- tools::file_path_sans_ext(basename(csv_path))

# Read CSV data
data <- read.csv(csv_path)

# Convert categorical columns to numeric
data <- convert_categorical_to_numeric(data, dataset_name)

# Assume first column is response variable, rest are predictors
response_col <- names(data)[1]
predictor_cols <- names(data)[-1]

# Build formula: response ~ predictor1 + predictor2 + ...
formula_str <- paste(response_col, "~", paste(predictor_cols, collapse = " + "))
formula <- as.formula(formula_str)

# Fit the model
model <- lm(formula, data = data)

# Run Rainbow test (R variant - Type 7 quantile)
# raintest parameters:
# - formula: the model formula
# - fraction: proportion of data in central subset (default: 0.5)
# - order.by: NULL uses original data order (lmtest default)
rainbow_result <- raintest(model, fraction = fraction)

# Print results
cat("Rainbow Test (R - lmtest::raintest)\n")
cat("==================================\n")
cat("Dataset:", dataset_name, "\n")
cat("Formula:", formula_str, "\n")
cat("Fraction:", fraction, "\n")
cat("Statistic (F):", rainbow_result$statistic, "\n")
cat("p-value:", rainbow_result$p.value, "\n")
cat("Passed:", rainbow_result$p.value > 0.05, "\n\n")

# Prepare output
output <- list(
  test_name = "Rainbow Test (R - lmtest::raintest)",
  dataset = dataset_name,
  formula = formula_str,
  statistic = as.numeric(rainbow_result$statistic),
  p_value = as.numeric(rainbow_result$p.value),
  passed = rainbow_result$p.value > 0.05,
  method = "R",
  fraction = fraction,
  description = "Tests for linearity by comparing fit on central subset vs full data. R variant uses lmtest::raintest algorithm with Type 7 quantile."
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_rainbow.json
output_file <- file.path(output_dir, paste0(dataset_name, "_rainbow.json"))
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
