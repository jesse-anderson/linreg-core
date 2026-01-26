# ============================================================================
# RESET Test Reference Implementation (R)
# ============================================================================
# This script generates reference values for the RESET test using R's
# lmtest::resettest function. The RESET (Regression Specification Error Test)
# checks for functional form misspecification by testing if powers of fitted
# values or regressors significantly improve the model fit.
#
# Source: lmtest package, resettest function
# Reference: Ramsey, J.B. (1969), "Tests for Specification Errors in Classical
#           Linear Least-Squares Regression Analysis", Journal of the Royal
#           Statistical Society, Series B 31: 350–371.
#
# Usage:
#   Rscript test_reset.R [csv_path] [output_dir] [powers] [type]
#   Args:
#     csv_path  - Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     output_dir- Path to output directory
#                 Default: ../../results/r
#     powers    - Powers to use (e.g., "2:3" for squared and cubed)
#                 Default: 2:3
#     type      - Type of terms: "fitted", "regressor", or "princomp"
#                 Default: "fitted" (matches R's resettest default)
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
default_powers <- "2:3"
default_type <- "fitted"

# Parse arguments
csv_path <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir <- ifelse(length(args) >= 2, args[2], default_output)
powers_str <- ifelse(length(args) >= 3, args[3], default_powers)
type_str <- ifelse(length(args) >= 4, args[4], default_type)

# Parse powers string (e.g., "2:3" -> c(2, 3))
if (grepl(":", powers_str)) {
  power_range <- as.numeric(strsplit(powers_str, ":")[[1]])
  powers <- power_range[1]:power_range[2]
} else {
  # Comma-separated values
  powers <- as.numeric(strsplit(powers_str, ",")[[1]])
}

# Validate type
valid_types <- c("fitted", "regressor", "princomp")
if (!(type_str %in% valid_types)) {
  stop(paste("Invalid type. Must be one of:", paste(valid_types, collapse = ", ")))
}

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

# Run RESET test
# resettest parameters:
# - formula: the model formula
# - power: vector of powers to use (default: 2:3)
# - type: "fitted" (default), "regressor", or "princomp"
# - data: data frame
reset_result <- resettest(model, power = powers, type = type_str)

# Print results
cat("RESET Test (R - lmtest::resettest)\n")
cat("==================================\n")
cat("Dataset:", dataset_name, "\n")
cat("Formula:", formula_str, "\n")
cat("Powers:", paste(powers, collapse = ", "), "\n")
cat("Type:", type_str, "\n")
cat("Statistic (F):", reset_result$statistic, "\n")
cat("p-value:", reset_result$p.value, "\n")
cat("Passed:", reset_result$p.value > 0.05, "\n\n")

# Prepare output
output <- list(
  test_name = "RESET Test (R - lmtest::resettest)",
  dataset = dataset_name,
  formula = formula_str,
  statistic = as.numeric(reset_result$statistic),
  p_value = as.numeric(reset_result$p.value),
  passed = reset_result$p.value > 0.05,
  power = as.list(powers),
  type = type_str,
  description = "Ramsey's RESET test for functional form misspecification. Tests whether powers of fitted values, regressors, or principal component significantly improve model fit."
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_reset.json
output_file <- file.path(output_dir, paste0(dataset_name, "_reset.json"))
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
