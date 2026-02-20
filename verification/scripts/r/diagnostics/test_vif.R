# ============================================================================
# VIF (Variance Inflation Factor) Reference Implementation (R)
# ============================================================================
# This script generates reference values for VIF using R's car::vif()
#
# The VIF measures how much the variance of a regression coefficient is
# inflated due to multicollinearity among predictor variables.
#
# Source: car package, vif function
# Reference: Fox & Weisberg (2019), "An R Companion to Applied Regression"
#
# Usage:
#   Rscript test_vif.R [csv_path] [output_dir]
#   Args:
#     csv_path  - Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     output_dir- Path to output directory
#                 Default: ../../results/r
# ============================================================================

library(car)
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

# Convert categorical columns to numeric
data <- convert_categorical_to_numeric(data, dataset_name)

# Assume first column is response variable, rest are predictors
response_col <- names(data)[1]
predictor_cols <- names(data)[-1]

# VIF requires at least 2 predictors
if (length(predictor_cols) < 2) {
  cat(paste0("SKIP: Dataset '", dataset_name, "' has only ", length(predictor_cols),
             " predictor. VIF requires at least 2 predictors.\n"))
  quit(status = 0)
}

# Build formula: response ~ predictor1 + predictor2 + ...
formula_str <- paste(response_col, "~", paste(predictor_cols, collapse = " + "))
formula <- as.formula(formula_str)

# Fit the model
model <- lm(formula, data = data)

# Calculate VIF using car::vif
vif_result <- car::vif(model)

# Print results
cat("VIF Test (R - car::vif)\n")
cat("=================================\n")
cat("Dataset:", dataset_name, "\n")
cat("Formula:", formula_str, "\n")
cat("Number of predictors:", length(vif_result), "\n\n")

# Print each variable's VIF
cat("VIF Results:\n")
for (i in 1:length(vif_result)) {
  cat(sprintf("  %s: VIF = %.6f, R² = %.6f\n",
              names(vif_result)[i],
              as.numeric(vif_result[i]),
              1 - 1/as.numeric(vif_result[i])))
}

cat(sprintf("\nMax VIF: %.6f\n\n", max(as.numeric(vif_result))))

# Interpretation
max_vif <- max(as.numeric(vif_result))
if (max_vif > 10) {
  interpretation <- "Severe multicollinearity detected (VIF > 10)"
} else if (max_vif > 5) {
  interpretation <- "Moderate multicollinearity detected (VIF > 5)"
} else {
  interpretation <- "Low multicollinearity (VIF ≤ 5)"
}
cat("Interpretation:", interpretation, "\n\n")

# Prepare output
output <- list(
  test_name = "VIF Test (R - car::vif)",
  dataset = dataset_name,
  formula = formula_str,
  vif_results = as.data.frame(vif_result),
  vif_numeric = as.numeric(vif_result),
  max_vif = max_vif,
  interpretation = interpretation,
  description = "Variance Inflation Factor measures multicollinearity among predictors. Uses car::vif()."
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_vif.json
output_file <- file.path(output_dir, paste0(dataset_name, "_vif.json"))
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
