# ============================================================================
# Breusch-Godfrey Test Reference Implementation (R)
# ============================================================================
# This script generates reference values for the Breusch-Godfrey test using R's
# lmtest::bgtest function. The Breusch-Godfrey test checks for higher-order
# serial correlation in residuals.
#
# Source: lmtest package, bgtest function
# Reference: Breusch, T.S. (1978). Testing for Autocorrelation in Dynamic Linear
#            Models, Australian Economic Papers, 17, 334-355.
#            Godfrey, L.G. (1978). Testing Against General Autoregressive and
#            Moving Average Error Models when the Regressors Include Lagged
#            Dependent Variables, Econometrica, 46, 1293-1301.
#
# Usage:
#   Rscript test_breusch_godfrey.R [csv_path] [output_dir] [order]
#   Args:
#     csv_path  - Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/synthetic_autocorrelated.csv
#     output_dir- Path to output directory
#                 Default: ../../results/r
#     order     - Order of serial correlation to test (default: 1)
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
default_csv <- "../../datasets/csv/synthetic_autocorrelated.csv"
default_output <- "../../results/r"
default_order <- 1

# Parse arguments
csv_path <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir <- ifelse(length(args) >= 2, args[2], default_output)
order <- ifelse(length(args) >= 3, as.integer(args[3]), default_order)

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

# Run Breusch-Godfrey test with Chi-squared statistic (default, asymptotic)
bg_chisq <- bgtest(model, order = order, type = "Chisq")

# Run Breusch-Godfrey test with F statistic (finite sample version)
bg_f <- bgtest(model, order = order, type = "F")

# Determine interpretation based on p-value
alpha <- 0.05
if (bg_chisq$p.value > alpha) {
  interpretation <- paste0("No significant serial correlation detected up to order ", order)
} else {
  interpretation <- paste0("Significant serial correlation detected at order <= ", order)
}

# Print results
cat("Breusch-Godfrey Test (R - lmtest::bgtest)\n")
cat("=========================================\n")
cat("Dataset:", dataset_name, "\n")
cat("Formula:", formula_str, "\n")
cat("Order:", order, "\n")
cat("\nChi-squared (LM) statistic:", bg_chisq$statistic, "\n")
cat("  p-value:", bg_chisq$p.value, "\n")
cat("  df:", bg_chisq$parameter[1], "\n")
cat("\nF statistic:", bg_f$statistic, "\n")
cat("  p-value:", bg_f$p.value, "\n")
cat("  df:", paste(bg_f$parameter, collapse = ", "), "\n")
cat("\nInterpretation:", interpretation, "\n")
cat("Passed (p > 0.05):", bg_chisq$p.value > alpha, "\n\n")

# Prepare output - we output the Chi-squared version as primary
# but include F statistic values for reference
output <- list(
  test_name = "Breusch-Godfrey Test (R - lmtest::bgtest)",
  dataset = dataset_name,
  formula = formula_str,
  order = as.numeric(order),
  statistic = as.numeric(bg_chisq$statistic),
  p_value = as.numeric(bg_chisq$p.value),
  test_type = "Chisq",
  df = as.numeric(bg_chisq$parameter),
  f_statistic = as.numeric(bg_f$statistic),
  f_p_value = as.numeric(bg_f$p.value),
  f_df = as.numeric(bg_f$parameter),
  passed = bg_chisq$p.value > alpha,
  interpretation = interpretation,
  description = paste0(
    "Tests for serial correlation up to order ", order, ". ",
    "The LM (Chi-squared) statistic is asymptotically distributed as chi-squared with ", order, " degrees of freedom. ",
    "The F statistic provides a finite-sample correction. ",
    "Uses lmtest::bgtest."
  )
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_breusch_godfrey.json
output_file <- file.path(output_dir, paste0(dataset_name, "_breusch_godfrey.json"))
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
