# ============================================================================
# Run All Diagnostic Tests (R)
# ============================================================================
# This runner script executes all diagnostic tests against all datasets in the
# datasets/csv/ directory. Results are saved to results/r/ with naming
# convention {dataset}_{test}.json
#
# Usage:
#   Rscript run_all_diagnostics_r.R [csv_dir] [output_dir] [script_dir]
#
#   Args:
#     csv_dir    Path to directory containing CSV files
#                 Default: verification/datasets/csv
#     output_dir Path to output directory for results
#                 Default: verification/results/r
#     script_dir Path to directory containing diagnostic scripts
#                 Default: verification/scripts/r/diagnostics
#
# Example:
#   Rscript run_all_diagnostics_r.R
#   Rscript run_all_diagnostics_r.R verification/datasets/csv verification/results/r
# ============================================================================

library(jsonlite)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults (relative to repo root)
default_csv_dir <- "verification/datasets/csv"
default_output_dir <- "verification/results/r"
default_script_dir <- "verification/scripts/r/diagnostics"

csv_dir <- ifelse(length(args) >= 1, args[1], default_csv_dir)
output_dir <- ifelse(length(args) >= 2, args[2], default_output_dir)
script_dir <- ifelse(length(args) >= 3, args[3], default_script_dir)

# Validate CSV directory
if (!dir.exists(csv_dir)) {
  stop(paste("CSV directory not found:", csv_dir))
}

# Get list of CSV files
csv_files <- list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE)

if (length(csv_files) == 0) {
  stop(paste("No CSV files found in:", csv_dir))
}

# Diagnostic scripts to run
diagnostic_scripts <- list(
  list(name = "Rainbow", script = "test_rainbow.R", args = ""),
  list(name = "Harvey-Collier", script = "test_harvey_collier.R", args = ""),
  list(name = "Breusch-Pagan", script = "test_breusch_pagan.R", args = ""),
  list(name = "White", script = "test_white.R", args = ""),
  list(name = "Jarque-Bera", script = "test_jarque_bera.R", args = ""),
  list(name = "Durbin-Watson", script = "test_durbin_watson.R", args = ""),
  list(name = "Shapiro-Wilk", script = "test_shapiro_wilk.R", args = ""),
  list(name = "Anderson-Darling", script = "test_anderson_darling.R", args = ""),
  list(name = "Cooks-Distance", script = "test_cooks_distance.R", args = "")
)

# Counter for results
total_tests <- length(csv_files) * length(diagnostic_scripts)
completed_tests <- 0
failed_tests <- 0

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("========================================================\n")
cat("Running All Diagnostic Tests (R)\n")
cat("========================================================\n")
cat("CSV Directory:", csv_dir, "\n")
cat("Output Directory:", output_dir, "\n")
cat("Script Directory:", script_dir, "\n")
cat("Datasets:", length(csv_files), "\n")
cat("Diagnostic Tests:", length(diagnostic_scripts), "\n")
cat("Total Tests to Run:", total_tests, "\n")
cat("========================================================\n\n")

# Loop through each CSV file
for (csv_file in csv_files) {
  dataset_name <- tools::file_path_sans_ext(basename(csv_file))

  cat("--- Dataset:", dataset_name, "---\n")

  # Loop through each diagnostic test
  for (diag in diagnostic_scripts) {
    test_name <- diag$name
    script_path <- file.path(script_dir, diag$script)

    # Check if script exists
    if (!file.exists(script_path)) {
      cat("  [SKIP]", test_name, "- Script not found:", script_path, "\n")
      next
    }

    # Build command
    cmd <- paste(
      "Rscript",
      shQuote(script_path),
      shQuote(csv_file),
      shQuote(output_dir)
    )

    # Run test
    result <- tryCatch({
      system(cmd, intern = TRUE, ignore.stderr = TRUE)
    }, error = function(e) {
      NULL
    })

    # Check if test ran successfully
    expected_output <- file.path(output_dir, paste0(dataset_name, "_",
                                              tolower(gsub("[- ]", "_", test_name)), ".json"))

    if (file.exists(expected_output)) {
      cat("  [PASS]", test_name, "\n")
      completed_tests <- completed_tests + 1
    } else {
      cat("  [FAIL]", test_name, "- Output file not created\n")
      failed_tests <- failed_tests + 1
    }
  }

  cat("\n")
}

# Summary
cat("========================================================\n")
cat("Summary\n")
cat("========================================================\n")
cat("Total Tests:", total_tests, "\n")
cat("Completed:", completed_tests, "\n")
cat("Failed:", failed_tests, "\n")
cat("========================================================\n")
