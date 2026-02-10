# ============================================================================
# Run All Diagnostic Tests (Python)
# ============================================================================
# This runner script executes all diagnostic tests against all datasets in the
# datasets/csv/ directory. Results are saved to results/python/ with naming
# convention {dataset}_{test}.json
#
# Usage:
#   python run_all_diagnostics_python.py [--csv-dir CSV_DIR] [--output-dir OUTPUT_DIR] [--script-dir SCRIPT_DIR]
#
#   Args:
#     --csv-dir    Path to directory containing CSV files
#                   Default: verification/datasets/csv
#     --output-dir Path to output directory for results
#                   Default: verification/results/python
#     --script-dir Path to directory containing diagnostic scripts
#                   Default: verification/scripts/python/diagnostics
#
# Example:
#   python run_all_diagnostics_python.py
#   python run_all_diagnostics_python.py --csv-dir verification/datasets/csv --output-dir verification/results/python
# ============================================================================

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run all diagnostic tests against all datasets"
    )
    parser.add_argument(
        "--csv-dir",
        default="verification/datasets/csv",
        help="Path to directory containing CSV files (default: verification/datasets/csv)"
    )
    parser.add_argument(
        "--output-dir",
        default="verification/results/python",
        help="Path to output directory for results (default: verification/results/python)"
    )
    parser.add_argument(
        "--script-dir",
        default="verification/scripts/python/diagnostics",
        help="Path to directory containing diagnostic scripts"
    )
    parser.add_argument(
        "--loess-dir",
        default="verification/scripts/python/loess",
        help="Path to directory containing LOESS script"
    )
    parser.add_argument(
        "--core-dir",
        default="verification/scripts/python/core",
        help="Path to directory containing core scripts (OLS, WLS)"
    )

    args = parser.parse_args()

    # Validate CSV directory
    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        print(f"Error: CSV directory not found: {args.csv_dir}", file=sys.stderr)
        sys.exit(1)

    # Get list of CSV files
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in: {args.csv_dir}", file=sys.stderr)
        sys.exit(1)

    # Core scripts to run (in core/ directory)
    core_scripts = [
        {"name": "WLS", "script": "test_wls.py", "suffix": "wls"},
    ]

    # Diagnostic scripts to run
    diagnostic_scripts = [
        {"name": "Rainbow", "script": "test_rainbow.py"},
        {"name": "Harvey-Collier", "script": "test_harvey_collier.py"},
        {"name": "Breusch-Pagan", "script": "test_breusch_pagan.py"},
        {"name": "White", "script": "test_white.py"},
        {"name": "Jarque-Bera", "script": "test_jarque_bera.py"},
        {"name": "Durbin-Watson", "script": "test_durbin_watson.py"},
        {"name": "Breusch-Godfrey", "script": "test_breusch_godfrey.py"},
        {"name": "Shapiro-Wilk", "script": "test_shapiro_wilk.py"},
        {"name": "Anderson-Darling", "script": "test_anderson_darling.py"},
        {"name": "Cooks-Distance", "script": "test_cooks_distance.py"},
        {"name": "DFBETAS", "script": "test_dfbetas.py"},
        {"name": "DFFITS", "script": "test_dffits.py"},
        {"name": "VIF", "script": "test_vif.py"},
        {"name": "RESET", "script": "test_reset.py"},
    ]

    # LOESS script (separate directory, produces 6 outputs per dataset)
    loess_dir = Path(args.loess_dir)
    loess_scripts = [
        {"name": "LOESS", "script": "test_loess.py", "suffix": "loess", "dir": loess_dir, "multi_output": True},
    ]

    script_dir = Path(args.script_dir)
    core_dir = Path(args.core_dir)

    # Counter for results
    total_tests = len(csv_files) * (len(diagnostic_scripts) + len(core_scripts) + len(loess_scripts))
    completed_tests = 0
    failed_tests = 0

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Running All Diagnostic Tests (Python)")
    print("=" * 60)
    print(f"CSV Directory: {args.csv_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Script Directory: {args.script_dir}")
    print(f"Core Directory: {args.core_dir}")
    print(f"LOESS Directory: {args.loess_dir}")
    print(f"Datasets: {len(csv_files)}")
    print(f"Core Tests: {len(core_scripts)}")
    print(f"Diagnostic Tests: {len(diagnostic_scripts)}")
    print(f"LOESS Tests: {len(loess_scripts)}")
    print(f"Total Tests to Run: {total_tests}")
    print("=" * 60)
    print()

    # Loop through each CSV file
    for csv_file in csv_files:
        dataset_name = csv_file.stem

        print(f"--- Dataset: {dataset_name} ---")

        # Run core tests (WLS, etc.)
        for core in core_scripts:
            test_name = core["name"]
            script_path = core_dir / core["script"]

            # Check if script exists
            if not script_path.exists():
                print(f"  [SKIP] {test_name} - Script not found: {script_path}")
                continue

            # Build command (core scripts use --csv and --output-dir)
            cmd = [
                sys.executable,
                str(script_path),
                "--csv", str(csv_file),
                "--output-dir", args.output_dir
            ]

            # Run test
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                # Check if output file was created
                suffix = core.get("suffix", test_name.lower().replace(' ', '_').replace('-', '_'))
                expected_output = Path(args.output_dir) / f"{dataset_name}_{suffix}.json"

                if expected_output.exists():
                    print(f"  [PASS] {test_name}")
                    completed_tests += 1
                else:
                    print(f"  [FAIL] {test_name} - Output file not created")
                    if result.stdout:
                        print(f"    stdout: {result.stdout.strip()}")
                    if result.stderr:
                        print(f"    stderr: {result.stderr.strip()}")
                    failed_tests += 1

            except subprocess.TimeoutExpired:
                print(f"  [FAIL] {test_name} - Timeout")
                failed_tests += 1
            except Exception as e:
                print(f"  [FAIL] {test_name} - {e}")
                failed_tests += 1

        # Loop through each diagnostic test
        for diag in diagnostic_scripts:
            test_name = diag["name"]
            script_path = script_dir / diag["script"]

            # Check if script exists
            if not script_path.exists():
                print(f"  [SKIP] {test_name} - Script not found: {script_path}")
                continue

            # Build command
            cmd = [
                sys.executable,
                str(script_path),
                "--csv", str(csv_file),
                "--output-dir", args.output_dir
            ]

            # Run test
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                # Check if output file was created
                expected_output = Path(args.output_dir) / f"{dataset_name}_{test_name.lower().replace(' ', '_').replace('-', '_')}.json"

                if expected_output.exists():
                    print(f"  [PASS] {test_name}")
                    completed_tests += 1
                else:
                    print(f"  [FAIL] {test_name} - Output file not created")
                    if result.stdout:
                        print(f"    stdout: {result.stdout.strip()}")
                    if result.stderr:
                        print(f"    stderr: {result.stderr.strip()}")
                    failed_tests += 1

            except subprocess.TimeoutExpired:
                print(f"  [FAIL] {test_name} - Timeout")
                failed_tests += 1
            except Exception as e:
                print(f"  [FAIL] {test_name} - {e}")
                failed_tests += 1

        # Run LOESS tests (separate directory, different command format)
        for loess in loess_scripts:
            test_name = loess["name"]
            script_path = loess["dir"] / loess["script"]

            # Check if script exists
            if not script_path.exists():
                print(f"  [SKIP] {test_name} - Script not found: {script_path}")
                continue

            # Build command (LOESS uses positional args)
            cmd = [
                sys.executable,
                str(script_path),
                str(csv_file),
                args.output_dir
            ]

            # Run test
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                # Check if output files were created (LOESS produces 3 outputs for LOWESS)
                if loess.get("multi_output", False):
                    # Python LOESS uses statsmodels.lowess (LOWESS = degree 1 only)
                    # Produces 3 outputs: 3 spans × 1 degree
                    expected_suffixes = [
                        f"{dataset_name}_loess_0.25_d1.json",
                        f"{dataset_name}_loess_0.50_d1.json",
                        f"{dataset_name}_loess_0.75_d1.json",
                    ]
                    all_found = all((Path(args.output_dir) / s).exists() for s in expected_suffixes)
                    if all_found:
                        print(f"  [PASS] {test_name} (3 outputs - LOWESS degree 1)")
                        completed_tests += 1
                    else:
                        print(f"  [FAIL] {test_name} - Some output files not created")
                        if result.stdout:
                            print(f"    stdout: {result.stdout.strip()}")
                        if result.stderr:
                            print(f"    stderr: {result.stderr.strip()}")
                        failed_tests += 1
                else:
                    # Single output file
                    expected_output = Path(args.output_dir) / f"{dataset_name}_loess_0.75_d1.json"
                    if expected_output.exists():
                        print(f"  [PASS] {test_name}")
                        completed_tests += 1
                    else:
                        print(f"  [FAIL] {test_name} - Output file not created")
                        if result.stdout:
                            print(f"    stdout: {result.stdout.strip()}")
                        if result.stderr:
                            print(f"    stderr: {result.stderr.strip()}")
                        failed_tests += 1

            except subprocess.TimeoutExpired:
                print(f"  [FAIL] {test_name} - Timeout")
                failed_tests += 1
            except Exception as e:
                print(f"  [FAIL] {test_name} - {e}")
                failed_tests += 1

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Completed: {completed_tests}")
    print(f"Failed: {failed_tests}")
    print("=" * 60)


if __name__ == "__main__":
    main()
