"""
Temporal Dynamics Analysis - Convergence Times

This script analyzes when key metrics converge across ALL simulation runs.

Convergence criteria:
- winner_RF: When it reaches its MAXIMUM value within the run
- knowledge: When it reaches N_total_satellites (full propagation)
- thinks_winner: When it reaches 1 (full consensus)
- runner_up_bid: When it reaches its MINIMUM value within the run

Output: Normalized convergence times (median, mean, std) for each metric.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ===================== CONFIGURATION ========================================
RESULTS_BASE = "results_grid_run1/14102025"
OUTPUT_FILE = "temporal_dynamics_results.csv"
# ============================================================================


def parse_folder_name(folder_name: str) -> Optional[Dict[str, any]]:
    """Extract configuration parameters from folder name."""
    pattern = r'S(\d+)_CN(\d+)_OMNICN(\d+)_C(.+)'
    match = re.match(pattern, folder_name)

    if not match:
        return None

    num_sats = int(match.group(1))
    num_cns = int(match.group(2))
    num_omni_cns = int(match.group(3))
    constellation = match.group(4)

    return {
        'num_satellites': num_sats,
        'num_cns': num_cns,
        'omni_fraction_CNs': num_omni_cns / num_cns if num_cns > 0 else 0.0,
        'constellation': constellation,
    }


def find_convergence_timestep(df: pd.DataFrame, metric: str,
                               num_satellites: int) -> float:
    """
    Find when a metric converges and return normalized time.

    Args:
        df: DataFrame with time series data
        metric: Name of the metric to analyze
        num_satellites: Total number of satellites in the configuration

    Returns:
        Normalized convergence time (0 to 1), where 1 = full duration
    """
    if metric not in df.columns or len(df) < 2:
        return np.nan

    total_timesteps = len(df) - 1
    values = df[metric].values

    # Define convergence criteria based on metric
    if metric == 'winner_RF':
        # Find when it reaches maximum value
        max_value = values.max()
        if np.isnan(max_value) or max_value == 0:
            return np.nan

        # First timestep where it reaches max
        convergence_idx = np.where(values >= max_value * 0.9999)[0]
        if len(convergence_idx) == 0:
            return 1.0
        return convergence_idx[0] / total_timesteps

    elif metric == 'knowledge':
        # Find when it reaches N_total_satellites
        target_value = num_satellites

        # First timestep where knowledge >= target (within 0.1% tolerance)
        convergence_idx = np.where(values >= target_value * 0.999)[0]
        if len(convergence_idx) == 0:
            return 1.0
        return convergence_idx[0] / total_timesteps

    elif metric == 'thinks_winner':
        # Find when it reaches 1 (full consensus)
        target_value = 1

        # First timestep where thinks_winner == 1
        convergence_idx = np.where(values == target_value)[0]
        if len(convergence_idx) == 0:
            return 1.0
        return convergence_idx[0] / total_timesteps

    elif metric == 'runner_up_bid':
        # Find when it reaches minimum value
        # Exclude initial timesteps where it might be infinity or zero
        valid_values = values[values > 0]
        valid_values = valid_values[np.isfinite(valid_values)]

        if len(valid_values) == 0:
            return np.nan

        min_value = valid_values.min()

        # First timestep where it reaches min (within 0.1% tolerance)
        convergence_idx = np.where(np.abs(values - min_value) / (min_value + 1e-10) < 0.001)[0]
        if len(convergence_idx) == 0:
            return 1.0
        return convergence_idx[0] / total_timesteps

    else:
        return np.nan


def analyze_single_run(excel_file: Path, num_satellites: int) -> Optional[Dict]:
    """
    Analyze convergence times for a single simulation run.

    Args:
        excel_file: Path to Excel file
        num_satellites: Number of satellites in this configuration

    Returns:
        Dict with convergence times for each metric, or None if failed
    """
    try:
        df = pd.read_excel(excel_file, sheet_name='task_0')

        if len(df) < 2:
            return None

        # Metrics to analyze
        metrics = ['winner_RF', 'knowledge', 'thinks_winner', 'runner_up_bid']

        result = {
            'file': excel_file.name,
            'duration_timesteps': len(df) - 1
        }

        for metric in metrics:
            conv_time = find_convergence_timestep(df, metric, num_satellites)
            result[f'{metric}_convergence'] = conv_time

        return result

    except Exception as e:
        return None


def analyze_configuration(config_folder: Path) -> List[Dict]:
    """Analyze all successful runs in a configuration folder."""
    config_name = config_folder.name

    params = parse_folder_name(config_name)
    if params is None:
        return []

    num_satellites = params['num_satellites']

    # Find all Excel files (excluding UNSOLVED)
    excel_files = [f for f in config_folder.glob("*.xlsx")
                   if not f.name.upper().startswith("UNSOLVED")]

    results = []

    for excel_file in excel_files:
        run_result = analyze_single_run(excel_file, num_satellites)

        if run_result is not None:
            # Add configuration metadata
            run_result.update({
                'config_name': config_name,
                'num_satellites': num_satellites,
                'num_cns': params['num_cns'],
                'omni_fraction_CNs': params['omni_fraction_CNs'],
                'constellation': params['constellation']
            })
            results.append(run_result)

    return results


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("TEMPORAL DYNAMICS ANALYSIS - CONVERGENCE TIMES")
    print("="*80)
    print()
    print("Analyzing when key metrics converge across all simulation runs...")
    print()
    print("Convergence criteria:")
    print("  - winner_RF:      Maximum value reached")
    print("  - knowledge:      Reaches N_total_satellites (full propagation)")
    print("  - thinks_winner:  Reaches 1 (full consensus)")
    print("  - runner_up_bid:  Minimum value reached")
    print()
    print("="*80)

    results_base = Path(RESULTS_BASE)

    # Find all configuration folders
    config_folders = sorted([f for f in results_base.iterdir() if f.is_dir()])

    print(f"\nFound {len(config_folders)} configuration folders")
    print(f"Starting analysis...\n")

    all_results = []
    total_runs_analyzed = 0

    for i, config_folder in enumerate(config_folders, 1):
        print(f"[{i:3d}/{len(config_folders)}] Processing {config_folder.name:60s}", end=' ')

        config_results = analyze_configuration(config_folder)

        if len(config_results) > 0:
            all_results.extend(config_results)
            total_runs_analyzed += len(config_results)
            print(f"OK ({len(config_results):2d} runs)")
        else:
            print("SKIP (no data)")

    if len(all_results) == 0:
        print("\n[ERROR] No results computed!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save detailed results
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal simulation runs analyzed: {total_runs_analyzed}")
    print(f"Detailed results saved to: {OUTPUT_FILE}")

    # Compute summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY: CONVERGENCE TIMES (% of total simulation duration)")
    print(f"{'='*80}\n")

    metrics = ['winner_RF', 'knowledge', 'thinks_winner', 'runner_up_bid']

    print(f"{'Metric':<20} {'Median':>10} {'Mean':>10} {'Std':>10} {'Valid':>10}")
    print("-"*80)

    results_summary = []

    for metric in metrics:
        col_name = f'{metric}_convergence'

        # Filter out NaN values
        valid_data = df[col_name].dropna()

        if len(valid_data) > 0:
            median_val = valid_data.median()
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            count = len(valid_data)

            print(f"{metric:<20} {median_val:>9.1%} {mean_val:>9.1%} {std_val:>9.1%} {count:>10}")

            results_summary.append({
                'metric': metric,
                'median': median_val,
                'mean': mean_val,
                'std': std_val,
                'count': count
            })
        else:
            print(f"{metric:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'0':>10}")

    print(f"\n{'='*80}")
    print("CONVERGENCE HIERARCHY (sorted by median)")
    print(f"{'='*80}\n")

    # Sort by median
    results_summary_df = pd.DataFrame(results_summary).sort_values('median')

    for i, row in results_summary_df.iterrows():
        print(f"{i+1}. {row['metric']:<20} converges at {row['median']:>6.1%} "
              f"(mean: {row['mean']:>6.1%}, std: {row['std']:>6.1%})")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")

    if len(results_summary_df) > 1:
        first = results_summary_df.iloc[0]
        last = results_summary_df.iloc[-1]

        print(f">> FIRST to converge: {first['metric']} at {first['median']:.1%}")
        print(f">> LAST to converge:  {last['metric']} at {last['median']:.1%}")
        print()

        ratio = last['median'] / first['median'] if first['median'] > 0 else 0
        print(f">> {last['metric']} converges {ratio:.1f}x slower than {first['metric']}")

    # Additional analysis: by configuration parameters
    print(f"\n{'='*80}")
    print("CONVERGENCE BY CONFIGURATION PARAMETERS")
    print(f"{'='*80}\n")

    # Group by number of satellites
    print("By number of satellites:")
    print("-"*80)
    for metric in ['winner_RF', 'knowledge', 'thinks_winner']:
        col_name = f'{metric}_convergence'
        grouped = df.groupby('num_satellites')[col_name].median()
        print(f"\n{metric}:")
        for n_sats, med_val in grouped.items():
            print(f"  S{n_sats:4d}: {med_val:.1%}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
