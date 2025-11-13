#!/usr/bin/env python3
"""
Post-processing and analysis script for FSS grid simulation results.

This script:
1. Processes 30 randomized runs per configuration
2. Filters out unsuccessful/unconverged runs
3. Generates 7 time-series plots per configuration (median + IQR + STD)
4. Creates one consolidated Excel file with summary statistics

Usage:
    python analyze_grid_results.py
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing


# ===================== CONFIGURATION ========================================
RESULTS_BASE = "results_grid_run1/14102025"
OUTPUT_BASE = "results_grid_run1/14102025_analysis"
CONSOLIDATED_EXCEL = "consolidated_summary.xlsx"

# Success criteria for filtering runs
# Simply exclude files marked as UNSOLVED by the simulator

# Plot styling
PLOT_DPI = 150
PLOT_WIDTH = 10
PLOT_HEIGHT = 6
# ============================================================================


def parse_folder_name(folder_name: str) -> Optional[Dict[str, any]]:
    """
    Extract configuration parameters from folder name.

    Example: S1000_CN100_OMNICN0_Cplanet_sso
    Returns: {num_satellites: 1000, fraction_CNs: 1.0, omni_fraction_CNs: 0.0,
              k_var: 5.0, constellation: 'planet_sso'}
    """
    pattern = r'S(\d+)_CN(\d+)_OMNICN(\d+)_C(.+)'
    match = re.match(pattern, folder_name)

    if not match:
        return None

    num_sats = int(match.group(1))
    num_cns = int(match.group(2))
    num_omni_cns = int(match.group(3))
    frac_cns = num_cns / num_sats
    omni_frac = (num_omni_cns / num_cns) if num_cns > 0 else 0.0
    constellation = match.group(4)

    # k_var is not in folder name, need to extract from file names
    # For now, return None and extract from first file

    return {
        'num_satellites': num_sats,
        'fraction_CNs': frac_cns,
        'omni_fraction_CNs': omni_frac,
        'constellation': constellation,
        'k_var': None  # Will be filled from file data
    }


def extract_k_var_from_filename(filename: str) -> Optional[float]:
    """Extract k_var from Excel filename."""
    pattern = r'_k([\d.]+)_'
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    return None


def is_run_successful(filename: str) -> bool:
    """
    Determine if a run successfully converged.

    Simply checks if the filename starts with "UNSOLVED" - these are marked
    as failed by the simulator itself.
    """
    return not filename.upper().startswith("UNSOLVED")


def load_configuration_runs(config_folder: Path, num_satellites: int) -> Tuple[List[pd.DataFrame], int, int]:
    """
    Load all Excel files from a configuration folder.

    Returns:
        - List of successful run dataframes
        - Total runs found
        - Successful runs count
    """
    excel_files = list(config_folder.glob("*.xlsx"))

    successful_runs = []
    total_runs = len(excel_files)

    for excel_file in excel_files:
        # Skip UNSOLVED files
        if not is_run_successful(excel_file.name):
            continue

        try:
            df = pd.read_excel(excel_file, sheet_name='task_0')
            successful_runs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {excel_file.name}: {e}")
            continue

    return successful_runs, total_runs, len(successful_runs)


def align_time_series(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Align time series to common length (use minimum length across all runs).
    """
    if not dfs:
        return []

    min_length = min(len(df) for df in dfs)
    return [df.iloc[:min_length].reset_index(drop=True) for df in dfs]


def compute_time_series_stats(dfs: List[pd.DataFrame], column: str) -> Dict[str, np.ndarray]:
    """
    Compute median, IQR, and STD for a time series column across runs.

    Returns:
        Dictionary with keys: 'median', 'q1', 'q3', 'std', 'mean'
    """
    if not dfs:
        return None

    # Stack all runs into a 2D array (runs x timesteps)
    data = np.array([df[column].values for df in dfs])

    # Compute statistics across runs (axis=0)
    median = np.median(data, axis=0)
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)

    return {
        'median': median,
        'q1': q1,
        'q3': q3,
        'std': std,
        'mean': mean,
        'timesteps': np.arange(len(median))
    }


def plot_time_series_metric(stats: Dict[str, np.ndarray],
                            metric_name: str,
                            ylabel: str,
                            output_path: Path,
                            config_name: str):
    """
    Create a time-series plot with median line, IQR shading, and STD error bars.
    """
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    timesteps = stats['timesteps']
    median = stats['median']
    q1 = stats['q1']
    q3 = stats['q3']
    std = stats['std']

    # Plot median line
    ax.plot(timesteps, median, 'b-', linewidth=2, label='Median')

    # Plot IQR as shaded region
    ax.fill_between(timesteps, q1, q3, alpha=0.3, color='blue', label='IQR (Q1-Q3)')

    # Plot STD as error bars (sample every N points to avoid clutter)
    sample_rate = max(1, len(timesteps) // 20)  # Show ~20 error bars
    sampled_indices = np.arange(0, len(timesteps), sample_rate)

    ax.errorbar(timesteps[sampled_indices], median[sampled_indices],
                yerr=std[sampled_indices], fmt='none', ecolor='red',
                alpha=0.5, capsize=3, capthick=1, label='STD')

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{metric_name}\n{config_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def process_configuration(config_folder: Path, output_folder: Path) -> Optional[Dict]:
    """
    Process one configuration folder:
    - Load successful runs
    - Generate 7 plots
    - Return summary statistics for Excel file
    """
    config_name = config_folder.name
    print(f"\nProcessing: {config_name}")

    # Parse configuration parameters
    params = parse_folder_name(config_name)
    if params is None:
        print(f"  Error: Could not parse folder name")
        return None

    num_satellites = params['num_satellites']

    # Load runs
    successful_dfs, total_runs, success_count = load_configuration_runs(config_folder, num_satellites)

    print(f"  Runs: {success_count}/{total_runs} successful")

    if success_count == 0:
        print(f"  Skipping: No successful runs")
        return None

    # Extract k_var from first file
    first_file = list(config_folder.glob("*.xlsx"))[0]
    params['k_var'] = extract_k_var_from_filename(first_file.name)

    # Align time series
    aligned_dfs = align_time_series(successful_dfs)

    # Create output directory for this configuration
    config_output = output_folder / config_name
    config_output.mkdir(parents=True, exist_ok=True)

    # Metrics to plot (with labels)
    metrics = [
        ('winner_RF', 'Winner RF', 'RF Value'),
        ('runner_up_RF', 'Runner-up RF', 'RF Value'),
        ('knowledge', 'Knowledge Propagation', 'Number of Satellites'),
        ('thinks_winner', 'Thinks Winner Count', 'Number of Satellites'),
        ('winner_bid', 'Winner Bid', 'Bid Value'),
        ('runner_up_bid', 'Runner-up Bid', 'Bid Value'),
        ('RF_time', 'Task Execution Time', 'Timesteps'),
    ]

    # Generate plots and collect final values for summary
    summary_stats = {
        'config_name': config_name,
        'num_satellites': params['num_satellites'],
        'fraction_CNs': params['fraction_CNs'],
        'omni_fraction_CNs': params['omni_fraction_CNs'],
        'k_var': params['k_var'],
        'constellation': params['constellation'],
        'successful_runs': success_count,
        'total_runs': total_runs,
    }

    for column, metric_name, ylabel in metrics:
        # Compute time-series statistics
        stats = compute_time_series_stats(aligned_dfs, column)

        # Plot
        plot_path = config_output / f"{column}_timeseries.png"
        plot_time_series_metric(stats, metric_name, ylabel, plot_path, config_name)

        # Extract final values for summary Excel
        final_values = np.array([df[column].iloc[-1] for df in aligned_dfs])

        summary_stats[f'{column}_median'] = np.median(final_values)
        summary_stats[f'{column}_mean'] = np.mean(final_values)
        summary_stats[f'{column}_std'] = np.std(final_values)
        summary_stats[f'{column}_q1'] = np.percentile(final_values, 25)
        summary_stats[f'{column}_q3'] = np.percentile(final_values, 75)

    # Add duration statistics (use ORIGINAL successful_dfs, not aligned!)
    durations = np.array([len(df) for df in successful_dfs])
    summary_stats['duration_median'] = np.median(durations)
    summary_stats['duration_mean'] = np.mean(durations)
    summary_stats['duration_std'] = np.std(durations)
    summary_stats['duration_q1'] = np.percentile(durations, 25)
    summary_stats['duration_q3'] = np.percentile(durations, 75)

    # Plot duration as a special case (histogram instead of time series)
    plot_duration_distribution(durations, config_output / "duration_distribution.png", config_name)

    print(f"  [OK] Generated 7 plots in {config_output}")

    return summary_stats


def plot_duration_distribution(durations: np.ndarray, output_path: Path, config_name: str):
    """Plot histogram of simulation durations."""
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    ax.hist(durations, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.median(durations), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(durations):.0f}')
    ax.axvline(np.mean(durations), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(durations):.1f}')

    ax.set_xlabel('Duration (timesteps)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Simulation Duration Distribution\n{config_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("FSS Grid Results Post-Processing and Analysis")
    print("="*80)

    results_base = Path(RESULTS_BASE)
    output_base = Path(OUTPUT_BASE)
    output_base.mkdir(parents=True, exist_ok=True)

    # Get all configuration folders
    config_folders = [f for f in results_base.iterdir() if f.is_dir()]
    config_folders.sort()

    print(f"\nFound {len(config_folders)} configuration folders")
    print(f"Output directory: {output_base}")

    # Process each configuration
    all_summaries = []

    for i, config_folder in enumerate(config_folders, 1):
        print(f"\n[{i}/{len(config_folders)}] ", end="")

        summary = process_configuration(config_folder, output_base)

        if summary is not None:
            all_summaries.append(summary)

    # Create consolidated Excel file
    if all_summaries:
        print(f"\n{'='*80}")
        print(f"Creating consolidated Excel file...")

        summary_df = pd.DataFrame(all_summaries)

        # Reorder columns for better readability
        column_order = [
            'config_name', 'num_satellites', 'fraction_CNs', 'omni_fraction_CNs',
            'k_var', 'constellation', 'successful_runs', 'total_runs',
            'duration_median', 'duration_mean', 'duration_std', 'duration_q1', 'duration_q3',
        ]

        # Add metric columns
        metrics = ['winner_RF', 'runner_up_RF', 'knowledge', 'thinks_winner', 'winner_bid', 'runner_up_bid', 'RF_time']
        for metric in metrics:
            column_order.extend([
                f'{metric}_median', f'{metric}_mean', f'{metric}_std', f'{metric}_q1', f'{metric}_q3'
            ])

        summary_df = summary_df[column_order]

        # Save intermediate JSON backup (in case Excel write fails)
        import json
        json_path = output_base / "aggregated_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"[OK] Saved backup: {json_path}")

        # Save to Excel
        excel_path = output_base / CONSOLIDATED_EXCEL
        try:
            summary_df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"[OK] Saved: {excel_path}")
            print(f"  Rows: {len(summary_df)}")
            print(f"  Columns: {len(summary_df.columns)}")
        except PermissionError:
            print(f"\n[WARNING] Could not write to {excel_path}")
            print(f"  The file may be open in Excel. Please close it and run:")
            print(f"  python regenerate_consolidated_summary.py")
        except Exception as e:
            print(f"\n[ERROR] Failed to write Excel file: {e}")
            print(f"  You can recover from the JSON backup: {json_path}")
    else:
        print("\nNo successful configurations processed!")

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
