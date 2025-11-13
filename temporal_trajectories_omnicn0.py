"""
Temporal Trajectory Analysis for OMNICN=0 Configurations

Analyzes how metrics evolve over simulation time to understand:
1. Learning curves and convergence patterns
2. Performance trajectory differences between successful configs
3. Comparison of Top 10 vs Bottom 10 configurations
4. Metric volatility and stability over time
5. Phase transitions (exploration → exploitation)

Key insights:
- Do better configs converge faster or just reach better final values?
- Are there distinct behavioral patterns/clusters?
- When do metrics stabilize?
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
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

# ===================== CONFIGURATION ========================================
RESULTS_BASE = "results_grid_run1/14102025"
CONSOLIDATED_SUMMARY = "results_grid_run1/14102025_analysis/consolidated_summary.xlsx"
OUTPUT_DIR = "results_grid_run1/temporal_analysis_omnicn0"

# Filter for OMNICN=0 only
FILTER_OMNICN0_ONLY = True

# Interpolation settings
TARGET_TIMESTEPS = 500  # Interpolate all trajectories to this many points
CONFIDENCE_LEVEL = 0.95  # For confidence intervals
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
    frac_cns = num_cns / num_sats
    omni_frac = num_omni_cns / num_cns if num_cns > 0 else 0.0
    constellation = match.group(4)

    return {
        'num_satellites': num_sats,
        'fraction_CNs': frac_cns,
        'omni_fraction_CNs': omni_frac,
        'constellation': constellation,
    }


def load_successful_runs(config_folder: Path) -> List[pd.DataFrame]:
    """Load all successful (non-UNSOLVED) runs from a configuration folder."""
    excel_files = list(config_folder.glob("*.xlsx"))

    successful_runs = []

    for excel_file in excel_files:
        # Skip UNSOLVED files
        if excel_file.name.upper().startswith("UNSOLVED"):
            continue

        try:
            df = pd.read_excel(excel_file, sheet_name='task_0')
            successful_runs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {excel_file.name}: {e}")
            continue

    return successful_runs


def interpolate_trajectory(df: pd.DataFrame, column: str, target_points: int = TARGET_TIMESTEPS) -> np.ndarray:
    """
    Interpolate a time series to a fixed number of points.

    This allows comparing trajectories of different lengths.
    """
    timesteps = df['time_step'].values
    values = df[column].values

    # Handle constant or near-constant trajectories
    if len(np.unique(values)) == 1:
        return np.full(target_points, values[0])

    # Create interpolation function
    try:
        # Use linear interpolation
        f = interp1d(timesteps, values, kind='linear', bounds_error=False, fill_value=(values[0], values[-1]))

        # Generate evenly spaced timesteps
        new_timesteps = np.linspace(timesteps[0], timesteps[-1], target_points)

        return f(new_timesteps)
    except Exception as e:
        # Fallback: return original values padded/truncated
        if len(values) > target_points:
            indices = np.linspace(0, len(values)-1, target_points, dtype=int)
            return values[indices]
        else:
            return np.pad(values, (0, target_points - len(values)), mode='edge')


def compute_trajectory_statistics(trajectories: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute statistics (median, mean, CI) across multiple trajectories.

    Args:
        trajectories: List of interpolated trajectories (all same length)

    Returns:
        Dict with 'median', 'mean', 'ci_lower', 'ci_upper', 'std'
    """
    if len(trajectories) == 0:
        return None

    # Stack into 2D array (runs × timesteps)
    data = np.array(trajectories)

    median = np.median(data, axis=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Confidence intervals
    n = len(trajectories)
    sem = std / np.sqrt(n)  # Standard error of the mean
    ci_offset = 1.96 * sem  # 95% CI
    ci_lower = mean - ci_offset
    ci_upper = mean + ci_offset

    # Percentiles
    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)

    return {
        'median': median,
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'q25': q25,
        'q75': q75,
        'timesteps': np.linspace(0, 1, len(median))  # Normalized time [0, 1]
    }


def load_and_process_configuration(config_folder: Path) -> Optional[Dict]:
    """Load a configuration and compute trajectory statistics for all metrics."""
    config_name = config_folder.name

    params = parse_folder_name(config_name)
    if params is None:
        return None

    # Load successful runs
    successful_runs = load_successful_runs(config_folder)

    if len(successful_runs) == 0:
        return None

    # Metrics to analyze
    metrics = ['winner_RF', 'runner_up_RF', 'knowledge', 'thinks_winner',
               'winner_bid', 'runner_up_bid', 'RF_time']

    trajectory_stats = {}

    for metric in metrics:
        # Interpolate all runs to same length
        interpolated = []
        for df in successful_runs:
            if metric in df.columns:
                traj = interpolate_trajectory(df, metric)
                interpolated.append(traj)

        if len(interpolated) > 0:
            stats = compute_trajectory_statistics(interpolated)
            trajectory_stats[metric] = stats

    return {
        'config_name': config_name,
        'params': params,
        'num_successful_runs': len(successful_runs),
        'trajectories': trajectory_stats
    }


def plot_metric_trajectories_comparison(configs_data: List[Dict],
                                        metric: str,
                                        output_path: Path,
                                        title_suffix: str = "",
                                        max_configs: int = 10):
    """
    Plot trajectories for a specific metric across multiple configurations.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(configs_data)))

    for i, config_data in enumerate(configs_data[:max_configs]):
        config_name = config_data['config_name']

        if metric not in config_data['trajectories']:
            continue

        stats = config_data['trajectories'][metric]

        timesteps_norm = stats['timesteps']
        median = stats['median']
        ci_lower = stats['ci_lower']
        ci_upper = stats['ci_upper']

        # Plot median line
        ax.plot(timesteps_norm, median, color=colors[i], linewidth=2,
                label=f"{config_name[:40]}..." if len(config_name) > 40 else config_name)

        # Plot confidence interval
        ax.fill_between(timesteps_norm, ci_lower, ci_upper,
                        color=colors[i], alpha=0.2)

    ax.set_xlabel('Normalized Time (0=start, 1=convergence)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Trajectories: {title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_vs_bottom_comparison(top_configs: List[Dict],
                                   bottom_configs: List[Dict],
                                   metric: str,
                                   output_path: Path):
    """
    Compare Top 10 vs Bottom 10 configurations for a specific metric.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Aggregate Top 10
    top_trajectories = []
    for config_data in top_configs:
        if metric in config_data['trajectories']:
            top_trajectories.append(config_data['trajectories'][metric]['median'])

    if len(top_trajectories) > 0:
        top_stats = compute_trajectory_statistics(top_trajectories)

        timesteps = top_stats['timesteps']
        ax.plot(timesteps, top_stats['mean'], 'g-', linewidth=3, label='Top 10 Configs (mean)')
        ax.fill_between(timesteps, top_stats['ci_lower'], top_stats['ci_upper'],
                        color='green', alpha=0.2, label='Top 10 CI')

    # Aggregate Bottom 10
    bottom_trajectories = []
    for config_data in bottom_configs:
        if metric in config_data['trajectories']:
            bottom_trajectories.append(config_data['trajectories'][metric]['median'])

    if len(bottom_trajectories) > 0:
        bottom_stats = compute_trajectory_statistics(bottom_trajectories)

        timesteps = bottom_stats['timesteps']
        ax.plot(timesteps, bottom_stats['mean'], 'r-', linewidth=3, label='Bottom 10 Configs (mean)')
        ax.fill_between(timesteps, bottom_stats['ci_lower'], bottom_stats['ci_upper'],
                        color='red', alpha=0.2, label='Bottom 10 CI')

    ax.set_xlabel('Normalized Time (0=start, 1=convergence)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric}: Top 10 vs Bottom 10 Configurations', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_metrics_single_config(config_data: Dict, output_path: Path):
    """
    Create a dashboard showing all metric trajectories for a single configuration.
    """
    metrics = ['winner_RF', 'runner_up_RF', 'knowledge', 'thinks_winner',
               'winner_bid', 'runner_up_bid', 'RF_time']

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    config_name = config_data['config_name']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        if metric not in config_data['trajectories']:
            ax.text(0.5, 0.5, f'{metric}\n(no data)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(metric, fontsize=11, fontweight='bold')
            continue

        stats = config_data['trajectories'][metric]

        timesteps = stats['timesteps']
        median = stats['median']
        q25 = stats['q25']
        q75 = stats['q75']

        ax.plot(timesteps, median, 'b-', linewidth=2, label='Median')
        ax.fill_between(timesteps, q25, q75, alpha=0.3, color='blue', label='IQR')

        ax.set_xlabel('Normalized Time', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Metric Trajectories: {config_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_convergence_time(trajectory: np.ndarray, threshold: float = 0.01) -> float:
    """
    Estimate when a trajectory converges (stabilizes).

    Returns normalized time (0 to 1) when the rolling coefficient of variation
    drops below threshold.
    """
    if len(trajectory) < 20:
        return 1.0  # Assume it takes full duration

    # Compute rolling mean and std
    window = min(50, len(trajectory) // 10)

    rolling_mean = pd.Series(trajectory).rolling(window).mean()
    rolling_std = pd.Series(trajectory).rolling(window).std()

    # Coefficient of variation
    cv = rolling_std / (rolling_mean + 1e-10)

    # Find first point where CV stays below threshold
    converged = cv < threshold
    if converged.any():
        first_converged_idx = converged.idxmax()
        return first_converged_idx / len(trajectory)

    return 1.0


def analyze_convergence_times(configs_data: List[Dict], output_dir: Path):
    """Analyze and compare convergence times across configurations."""

    convergence_data = []

    for config_data in configs_data:
        config_name = config_data['config_name']

        # Analyze convergence for each metric
        for metric in ['winner_RF', 'knowledge', 'thinks_winner']:
            if metric in config_data['trajectories']:
                median_traj = config_data['trajectories'][metric]['median']
                conv_time = compute_convergence_time(median_traj)

                convergence_data.append({
                    'config_name': config_name,
                    'metric': metric,
                    'convergence_time_normalized': conv_time,
                    **config_data['params']
                })

    conv_df = pd.DataFrame(convergence_data)
    conv_df.to_csv(output_dir / 'convergence_times.csv', index=False)

    # Visualize convergence times
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['winner_RF', 'knowledge', 'thinks_winner']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        subset = conv_df[conv_df['metric'] == metric]

        if len(subset) > 0:
            ax.hist(subset['convergence_time_normalized'], bins=20,
                   color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(subset['convergence_time_normalized'].median(),
                      color='red', linestyle='--', linewidth=2,
                      label=f"Median: {subset['convergence_time_normalized'].median():.2f}")
            ax.set_xlabel('Normalized Convergence Time', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Convergence Time Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_convergence_times.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Convergence analysis saved")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("TEMPORAL TRAJECTORY ANALYSIS - OMNICN=0 CONFIGURATIONS")
    print("="*80)

    results_base = Path(RESULTS_BASE)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load consolidated summary to identify top/bottom configs
    print("\n[1/6] Loading consolidated summary...")
    try:
        summary_df = pd.read_excel(CONSOLIDATED_SUMMARY)

        # Filter for OMNICN=0 only
        if FILTER_OMNICN0_ONLY:
            summary_df = summary_df[summary_df['omni_fraction_CNs'] == 0.0]

        # Sort by successful_runs
        summary_df = summary_df.sort_values('successful_runs', ascending=False)

        top_10_configs = summary_df.head(10)['config_name'].tolist()
        bottom_10_configs = summary_df.tail(10)['config_name'].tolist()

        print(f"  Identified {len(summary_df)} configurations")
        print(f"  Top 10 configs: {top_10_configs[0]} ... {top_10_configs[-1]}")
        print(f"  Bottom 10 configs: {bottom_10_configs[0]} ... {bottom_10_configs[-1]}")
    except Exception as e:
        print(f"  Warning: Could not load consolidated summary: {e}")
        print(f"  Will analyze all configurations without ranking")
        top_10_configs = []
        bottom_10_configs = []

    # Load all configuration folders
    print("\n[2/6] Processing configuration trajectories...")
    config_folders = [f for f in results_base.iterdir() if f.is_dir()]

    if FILTER_OMNICN0_ONLY:
        config_folders = [f for f in config_folders if 'OMNICN0' in f.name]

    all_configs_data = []

    for i, config_folder in enumerate(config_folders, 1):
        print(f"  [{i}/{len(config_folders)}] Processing {config_folder.name}...", end=' ')

        config_data = load_and_process_configuration(config_folder)

        if config_data is not None:
            all_configs_data.append(config_data)
            print(f"OK ({config_data['num_successful_runs']} runs)")
        else:
            print("SKIP")

    if len(all_configs_data) == 0:
        print("\n[ERROR] No configurations processed!")
        return

    print(f"\n  Total configurations processed: {len(all_configs_data)}")

    # Separate top and bottom configs
    top_configs_data = [c for c in all_configs_data if c['config_name'] in top_10_configs]
    bottom_configs_data = [c for c in all_configs_data if c['config_name'] in bottom_10_configs]

    # Create visualizations
    print("\n[3/6] Creating Top 10 trajectory plots...")
    metrics = ['winner_RF', 'runner_up_RF', 'knowledge', 'thinks_winner',
               'winner_bid', 'runner_up_bid', 'RF_time']

    if len(top_configs_data) > 0:
        for metric in metrics:
            output_path = output_dir / f'top10_{metric}_trajectories.png'
            plot_metric_trajectories_comparison(top_configs_data, metric, output_path,
                                                "Top 10 Configurations", max_configs=10)
            print(f"  [OK] top10_{metric}_trajectories.png")

    print("\n[4/6] Creating Bottom 10 trajectory plots...")
    if len(bottom_configs_data) > 0:
        for metric in metrics:
            output_path = output_dir / f'bottom10_{metric}_trajectories.png'
            plot_metric_trajectories_comparison(bottom_configs_data, metric, output_path,
                                                "Bottom 10 Configurations", max_configs=10)
            print(f"  [OK] bottom10_{metric}_trajectories.png")

    print("\n[5/6] Creating Top vs Bottom comparison plots...")
    if len(top_configs_data) > 0 and len(bottom_configs_data) > 0:
        for metric in metrics:
            output_path = output_dir / f'comparison_{metric}_top_vs_bottom.png'
            plot_top_vs_bottom_comparison(top_configs_data, bottom_configs_data,
                                         metric, output_path)
            print(f"  [OK] comparison_{metric}_top_vs_bottom.png")

    print("\n[6/6] Analyzing convergence times...")
    analyze_convergence_times(all_configs_data, output_dir)

    # Save sample individual config dashboard
    if len(top_configs_data) > 0:
        best_config = top_configs_data[0]
        output_path = output_dir / f'dashboard_{best_config["config_name"]}.png'
        plot_all_metrics_single_config(best_config, output_path)
        print(f"  [OK] Sample dashboard for best config saved")

    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS COMPLETE!")
    print(f"All results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
