#!/usr/bin/env python3
"""
Final Consolidated Analysis Script for FSS Grid Results

This script generates all final publication-ready plots from the consolidated summary data:
1. Four heatmaps: cost, quality, resolution_time, time_to_convergence (by #Satellites x #CN)
2. Correlation matrix with strength annotations
3. Failure timing analysis (fig3_failure_timing.png)
4. Interaction speedup heatmap (fig4_interaction_speedup_heatmap.png)

All plots are saved to the 'plots/' folder.

Usage:
    python analyze_final_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')

# ===================== CONFIGURATION ========================================
CONSOLIDATED_SUMMARY = Path("results_grid_run1/14102025_analysis/consolidated_summary.xlsx")
FAILURE_ANALYSIS_DIR = Path("results_grid_run1/14102025")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Plotting configuration
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'
sns.set_palette("husl")
sns.set_style("whitegrid")
# ============================================================================


def load_data():
    """Load consolidated summary data."""
    print("Loading consolidated summary data...")
    df = pd.read_excel(CONSOLIDATED_SUMMARY)

    # Calculate absolute number of CNs
    df['num_cn'] = (df['num_satellites'] * df['fraction_CNs']).round().astype(int)

    print(f"Loaded {len(df)} configurations")
    print(f"  Satellite counts: {sorted(df['num_satellites'].unique())}")
    print(f"  CN counts: {sorted(df['num_cn'].unique())}")

    return df


def aggregate_by_sat_cn(df):
    """
    Aggregate data by (#Satellites, #CN) for OMNICN=0 configurations only.
    This averages across different constellation types.
    """
    print("\nAggregating by (#Satellites, #CN) for OMNICN=0 only...")

    # Filter for OMNICN=0 only
    df_filtered = df[df['omni_fraction_CNs'] == 0.0].copy()

    print(f"  Filtered to {len(df_filtered)} OMNICN=0 configurations (from {len(df)} total)")

    # Group by absolute counts
    agg_df = df_filtered.groupby(['num_satellites', 'num_cn']).agg({
        'winner_bid_mean': 'mean',
        'winner_RF_mean': 'mean',
        'RF_time_mean': 'mean',
        'duration_mean': 'mean'
    }).reset_index()

    # Rename for clarity
    agg_df.rename(columns={
        'winner_bid_mean': 'cost_mean',
        'winner_RF_mean': 'quality_mean',
        'RF_time_mean': 'resolution_time_mean',
        'duration_mean': 'time_to_convergence'
    }, inplace=True)

    print(f"  Aggregated to {len(agg_df)} unique (#Satellites, #CN) combinations")

    return agg_df


def create_heatmap(pivot_data, title, cmap, output_filename, fmt='.2f'):
    """
    Create a heatmap with specified styling.

    Args:
        pivot_data: Pivot table with #Satellites as index, #CN as columns
        title: Plot title
        cmap: Colormap name
        output_filename: Output file name
        fmt: Format string for annotations
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create heatmap with MUCH larger fonts
    sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                cbar_kws={'label': title.split(' by ')[0], 'shrink': 0.8, 'pad': 0.02},
                linewidths=2.5, linecolor='white',
                annot_kws={'fontsize': 20, 'fontweight': 'bold'})

    # Much larger labels
    ax.set_xlabel('Central Nodes', fontsize=22, fontweight='bold', labelpad=12)
    ax.set_ylabel('Satellites', fontsize=22, fontweight='bold', labelpad=12)
    ax.set_title(title, fontsize=24, fontweight='bold', pad=18)

    # Larger tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18, fontweight='bold')

    # Increase colorbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(title.split(' by ')[0], fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] {output_filename}")


def generate_heatmaps(agg_df):
    """Generate all four heatmaps."""
    print("\nGenerating heatmaps...")

    metrics = [
        ('cost_mean', 'Bidding Cost', 'RdYlGn_r', 'heatmap_cost_mean.png'),
        ('quality_mean', 'Solution Quality (RF)', 'RdYlGn', 'heatmap_quality_mean.png'),
        ('resolution_time_mean', 'Task Resolution Time', 'YlOrRd_r', 'heatmap_resolution_time_mean.png'),
        ('time_to_convergence', 'Convergence Time', 'YlOrRd_r', 'heatmap_time_to_convergence.png')
    ]

    for metric, title, cmap, filename in metrics:
        # Create pivot table
        pivot = agg_df.pivot(index='num_satellites', columns='num_cn', values=metric)

        # Sort index and columns
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot.sort_index(axis=1, ascending=True)

        create_heatmap(pivot, title, cmap, filename)


def generate_correlation_matrix(df):
    """
    Generate correlation matrix with strength annotations (OMNICN=0 only).
    """
    print("\nGenerating correlation matrix (OMNICN=0 only)...")

    # Filter for OMNICN=0 only
    df = df[df['omni_fraction_CNs'] == 0.0].copy()
    print(f"  Using {len(df)} OMNICN=0 configurations")

    # Prepare data for correlation
    # Encode constellation as numeric (simple label encoding)
    constellation_mapping = {const: idx for idx, const in enumerate(df['constellation'].unique())}
    df['constellation_encoded'] = df['constellation'].map(constellation_mapping)

    # Calculate num_cn
    df['num_cn'] = (df['num_satellites'] * df['fraction_CNs']).round().astype(int)

    # Select variables for correlation
    corr_vars = ['num_satellites', 'num_cn', 'constellation_encoded',
                 'winner_RF_mean', 'winner_bid_mean', 'RF_time_mean', 'duration_mean']

    # Rename for display
    display_names = {
        'num_satellites': 'num_sats',
        'winner_RF_mean': 'quality_mean',
        'winner_bid_mean': 'cost_mean',
        'RF_time_mean': 'resolution_time_mean',
        'duration_mean': 'time_to_convergence'
    }

    corr_df = df[corr_vars].rename(columns=display_names)

    # Calculate correlation matrix
    corr_matrix = corr_df.corr()

    # Create annotations with strength indicators
    annot = np.empty_like(corr_matrix, dtype=object)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            abs_val = abs(val)

            if abs_val == 1.0:
                strength = ""
            elif abs_val > 0.5:
                strength = "\n(STRONG)"
            elif abs_val > 0.3:
                strength = "\n(moderate)"
            elif abs_val > 0.1:
                strength = "\n(weak)"
            else:
                strength = "\n(negligible)"

            annot[i, j] = f"{val:.3f}{strength}"

    # Create figure with better sizing
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(corr_matrix, annot=annot, fmt='', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={'label': 'Correlation', 'shrink': 0.8},
                linewidths=3, linecolor='white',
                annot_kws={'fontsize': 13, 'fontweight': 'bold'})

    ax.set_title('Correlation Analysis',
                fontsize=26, fontweight='bold', pad=25)

    # Rotate labels for better readability with larger fonts
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16, fontweight='bold')

    # Increase colorbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Correlation', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_strength_annotated.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  [OK] correlation_strength_annotated.png")


def generate_failure_timing_plot():
    """
    Generate failure timing analysis plot (fig3).
    Analyzes UNSOLVED runs from the raw data.
    """
    print("\nGenerating failure timing analysis...")

    # Find all UNSOLVED files
    failures = []

    config_folders = [f for f in FAILURE_ANALYSIS_DIR.iterdir() if f.is_dir()]

    for config_folder in config_folders:
        for xlsx_file in config_folder.glob("UNSOLVED*.xlsx"):
            try:
                df = pd.read_excel(xlsx_file, sheet_name='task_0')

                failure_info = {
                    'config_name': config_folder.name,
                    'filename': xlsx_file.name,
                    'final_timestep': df['time_step'].iloc[-1],
                    'duration': len(df),
                }

                # Identify failure cause
                if len(df) >= 500:
                    recent_knowledge = df['knowledge'].tail(500)
                    if recent_knowledge.nunique() == 1:
                        failure_info['failure_cause'] = 'STAGNATION'
                    else:
                        failure_info['failure_cause'] = 'INSUFFICIENT_BIDS'
                else:
                    failure_info['failure_cause'] = 'UNKNOWN'

                failures.append(failure_info)

            except Exception as e:
                print(f"    Warning: Could not process {xlsx_file.name}: {e}")
                continue

    if len(failures) == 0:
        print("  [WARNING] No UNSOLVED files found, skipping failure timing plot")
        return

    failures_df = pd.DataFrame(failures)

    print(f"  Found {len(failures_df)} failed runs")

    # Create failure timing plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Color scheme
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    # Create consistent bins for both plots
    min_val = failures_df['final_timestep'].min()
    max_val = failures_df['final_timestep'].max()
    bins = np.linspace(min_val, max_val, 31)  # 30 bins

    ax = axes[0]
    ax.hist(failures_df['final_timestep'], bins=bins, color='#e74c3c', edgecolor='white', alpha=0.85, linewidth=1.5)
    median_val = failures_df['final_timestep'].median()
    ax.axvline(median_val, color='#2c3e50', linestyle='--',
               linewidth=3.5, label=f"Median: {median_val:.0f}")
    ax.set_xlabel('Timestep at Failure', fontsize=18, fontweight='bold', labelpad=12)
    ax.set_ylabel('Count', fontsize=18, fontweight='bold', labelpad=12)
    ax.set_title('Failure Distribution', fontsize=20, fontweight='bold', pad=18)
    ax.legend(fontsize=16, frameon=True, shadow=True, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=15)

    ax = axes[1]
    causes = failures_df['failure_cause'].unique()
    for idx, cause in enumerate(causes):
        subset = failures_df[failures_df['failure_cause'] == cause]
        ax.hist(subset['final_timestep'], bins=bins, alpha=0.75, label=cause,
               edgecolor='white', color=colors[idx % len(colors)], linewidth=1.5)
    ax.set_xlabel('Timestep at Failure', fontsize=18, fontweight='bold', labelpad=12)
    ax.set_ylabel('Count', fontsize=18, fontweight='bold', labelpad=12)
    ax.set_title('Failure by Cause', fontsize=20, fontweight='bold', pad=18)
    ax.legend(fontsize=16, frameon=True, shadow=True, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_failure_timing.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  [OK] fig3_failure_timing.png")


def generate_dose_response_overall(df):
    """
    Generate overall dose-response curves showing omniscience effects.
    Shows how performance metrics change with omniscience fraction (averaged across all configs).
    """
    print("\nGenerating dose-response overall plot...")

    # Map omniscience fractions to discrete values for cleaner x-axis
    def map_to_discrete_omni(val):
        if val < 0.01:
            return 0.0
        elif 0.20 <= val < 0.30:
            return 0.25
        elif 0.45 <= val < 0.55:
            return 0.50
        elif val >= 0.95:
            return 1.0
        else:
            return round(val, 2)

    df['omni_discrete'] = df['omni_fraction_CNs'].apply(map_to_discrete_omni)

    # Calculate success rate
    df['success_rate'] = df['successful_runs'] / df['total_runs']

    # Define metrics to plot with shorter labels
    metrics = {
        'success_rate': ('Success Rate', '#2ecc71', '#27ae60'),
        'duration_mean': ('Convergence Time', '#e74c3c', '#c0392b'),
        'winner_RF_mean': ('Solution Quality', '#3498db', '#2980b9'),
        'knowledge_mean': ('Knowledge Level', '#9b59b6', '#8e44ad')
    }

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = axes.flatten()

    for idx, (metric_name, (metric_label, color_main, color_dark)) in enumerate(metrics.items()):
        ax = axes[idx]

        # Group by omniscience fraction and compute statistics
        omni_stats = df.groupby('omni_discrete')[metric_name].agg(['mean', 'std', 'count'])

        # Calculate standard error
        omni_stats['se'] = omni_stats['std'] / np.sqrt(omni_stats['count'])

        # Convert index to percentage for x-axis labels
        x_values = omni_stats.index * 100  # Convert to percentage

        # Plot with error bars and shaded region
        ax.plot(x_values, omni_stats['mean'], marker='o', markersize=14,
               linewidth=4, color=color_main, markeredgewidth=2.5,
               markeredgecolor=color_dark, markerfacecolor=color_main)

        # Add shaded region for standard error
        ax.fill_between(x_values,
                       omni_stats['mean'] - omni_stats['se'],
                       omni_stats['mean'] + omni_stats['se'],
                       alpha=0.25, color=color_main)

        ax.set_xlabel('Omniscient CNs (%)', fontsize=18, fontweight='bold', labelpad=12)
        ax.set_ylabel(metric_label, fontsize=18, fontweight='bold', labelpad=12)
        ax.set_title(metric_label, fontsize=20, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(-5, 105)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=15)

    plt.suptitle('Omniscience Effects',
                fontsize=24, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dose_response_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  [OK] dose_response_overall.png")


def generate_interaction_speedup_heatmap(df):
    """
    Generate interaction speedup heatmap (fig4).
    Shows speedup from omniscient CNs across CN fractions.
    """
    print("\nGenerating interaction speedup heatmap...")

    # Map to discrete omniscience fractions used in the grid [0.0, 0.25, 0.50, 1.0]
    def map_to_discrete_omni(val):
        """Map floating-point omni values to discrete grid values."""
        if val < 0.01:
            return 0.0
        elif 0.20 <= val < 0.30:
            return 0.25
        elif 0.45 <= val < 0.55:
            return 0.50
        elif val >= 0.95:
            return 1.0
        else:
            return round(val, 2)  # fallback

    df['omni_fraction_CNs_discrete'] = df['omni_fraction_CNs'].apply(map_to_discrete_omni)

    # Compute baseline comparisons (OMNICN=0 vs OMNICN>0)
    comparisons = []

    grouping_cols = ['num_satellites', 'fraction_CNs', 'constellation']

    for group_key, group_df in df.groupby(grouping_cols):
        # Get baseline (OMNICN=0)
        baseline = group_df[group_df['omni_fraction_CNs_discrete'] == 0.0]

        if len(baseline) == 0:
            continue

        baseline_row = baseline.iloc[0]

        # Compare each OMNICN>0 against baseline
        for _, treatment_row in group_df[group_df['omni_fraction_CNs_discrete'] > 0].iterrows():
            speedup_factor = baseline_row['duration_mean'] / max(treatment_row['duration_mean'], 1)

            comparisons.append({
                'cn_fraction': group_key[1],
                'omnicn_fraction': treatment_row['omni_fraction_CNs_discrete'],
                'speedup_factor': speedup_factor
            })

    if len(comparisons) == 0:
        print("  [WARNING] No OMNICN>0 configurations found, skipping interaction heatmap")
        return

    comparison_df = pd.DataFrame(comparisons)

    # Create pivot table
    pivot_speedup = comparison_df.pivot_table(
        values='speedup_factor',
        index='cn_fraction',
        columns='omnicn_fraction',
        aggfunc='mean'
    )

    # Create heatmap with MUCH larger fonts
    fig, ax = plt.subplots(figsize=(16, 10))

    sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='RdYlGn',
                ax=ax, cbar_kws={'label': 'Speedup Factor', 'shrink': 0.8},
                linewidths=3, linecolor='white', center=1.0,
                annot_kws={'fontsize': 24, 'fontweight': 'bold'})

    # Much larger axis labels
    ax.set_xlabel('Omniscient CN Fraction', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('CN Fraction', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title('Convergence Speedup by Omniscience',
                fontsize=26, fontweight='bold', pad=20)

    # Much larger tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontweight='bold')

    # Increase colorbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Speedup Factor', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_interaction_speedup_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  [OK] fig4_interaction_speedup_heatmap.png")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("FINAL CONSOLIDATED ANALYSIS - PUBLICATION PLOTS")
    print("="*80)

    # Load data
    df = load_data()

    # Aggregate by (#Satellites, #CN)
    agg_df = aggregate_by_sat_cn(df)

    # Generate heatmaps
    generate_heatmaps(agg_df)

    # Generate correlation matrix
    generate_correlation_matrix(df)

    # Generate failure timing plot
    generate_failure_timing_plot()

    # Generate dose-response overall plot
    generate_dose_response_overall(df)

    # Generate interaction speedup heatmap
    generate_interaction_speedup_heatmap(df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - heatmap_cost_mean.png")
    print("  - heatmap_quality_mean.png")
    print("  - heatmap_resolution_time_mean.png")
    print("  - heatmap_time_to_convergence.png")
    print("  - correlation_strength_annotated.png")
    print("  - fig3_failure_timing.png")
    print("  - dose_response_overall.png")
    print("  - fig4_interaction_speedup_heatmap.png")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
