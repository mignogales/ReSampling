#!/usr/bin/env python3
"""
Combine sampling rate analysis results from multiple experiments into a single graph.

This script searches for all 'sampling_rate_analysis' folders under logs/ and combines
their CSV results into unified plots showing model performance across different datasets
and sampling rates.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from colorama import Fore
import seaborn as sns


def extract_model_dataset_from_path(path: Path) -> tuple:
    """
    Extract model and dataset names from the parent directory path.
    
    Expected structure: logs/sweeps/sweep_{model}_{dataset}/best_models/sampling_rate_analysis/
    or with subfolders: logs/sweeps/sweep_{model}_{dataset}/best_models/sampling_rate_analysis/{subfolder}/
    """
    path_parts = path.parts
    
    # Check if this is a subfolder within sampling_rate_analysis
    subfolder_suffix = ""
    if path.parent.name == "sampling_rate_analysis":
        subfolder_name = path.name
        # Handle special cases like s5_modified_deltas, s5_original_deltas
        if "modified_deltas" in subfolder_name:
            subfolder_suffix = "_modified"
        elif "original_deltas" in subfolder_name:
            subfolder_suffix = "_original"
        else:
            subfolder_suffix = f"_{subfolder_name}"
    
    # Look for patterns like sweep_mlp_Solar, sweep_gru_Wind, etc.
    for part in path_parts:
        if part.startswith('sweep_'):
            # Extract model and dataset from sweep_model_dataset
            match = re.match(r'sweep_([^_]+)_(.+)', part)
            if match:
                model, dataset = match.groups()
                return model.upper(), dataset + subfolder_suffix
    
    # Fallback: try to extract from any part containing underscores
    for part in path_parts:
        if '_' in part and any(dataset in part.lower() for dataset in ['solar', 'wind', 'electricity']):
            parts_split = part.split('_')
            for i, p in enumerate(parts_split):
                if p.lower() in ['solar', 'wind', 'electricity']:
                    dataset = p + subfolder_suffix
                    model = parts_split[i-1] if i > 0 else 'unknown'
                    return model.upper(), dataset
    
    return 'unknown', 'unknown' + subfolder_suffix


def find_sampling_analysis_dirs(base_path: Path = Path("logs")) -> list:
    """Find all sampling_rate_analysis directories and their subfolders."""
    analysis_dirs = []
    
    if not base_path.exists():
        print(f"Base path {base_path} does not exist")
        return analysis_dirs
    
    # Recursively find all sampling_rate_analysis directories
    for analysis_dir in base_path.rglob("sampling_rate_analysis"):
        if analysis_dir.is_dir():
            # Check if there's a CSV directly in this directory
            csv_file = analysis_dir / "sampling_rate_results.csv"
            if csv_file.exists():
                analysis_dirs.append(analysis_dir)
                print(f"Found: {analysis_dir}")
            else:
                # Check for subfolders with CSV files
                for subfolder in analysis_dir.iterdir():
                    if subfolder.is_dir():
                        sub_csv = subfolder / "sampling_rate_results.csv"
                        if sub_csv.exists():
                            analysis_dirs.append(subfolder)
                            print(f"Found: {subfolder}")
                        else:
                            print(f"Found subfolder but no CSV: {subfolder}")
                
                # If no CSV found in main dir or subfolders
                if not any((analysis_dir / sub).exists() and (analysis_dir / sub / "sampling_rate_results.csv").exists() 
                          for sub in analysis_dir.iterdir() if sub.is_dir()):
                    if not csv_file.exists():
                        print(f"Found directory but no CSV: {analysis_dir}")
    
    return analysis_dirs


def load_and_combine_results(analysis_dirs: list) -> pd.DataFrame:
    """Load CSV files and combine with model/dataset labels."""
    all_results = []
    
    for analysis_dir in analysis_dirs:
        csv_file = analysis_dir / "sampling_rate_results.csv"
        
        try:
            df = pd.read_csv(csv_file)
            model, dataset = extract_model_dataset_from_path(analysis_dir)
            
            df['model'] = model
            df['dataset'] = dataset
            df['experiment'] = f"{model}_{dataset}"
            
            all_results.append(df)
            print(f"Loaded {len(df)} rows from {model} on {dataset}")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_results:
        raise ValueError("No valid CSV files found")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


def create_combined_plots(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive plots showing all model-dataset combinations."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    colors = plt.cm.tab10(range(len(df['experiment'].unique())))
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Sampling Rate Robustness Across Models and Datasets', fontsize=16, fontweight='bold')
    
    metrics = ['mse', 'mae']
    metric_names = ['Mean Squared Error', 'Mean Absolute Error']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        # Plot each model-dataset combination
        for i, experiment in enumerate(df['experiment'].unique()):
            exp_data = df[df['experiment'] == experiment].sort_values('sampling_rate')
            
            ax.plot(exp_data['sampling_rate'], exp_data[metric], 
                   'o-', label=experiment, linewidth=2, markersize=6, 
                   color=colors[i])
        
        ax.set_xlabel('Sampling Rate Multiplier', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} vs Temporal Resolution')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plots
    plot_path = output_dir / 'combined_sampling_rate_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_sampling_rate_analysis.pdf', bbox_inches='tight')
    print(Fore.GREEN + f"Combined plots saved to {plot_path}" + Fore.RESET)
    
    plt.show()


def create_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Create heatmaps showing performance degradation."""
    
    # Calculate relative performance (normalized by sampling_rate=1.0)
    df_norm = df.copy()
    
    for experiment in df['experiment'].unique():
        exp_mask = df_norm['experiment'] == experiment
        baseline_mse = df_norm[(exp_mask) & (df_norm['sampling_rate'] == 1.0)]['mse'].iloc[0]
        baseline_mae = df_norm[(exp_mask) & (df_norm['sampling_rate'] == 1.0)]['mae'].iloc[0]
        
        df_norm.loc[exp_mask, 'mse_relative'] = df_norm.loc[exp_mask, 'mse'] / baseline_mse
        df_norm.loc[exp_mask, 'mae_relative'] = df_norm.loc[exp_mask, 'mae'] / baseline_mae
    
    # Create pivot tables for heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Relative Performance Degradation (Normalized by 1.0x Rate)', fontsize=14, fontweight='bold')
    
    for idx, (metric, title) in enumerate([('mse_relative', 'MSE Degradation'), ('mae_relative', 'MAE Degradation')]):
        pivot_df = df_norm.pivot_table(values=metric, index='experiment', columns='sampling_rate')
        
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=axes[idx], cbar_kws={'label': 'Relative Performance'})
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Sampling Rate Multiplier')
        axes[idx].set_ylabel('Model-Dataset')
    
    plt.tight_layout()
    
    heatmap_path = output_dir / 'sampling_rate_heatmaps.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'sampling_rate_heatmaps.pdf', bbox_inches='tight')
    print(Fore.GREEN + f"Heatmaps saved to {heatmap_path}" + Fore.RESET)
    
    plt.show()


def create_degradation_plots(df: pd.DataFrame, output_dir: Path):
    """Create line plots showing relative performance degradation."""
    
    # Calculate relative performance (normalized by sampling_rate=1.0)
    df_norm = df.copy()
    
    for experiment in df['experiment'].unique():
        exp_mask = df_norm['experiment'] == experiment
        baseline_mse = df_norm[(exp_mask) & (df_norm['sampling_rate'] == 1.0)]['mse'].iloc[0]
        baseline_mae = df_norm[(exp_mask) & (df_norm['sampling_rate'] == 1.0)]['mae'].iloc[0]
        
        df_norm.loc[exp_mask, 'mse_relative'] = df_norm.loc[exp_mask, 'mse'] / baseline_mse
        df_norm.loc[exp_mask, 'mae_relative'] = df_norm.loc[exp_mask, 'mae'] / baseline_mae
    
    # Set up colors
    colors = plt.cm.tab10(range(len(df['experiment'].unique())))
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Relative Performance Degradation vs Sampling Rate', fontsize=16, fontweight='bold')
    
    metrics = [('mse_relative', 'Relative MSE'), ('mae_relative', 'Relative MAE')]
    
    for idx, (metric, metric_name) in enumerate(metrics):
        ax = axes[idx]
        
        # Plot each model-dataset combination
        for i, experiment in enumerate(df_norm['experiment'].unique()):
            exp_data = df_norm[df_norm['experiment'] == experiment].sort_values('sampling_rate')
            
            ax.plot(exp_data['sampling_rate'], exp_data[metric], 
                   'o-', label=experiment, linewidth=2, markersize=6, 
                   color=colors[i])
        
        # Add baseline reference line at 1.0
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1.0x)')
        
        ax.set_xlabel('Sampling Rate Multiplier', fontsize=12)
        ax.set_ylabel(f'{metric_name} (normalized to 1.0x)', fontsize=12)
        ax.set_title(f'{metric_name} Degradation')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plots
    plot_path = output_dir / 'degradation_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'degradation_plots.pdf', bbox_inches='tight')
    print(Fore.GREEN + f"Degradation plots saved to {plot_path}" + Fore.RESET)
    
    plt.show()


def main():
    """Main function to combine and visualize sampling rate analysis results."""
    print(Fore.CYAN + "="*60)
    print("Combining Sampling Rate Analysis Results")
    print("="*60 + Fore.RESET)
    
    # Find all analysis directories
    analysis_dirs = find_sampling_analysis_dirs()

    # for now, delete folders that contain "64_12"
    analysis_dirs = [d for d in analysis_dirs if "64_12" not in str(d)]

    if not analysis_dirs:
        print(Fore.RED + "No sampling_rate_analysis directories found!" + Fore.RESET)
        return
    
    print(f"\nFound {len(analysis_dirs)} analysis directories")
    
    # Load and combine results
    combined_df = load_and_combine_results(analysis_dirs)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Experiments: {', '.join(combined_df['experiment'].unique())}")
    
    # Create output directory
    output_dir = Path("logs/combined_sampling_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined CSV
    combined_csv = output_dir / 'all_sampling_rate_results.csv'
    combined_df.to_csv(combined_csv, index=False)
    print(Fore.GREEN + f"Combined CSV saved to {combined_csv}" + Fore.RESET)
    
    # Create visualizations
    print(Fore.YELLOW + "\nGenerating combined plots..." + Fore.RESET)
    create_combined_plots(combined_df, output_dir)
    
    print(Fore.YELLOW + "Generating heatmaps..." + Fore.RESET)
    create_heatmaps(combined_df, output_dir)
    
    print(Fore.YELLOW + "Generating degradation plots..." + Fore.RESET)
    create_degradation_plots(combined_df, output_dir)
    
    print(Fore.GREEN + f"\nAnalysis complete! Results saved to {output_dir}" + Fore.RESET)


if __name__ == "__main__":
    main()