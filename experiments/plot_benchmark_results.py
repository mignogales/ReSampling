"""
Plot benchmark results for parallel_chunking experiments.

This script reads the benchmark results JSON file and creates plots showing
the relationship between parallel_chunking values and both training time
and test metrics (MSE/MAE).
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from colorama import Fore


def load_benchmark_results(results_path: str) -> list:
    """Load benchmark results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    print(Fore.GREEN + f"Loaded {len(results)} results from: {results_path}" + Fore.RESET)
    return results


def plot_dual_axis(results: list, metric_name: str, output_dir: str):
    """
    Create a dual-axis plot with parallel_chunking on x-axis,
    training time on left y-axis, and metric (MSE/MAE) on right y-axis.
    
    Args:
        results: List of result dictionaries
        metric_name: Name of the metric to plot ('test/mae' or 'test/mse')
        output_dir: Directory to save the plot
    """
    # Sort results by parallel_chunking value
    results = sorted(results, key=lambda x: x['parallel_chunking'])
    
    # Extract data
    parallel_chunking_values = [r['parallel_chunking'] for r in results]
    training_times = [r['training_time_minutes'] for r in results]
    
    # Find the metric in test_metrics
    metric_values = []
    for r in results:
        metric_value = r['test_metrics'].get(metric_name, None)
        if metric_value is None:
            # Try alternative metric names
            for key in r['test_metrics'].keys():
                if metric_name.split('/')[-1].lower() in key.lower():
                    metric_value = r['test_metrics'][key]
                    break
        metric_values.append(metric_value if metric_value is not None else 0)
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot training time on primary y-axis (left)
    color_time = '#2ecc71'
    ax1.set_xlabel('Parallel Chunking', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Training Time (minutes)', fontsize=13, fontweight='bold', color=color_time)
    line1 = ax1.plot(parallel_chunking_values, training_times, marker='o', linewidth=2.5, 
                     markersize=8, color=color_time, label='Training Time', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_time, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Create secondary y-axis (right) for metric
    ax2 = ax1.twinx()
    color_metric = '#e74c3c'
    metric_label = metric_name.split('/')[-1].upper()
    ax2.set_ylabel(f'Test {metric_label}', fontsize=13, fontweight='bold', color=color_metric)
    line2 = ax2.plot(parallel_chunking_values, metric_values, marker='s', linewidth=2.5,
                     markersize=8, color=color_metric, label=f'Test {metric_label}', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_metric, labelsize=11)
    
    # Add title
    plt.title(f'Parallel Chunking: Training Time vs {metric_label}', 
              fontsize=15, fontweight='bold', pad=20)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
               ncol=2, fontsize=11, frameon=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'parallel_chunking_vs_{metric_label.lower()}_and_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(Fore.GREEN + f"Plot saved to: {output_path}" + Fore.RESET)
    return str(output_path)


def plot_combined_metrics(results: list, output_dir: str):
    """
    Create plots for both MAE and MSE if available.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    if not results:
        print(Fore.RED + "No results to plot!" + Fore.RESET)
        return
    
    # Get available metrics from first result
    available_metrics = list(results[0]['test_metrics'].keys())
    print(f"Available metrics: {available_metrics}")
    
    # Plot for each available metric
    for metric_key in available_metrics:
        try:
            plot_dual_axis(results, metric_key, output_dir)
        except Exception as e:
            print(Fore.YELLOW + f"Could not plot metric '{metric_key}': {e}" + Fore.RESET)
    
    # Also create a summary table
    create_summary_table(results, output_dir)


def create_summary_table(results: list, output_dir: str):
    """Create a summary table showing all metrics."""
    results = sorted(results, key=lambda x: x['parallel_chunking'])
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(results) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Parallel\nChunking', 'Training Time\n(min)', 'Epochs']
    metric_keys = list(results[0]['test_metrics'].keys()) if results[0]['test_metrics'] else []
    headers.extend([k.split('/')[-1].upper() for k in metric_keys])
    
    table_data = []
    for r in results:
        row = [
            f"{r['parallel_chunking']:.1f}",
            f"{r['training_time_minutes']:.2f}",
            f"{r['epochs_trained']}"
        ]
        for mk in metric_keys:
            row.append(f"{r['test_metrics'].get(mk, 0):.4f}")
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Benchmark Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    output_path = Path(output_dir) / 'summary_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(Fore.GREEN + f"Summary table saved to: {output_path}" + Fore.RESET)


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results for parallel_chunking experiments')
    parser.add_argument('results_path', type=str, help='Path to benchmark_results.json file')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Output directory for plots (default: same as results file)')
    
    args = parser.parse_args()
    
    # Load results
    results = load_benchmark_results(args.results_path)
    
    # Determine output directory
    output_dir = args.output_dir if args.output_dir else Path(args.results_path).parent
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print(Fore.CYAN + "\nGenerating plots..." + Fore.RESET)
    plot_combined_metrics(results, str(output_dir))
    
    print(Fore.CYAN + f"\nAll plots saved to: {output_dir}" + Fore.RESET)


if __name__ == "__main__":
    main()
