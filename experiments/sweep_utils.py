#!/usr/bin/env python3
"""
Sweep utilities for managing and analyzing sweep results.

This script provides commands to:
- Analyze completed sweeps
- Clean up old sweep directories
- Compare models across sweeps
- Generate reports
"""

import argparse
import json
import yaml
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from extras.sweep_manager import SweepManager, ModelResult


def find_sweep_directories(base_dir: str = "logs/sweeps") -> List[Path]:
    """Find all sweep directories."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    sweep_dirs = []
    for path in base_path.rglob("sweep_results.json"):
        sweep_dirs.append(path.parent)
    
    return sorted(sweep_dirs, key=lambda x: x.stat().st_mtime, reverse=True)


def load_sweep_results(sweep_dir: Path) -> Optional[List[ModelResult]]:
    """Load results from a sweep directory."""
    results_file = sweep_dir / "sweep_results.json"
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        return [ModelResult(**item) for item in data]
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None


def analyze_sweep(sweep_dir: Path, show_details: bool = False):
    """Analyze a single sweep."""
    print(f"\n{'='*80}")
    print(f"SWEEP ANALYSIS: {sweep_dir.name}")
    print(f"{'='*80}")
    print(f"Directory: {sweep_dir}")
    
    # Load results
    results = load_sweep_results(sweep_dir)
    if results is None:
        print("❌ No valid results found")
        return
    
    print(f"Total runs: {len(results)}")
    
    if not results:
        print("No results to analyze")
        return
    
    # Get metric info
    metric_name = results[0].metric_name
    print(f"Metric: {metric_name}")
    
    # Basic statistics
    metric_values = [r.metric_value for r in results]
    print(f"Best {metric_name}: {min(metric_values):.6f}")
    print(f"Worst {metric_name}: {max(metric_values):.6f}")
    print(f"Mean {metric_name}: {sum(metric_values)/len(metric_values):.6f}")
    
    # Model info
    models = set(r.run_name.split('_')[1] if '_' in r.run_name else 'unknown' for r in results)
    print(f"Models tested: {', '.join(sorted(models))}")
    
    # Time span
    timestamps = [datetime.fromisoformat(r.timestamp) for r in results]
    time_span = max(timestamps) - min(timestamps)
    print(f"Duration: {time_span}")
    print(f"Start: {min(timestamps).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {max(timestamps).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Top models
    print(f"\n📊 TOP MODELS:")
    for i, result in enumerate(results[:5], 1):
        status = "✅" if i <= 3 else "📁"
        print(f"  {status} {i:2d}. {result.run_name}")
        print(f"      {metric_name}: {result.metric_value:.6f} (epoch {result.epoch})")
        if show_details and result.additional_metrics:
            for key, value in result.additional_metrics.items():
                print(f"      {key}: {value:.6f}")
        print(f"      Model: {Path(result.model_path).name}")
        if result.wandb_run_id:
            print(f"      W&B: {result.wandb_run_id}")
        print()


def compare_sweeps(sweep_dirs: List[Path], metric_name: str = None):
    """Compare multiple sweeps."""
    print(f"\n{'='*80}")
    print(f"SWEEP COMPARISON")
    print(f"{'='*80}")
    
    sweep_data = []
    
    for sweep_dir in sweep_dirs:
        results = load_sweep_results(sweep_dir)
        if results is None:
            continue
            
        if metric_name is None:
            metric_name = results[0].metric_name
            
        best_result = results[0] if results else None
        
        sweep_data.append({
            'sweep_dir': sweep_dir.name,
            'full_path': str(sweep_dir),
            'total_runs': len(results),
            'best_metric': best_result.metric_value if best_result else float('inf'),
            'best_model': best_result.run_name if best_result else 'N/A',
            'duration': _calculate_duration(results),
            'timestamp': max(datetime.fromisoformat(r.timestamp) for r in results) if results else datetime.min
        })
    
    # Sort by best metric
    sweep_data.sort(key=lambda x: x['best_metric'])
    
    print(f"Metric: {metric_name}")
    print()
    
    for i, data in enumerate(sweep_data, 1):
        print(f"{i:2d}. {data['sweep_dir']}")
        print(f"    Best {metric_name}: {data['best_metric']:.6f}")
        print(f"    Best model: {data['best_model']}")
        print(f"    Runs: {data['total_runs']}")
        print(f"    Duration: {data['duration']}")
        print(f"    Path: {data['full_path']}")
        print()


def _calculate_duration(results: List[ModelResult]) -> str:
    """Calculate duration of a sweep."""
    if not results:
        return "0:00:00"
    
    timestamps = [datetime.fromisoformat(r.timestamp) for r in results]
    duration = max(timestamps) - min(timestamps)
    
    # Format as HH:MM:SS
    total_seconds = int(duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def cleanup_old_sweeps(base_dir: str = "logs/sweeps", days_old: int = 7, dry_run: bool = True):
    """Clean up old sweep directories."""
    print(f"\n{'='*80}")
    print(f"CLEANUP OLD SWEEPS (older than {days_old} days)")
    print(f"{'='*80}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()
    
    cutoff_date = datetime.now() - timedelta(days=days_old)
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Base directory {base_dir} does not exist")
        return
    
    total_size = 0
    dirs_to_delete = []
    
    for sweep_dir in find_sweep_directories(base_dir):
        # Check if directory is old enough
        dir_mtime = datetime.fromtimestamp(sweep_dir.stat().st_mtime)
        
        if dir_mtime < cutoff_date:
            # Calculate directory size
            dir_size = sum(f.stat().st_size for f in sweep_dir.rglob('*') if f.is_file())
            total_size += dir_size
            dirs_to_delete.append((sweep_dir, dir_size, dir_mtime))
    
    if not dirs_to_delete:
        print("No old sweep directories found")
        return
    
    print(f"Found {len(dirs_to_delete)} directories to delete:")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print()
    
    for sweep_dir, size, mtime in dirs_to_delete:
        print(f"📁 {sweep_dir.name}")
        print(f"   Size: {size / (1024**2):.1f} MB")
        print(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Path: {sweep_dir}")
        print()
    
    if not dry_run:
        confirm = input(f"Delete {len(dirs_to_delete)} directories? [y/N]: ")
        if confirm.lower() == 'y':
            for sweep_dir, _, _ in dirs_to_delete:
                try:
                    shutil.rmtree(sweep_dir)
                    print(f"✅ Deleted {sweep_dir}")
                except Exception as e:
                    print(f"❌ Failed to delete {sweep_dir}: {e}")
        else:
            print("Cleanup cancelled")


def export_results(sweep_dirs: List[Path], output_file: str = "sweep_results.csv"):
    """Export sweep results to CSV."""
    print(f"\n{'='*80}")
    print(f"EXPORT RESULTS")
    print(f"{'='*80}")
    
    all_results = []
    
    for sweep_dir in sweep_dirs:
        results = load_sweep_results(sweep_dir)
        if results is None:
            continue
            
        for result in results:
            row = {
                'sweep_dir': sweep_dir.name,
                'run_id': result.run_id,
                'run_name': result.run_name,
                'metric_name': result.metric_name,
                'metric_value': result.metric_value,
                'epoch': result.epoch,
                'timestamp': result.timestamp,
                'model_path': result.model_path,
                'wandb_run_id': result.wandb_run_id
            }
            
            # Add additional metrics as columns
            if result.additional_metrics:
                for key, value in result.additional_metrics.items():
                    row[f'additional_{key}'] = value
                    
            all_results.append(row)
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_file, index=False)
        print(f"✅ Exported {len(all_results)} results to {output_file}")
        print(f"Columns: {', '.join(df.columns)}")
    else:
        print("No results to export")


def main():
    parser = argparse.ArgumentParser(description="Sweep utilities for managing and analyzing sweep results")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all sweep directories')
    list_parser.add_argument('--base-dir', default='logs/sweeps', help='Base directory to search')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze sweep results')
    analyze_parser.add_argument('sweep_dir', nargs='?', help='Specific sweep directory to analyze')
    analyze_parser.add_argument('--base-dir', default='logs/sweeps', help='Base directory to search')
    analyze_parser.add_argument('--details', action='store_true', help='Show detailed metrics')
    analyze_parser.add_argument('--all', action='store_true', help='Analyze all sweeps')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple sweeps')
    compare_parser.add_argument('--base-dir', default='logs/sweeps', help='Base directory to search')
    compare_parser.add_argument('--metric', help='Metric to compare (auto-detect if not specified)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old sweep directories')
    cleanup_parser.add_argument('--base-dir', default='logs/sweeps', help='Base directory to search')
    cleanup_parser.add_argument('--days', type=int, default=7, help='Delete directories older than this many days')
    cleanup_parser.add_argument('--live', action='store_true', help='Actually delete (default is dry run)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export results to CSV')
    export_parser.add_argument('--base-dir', default='logs/sweeps', help='Base directory to search')
    export_parser.add_argument('--output', default='sweep_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        sweep_dirs = find_sweep_directories(args.base_dir)
        print(f"Found {len(sweep_dirs)} sweep directories in {args.base_dir}:")
        for i, sweep_dir in enumerate(sweep_dirs, 1):
            print(f"{i:3d}. {sweep_dir.name} ({sweep_dir})")
    
    elif args.command == 'analyze':
        if args.sweep_dir:
            # Analyze specific sweep
            sweep_path = Path(args.sweep_dir)
            if not sweep_path.exists():
                # Try relative to base_dir
                sweep_path = Path(args.base_dir) / args.sweep_dir
            if not sweep_path.exists():
                print(f"Sweep directory not found: {args.sweep_dir}")
                return
            analyze_sweep(sweep_path, args.details)
        elif args.all:
            # Analyze all sweeps
            sweep_dirs = find_sweep_directories(args.base_dir)
            for sweep_dir in sweep_dirs:
                analyze_sweep(sweep_dir, args.details)
        else:
            # Analyze most recent sweep
            sweep_dirs = find_sweep_directories(args.base_dir)
            if sweep_dirs:
                analyze_sweep(sweep_dirs[0], args.details)
            else:
                print(f"No sweep directories found in {args.base_dir}")
    
    elif args.command == 'compare':
        sweep_dirs = find_sweep_directories(args.base_dir)
        if sweep_dirs:
            compare_sweeps(sweep_dirs, args.metric)
        else:
            print(f"No sweep directories found in {args.base_dir}")
    
    elif args.command == 'cleanup':
        cleanup_old_sweeps(args.base_dir, args.days, not args.live)
    
    elif args.command == 'export':
        sweep_dirs = find_sweep_directories(args.base_dir)
        if sweep_dirs:
            export_results(sweep_dirs, args.output)
        else:
            print(f"No sweep directories found in {args.base_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
