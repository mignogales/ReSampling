#!/usr/bin/env python3
"""
Sweep Results Analyzer

This script analyzes the best model YAML files from hyperparameter sweeps
to extract performance statistics and identify the best models for each dataset.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Any, Optional
import glob
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SweepResultsAnalyzer:
    """Analyzes sweep results from YAML configuration files."""
    
    def __init__(self, sweeps_base_dir: str = "logs/sweeps"):
        """Initialize the analyzer."""
        self.sweeps_base_dir = Path(sweeps_base_dir)
        self.results = []
        self.best_models = {}
        
    def find_best_model_configs(self) -> List[Dict[str, Any]]:
        """Find all best model configuration files in the sweeps directory."""
        config_files = []
        
        # Search patterns for best model configs
        patterns = [
            "**/best_models/*config.yaml",
            "**/best_models/*_config.yaml",
            "**/*best*config*.yaml"
        ]
        
        for pattern in patterns:
            files = list(self.sweeps_base_dir.glob(pattern))
            config_files.extend(files)
        
        # Remove duplicates
        config_files = list(set(config_files))
        
        logger.info(f"Found {len(config_files)} best model configuration files")
        return config_files
    
    def extract_model_info(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Extract relevant information from a model configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract basic info
            model_info = {
                'config_path': str(config_path),
                'model_name': config.get('model', {}).get('name', 'unknown'),
                'dataset_name': config.get('dataset', {}).get('name', 'unknown'),
                'file_name': config_path.name,
                'directory': str(config_path.parent)
            }
            
            # Extract model hyperparameters
            model_hparams = config.get('model', {}).get('hparams', {})
            for param, value in model_hparams.items():
                model_info[f'model_{param}'] = value
            
            # Extract optimizer info
            optimizer = config.get('optimizer', {})
            model_info['batch_size'] = optimizer.get('batch_size')
            model_info['epochs'] = optimizer.get('epochs')
            model_info['grad_clip_val'] = optimizer.get('grad_clip_val')
            model_info['loss_fn'] = optimizer.get('loss_fn')
            model_info['patience'] = optimizer.get('patience')
            
            # Extract optimizer hyperparameters
            opt_hparams = optimizer.get('hparams', {})
            model_info['learning_rate'] = opt_hparams.get('lr')
            model_info['weight_decay'] = opt_hparams.get('weight_decay')
            
            # Extract dataset info
            dataset = config.get('dataset', {})
            model_info['horizon'] = dataset.get('horizon')
            model_info['window_size'] = dataset.get('window_size')
            model_info['stride'] = dataset.get('stride')
            model_info['delay'] = dataset.get('delay')
            
            # Try to extract performance metrics from filename or path
            performance_info = self._extract_performance_from_path(config_path)
            model_info.update(performance_info)
            
            # Extract validation loss from load_model_path if available
            load_model_path = config.get('load_model_path', '')
            if load_model_path:
                val_loss = self._extract_val_loss_from_path(load_model_path)
                if val_loss:
                    model_info['val_loss'] = val_loss
            
            return model_info
            
        except Exception as e:
            logger.warning(f"Error processing {config_path}: {e}")
            return None
    
    def _extract_performance_from_path(self, config_path: Path) -> Dict[str, Any]:
        """Extract performance metrics from file path or name."""
        performance = {}
        
        # Extract from filename
        filename = config_path.name
        
        # Look for validation loss in filename
        val_loss_match = re.search(r'val_loss_(\d+\.?\d*)', filename)
        if val_loss_match:
            performance['val_loss'] = float(val_loss_match.group(1))
        
        # Look for epoch information
        epoch_match = re.search(r'epoch_(\d+)', filename)
        if epoch_match:
            performance['best_epoch'] = int(epoch_match.group(1))
        
        # Extract timestamp if available
        timestamp_match = re.search(r'(\d{8}_\d{6})', str(config_path))
        if timestamp_match:
            performance['timestamp'] = timestamp_match.group(1)
        
        return performance
    
    def _extract_val_loss_from_path(self, model_path: str) -> Optional[float]:
        """Extract validation loss from model file path."""
        val_loss_match = re.search(r'val_loss_(\d+\.?\d*)', model_path)
        if val_loss_match:
            return float(val_loss_match.group(1))
        return None
    
    def analyze_sweeps(self) -> pd.DataFrame:
        """Analyze all sweep results and create a comprehensive dataset."""
        logger.info("Starting sweep results analysis...")
        
        config_files = self.find_best_model_configs()
        
        for config_path in config_files:
            model_info = self.extract_model_info(config_path)
            if model_info:
                self.results.append(model_info)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        if df.empty:
            logger.warning("No valid model configurations found!")
            return df
        
        # Clean and process data
        df = self._clean_data(df)
        
        logger.info(f"Analyzed {len(df)} model configurations")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data."""
        # Convert numeric columns
        numeric_cols = [
            'val_loss', 'learning_rate', 'weight_decay', 'batch_size', 
            'epochs', 'grad_clip_val', 'horizon', 'window_size', 'stride',
            'delay', 'best_epoch'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values for categorical columns
        categorical_cols = ['model_name', 'dataset_name', 'loss_fn']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def find_best_models_per_dataset(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Find the best model for each dataset based on validation loss."""
        best_models = {}
        
        if 'val_loss' not in df.columns:
            logger.warning("No validation loss found in data")
            return best_models
        
        # Group by dataset and find minimum validation loss
        for dataset in df['dataset_name'].unique():
            if dataset == 'unknown':
                continue
                
            dataset_df = df[df['dataset_name'] == dataset]
            
            # Find row with minimum validation loss
            best_idx = dataset_df['val_loss'].idxmin()
            if pd.notna(best_idx):
                best_model = dataset_df.loc[best_idx].to_dict()
                best_models[dataset] = best_model
        
        return best_models
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics from the sweep results."""
        stats = {}
        
        # Overall statistics
        stats['total_experiments'] = len(df)
        stats['unique_datasets'] = df['dataset_name'].nunique()
        stats['unique_models'] = df['model_name'].nunique()
        
        # Performance statistics
        if 'val_loss' in df.columns:
            stats['validation_loss'] = {
                'mean': df['val_loss'].mean(),
                'std': df['val_loss'].std(),
                'min': df['val_loss'].min(),
                'max': df['val_loss'].max(),
                'median': df['val_loss'].median()
            }
        
        # Model-wise statistics
        model_stats = {}
        for model in df['model_name'].unique():
            if model == 'unknown':
                continue
            model_df = df[df['model_name'] == model]
            model_stats[model] = {
                'count': len(model_df),
                'datasets': model_df['dataset_name'].unique().tolist(),
                'avg_val_loss': model_df['val_loss'].mean() if 'val_loss' in model_df.columns else None,
                'best_val_loss': model_df['val_loss'].min() if 'val_loss' in model_df.columns else None
            }
        stats['model_statistics'] = model_stats
        
        # Dataset-wise statistics
        dataset_stats = {}
        for dataset in df['dataset_name'].unique():
            if dataset == 'unknown':
                continue
            dataset_df = df[df['dataset_name'] == dataset]
            dataset_stats[dataset] = {
                'count': len(dataset_df),
                'models': dataset_df['model_name'].unique().tolist(),
                'avg_val_loss': dataset_df['val_loss'].mean() if 'val_loss' in dataset_df.columns else None,
                'best_val_loss': dataset_df['val_loss'].min() if 'val_loss' in dataset_df.columns else None
            }
        stats['dataset_statistics'] = dataset_stats
        
        return stats
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create visualization plots for the analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        
        # 1. Validation loss by model and dataset
        if 'val_loss' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='dataset_name', y='val_loss', hue='model_name')
            plt.title('Validation Loss Distribution by Model and Dataset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'val_loss_by_model_dataset.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Hyperparameter distributions
        hyperparams = ['learning_rate', 'weight_decay', 'batch_size', 'model_dropout']
        available_hyperparams = [hp for hp in hyperparams if hp in df.columns]
        
        if available_hyperparams:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, param in enumerate(available_hyperparams[:4]):
                if i < 4:
                    df[param].hist(bins=20, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {param}')
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'hyperparameter_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Best models comparison
        if 'val_loss' in df.columns:
            best_models = self.find_best_models_per_dataset(df)
            if best_models:
                datasets = list(best_models.keys())
                val_losses = [best_models[d]['val_loss'] for d in datasets]
                model_names = [best_models[d]['model_name'] for d in datasets]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(datasets)), val_losses, color=['skyblue' if m == 'mlp' else 'orange' for m in model_names])
                plt.xlabel('Dataset')
                plt.ylabel('Best Validation Loss')
                plt.title('Best Model Performance by Dataset')
                plt.xticks(range(len(datasets)), datasets, rotation=45)
                
                # Add model names as labels
                for i, (bar, model) in enumerate(zip(bars, model_names)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                            model, ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'best_models_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def save_results(self, df: pd.DataFrame, stats: Dict, best_models: Dict, output_dir: str):
        """Save all analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        df.to_csv(output_dir / 'sweep_analysis_raw_data.csv', index=False)
        
        # Save statistics
        with open(output_dir / 'sweep_statistics.yaml', 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, indent=2)
        
        # Save best models
        with open(output_dir / 'best_models_per_dataset.yaml', 'w') as f:
            yaml.dump(best_models, f, default_flow_style=False, indent=2)
        
        # Create summary report
        self._create_summary_report(df, stats, best_models, output_dir)
    
    def _create_summary_report(self, df: pd.DataFrame, stats: Dict, best_models: Dict, output_dir: Path):
        """Create a human-readable summary report."""
        report_path = output_dir / 'analysis_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("SWEEP RESULTS ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Experiments: {stats['total_experiments']}\n")
            f.write(f"Unique Datasets: {stats['unique_datasets']}\n")
            f.write(f"Unique Models: {stats['unique_models']}\n\n")
            
            # Validation loss statistics
            if 'validation_loss' in stats:
                val_stats = stats['validation_loss']
                f.write("VALIDATION LOSS STATISTICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Mean: {val_stats['mean']:.4f}\n")
                f.write(f"Std: {val_stats['std']:.4f}\n")
                f.write(f"Min: {val_stats['min']:.4f}\n")
                f.write(f"Max: {val_stats['max']:.4f}\n")
                f.write(f"Median: {val_stats['median']:.4f}\n\n")
            
            # Best models per dataset
            f.write("BEST MODELS PER DATASET:\n")
            f.write("-" * 30 + "\n")
            for dataset, model_info in best_models.items():
                f.write(f"\n{dataset.upper()}:\n")
                f.write(f"  Model: {model_info.get('model_name', 'N/A')}\n")
                f.write(f"  Validation Loss: {model_info.get('val_loss', 'N/A'):.4f}\n")
                f.write(f"  Learning Rate: {model_info.get('learning_rate', 'N/A')}\n")
                f.write(f"  Batch Size: {model_info.get('batch_size', 'N/A')}\n")
                f.write(f"  Config Path: {model_info.get('config_path', 'N/A')}\n")
            
            f.write(f"\nDetailed results saved to: {output_dir}\n")

def main():
    """Main function to run the sweep analysis."""
    parser = argparse.ArgumentParser(description='Analyze hyperparameter sweep results')
    parser.add_argument(
        '--sweeps-dir',
        type=str,
        default='logs/sweeps',
        help='Directory containing sweep results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results',
        help='Directory to save analysis results'
    )
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SweepResultsAnalyzer(args.sweeps_dir)
    
    # Run analysis
    df = analyzer.analyze_sweeps()
    
    if df.empty:
        logger.error("No data found to analyze!")
        return
    
    # Generate statistics and find best models
    stats = analyzer.generate_statistics(df)
    best_models = analyzer.find_best_models_per_dataset(df)
    
    # Create visualizations
    analyzer.create_visualizations(df, args.output_dir)
    
    # Save results
    analyzer.save_results(df, stats, best_models, args.output_dir)
    
    # Print summary
    print(f"\nAnalysis completed! Results saved to: {args.output_dir}")
    print(f"Analyzed {len(df)} model configurations")
    print(f"Found best models for {len(best_models)} datasets")
    
    # Print best models summary
    if best_models:
        print("\nBest Models by Dataset:")
        for dataset, model_info in best_models.items():
            print(f"  {dataset}: {model_info['model_name']} (val_loss: {model_info.get('val_loss', 'N/A')})")

if __name__ == "__main__":
    main()