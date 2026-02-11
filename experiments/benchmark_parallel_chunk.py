"""
Benchmark script for evaluating parallel_chunking configurations.

This script systematically evaluates model performance across different 
parallel_chunking settings (0 and 1), capturing training time and validation/test metrics.
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
import os
from pathlib import Path
from datetime import datetime
import time
import json
import matplotlib.pyplot as plt
import numpy as np

from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

from colorama import Fore

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Assuming these imports match your project structure
from models.gru import GRU
from models.dmixer import DMixer
from models.timemixer import TimeMixer
from models.s5_model import S5
from models.mlp import SimpleTemporalMLP
from models.dlinear import LTSFDLinear
from models.patch_tst import PatchTST

from extras.data_loader import convert_tsf_to_dataframe
from extras.predictor import WrapPredictor
from extras.metrics_logging import MetricsLogger
from extras.callbacks import Wandb_callback
from extras.timeseriesdataset import TimeSeriesDataset
from extras.timeseriesdatamodule import TimeSeriesDataModule
from extras.notifications import notify_update


def get_model(name: str):
    """Model factory function."""
    models = {
        'gru': GRU,
        'dmixer': DMixer,
        'timemixer': TimeMixer,
        's5': S5,
        'mlp': SimpleTemporalMLP,
        'dlinear': LTSFDLinear,
        'patch_tst': PatchTST,
    }
    if name not in models:
        raise NotImplementedError(f"Model {name} is not implemented.")
    return models[name]


def run_single_experiment(cfg: DictConfig, parallel_chunking: float) -> dict:
    """
    Run a single training experiment with specified parallel_chunking value.
    
    Args:
        cfg: Base configuration
        parallel_chunking: Value for parallel_chunking parameter (0 or 1)
    
    Returns:
        Dictionary containing training time and metrics
    """
    # Deep copy and modify config
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    with open_dict(cfg):
        cfg.model.parallel_chunking = parallel_chunking
        cfg.run_dir = f"{cfg.logging.experiments_dir}/parallel_chunking_{parallel_chunking}"
        cfg.run_name = f"parallel_chunking_{parallel_chunking}"
        
        if 'save_model_weights' not in cfg:
            cfg.save_model_weights = False
    
    # Create experiment directory
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    
    print(Fore.CYAN + f"\n{'='*60}" + Fore.RESET)
    print(Fore.CYAN + f"Running experiment with parallel_chunking = {parallel_chunking}" + Fore.RESET)
    print(Fore.CYAN + f"{'='*60}\n" + Fore.RESET)
    
    # Dataset initialization
    data_paths = {
        "Wind": "datasets/Wind/wind_farms_minutely_dataset_with_missing_values.tsf",
        "Solar": "datasets/Solar/solar_10_minutes_dataset.tsf",
        "Electricity": "datasets/Electricity/australian_electricity_demand_dataset.tsf",
    }
    
    if cfg.dataset.name not in data_paths:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    data_path = data_paths[cfg.dataset.name]
    print(f"Loading data from {data_path}...")
    
    (loaded_data, frequency, forecast_horizon, 
     contain_missing_values, contain_equal_length) = convert_tsf_to_dataframe(data_path)
    
    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {'target': StandardScaler(axis=scale_axis)}
    
    training_mode = getattr(cfg.model, 'training_mode', 'normal')
    
    data_module = TimeSeriesDataModule(
        data=loaded_data,
        window_size=cfg.dataset.window_size,
        transform=transform,
        batch_size=cfg.optimizer.batch_size,
        frequency=frequency,
        forecast_horizon=cfg.dataset.horizon,
        contain_missing_values=contain_missing_values,
        contain_equal_length=contain_equal_length,
        workers=cfg.optimizer.num_workers,
        splits=cfg.dataset.splitting,
        training_mode=training_mode
    )
    data_module.setup()
    
    # Model initialization
    model = get_model(cfg.model.name)
    
    model_kwargs = dict(
        input_size=1,
        exog_size=0,
        output_size=1,
        weighted_graph=None,
        embedding_cfg=cfg.get('embedding'),
        horizon=cfg.dataset.horizon,
        window_size=cfg.dataset.window_size,
        parallel_chunking=cfg.model.parallel_chunking,
    )
    
    model.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)
    
    # Loss and metrics
    loss_fn_map = {
        "mae": torch_metrics.MaskedMAE(compute_on_step=True),
        "mse": torch_metrics.MaskedMSE(compute_on_step=True),
    }
    
    if cfg.optimizer.loss_fn not in loss_fn_map:
        raise ValueError(f"Unknown loss type: {cfg.optimizer.loss_fn}")
    
    loss_fn = loss_fn_map[cfg.optimizer.loss_fn]
    
    log_metrics = MetricsLogger()
    metrics = log_metrics.filter_metrics(cfg.dataset.log_metrics)
    
    # Scheduler
    scheduler_class = scheduler_kwargs = None
    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    
    # Predictor
    predictor = WrapPredictor(
        model_class=model,
        n_nodes=0,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=False,
        transform=data_module.transform,
        batch_size=cfg.optimizer.batch_size,
    )
    
    exp_logger = TensorBoardLogger(save_dir=cfg.run_dir, name=cfg.run_name)
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor=cfg.optimizer.monitor,
        patience=cfg.optimizer.patience,
        mode='min'
    )
    
    # WandB initialization (optional)
    run = None
    callbacks = [early_stop_callback]
    
    if cfg.wandb.enable:
        config = {**dict(cfg.optimizer.hparams), **dict(cfg.dataset)}
        config['parallel_chunking'] = parallel_chunking
        
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=config,
            name=f"{cfg.wandb.name}_pc{parallel_chunking}",
            reinit=True
        )
        
        wandb_logger_callback = Wandb_callback(
            log_dir=cfg.run_dir,
            run=run,
            log_metrics=cfg.dataset.log_metrics,
        )
        callbacks.append(wandb_logger_callback)
    
    trainer = Trainer(
        max_epochs=cfg.optimizer.epochs,
        limit_train_batches=None,
        default_root_dir=cfg.run_dir,
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        gradient_clip_val=cfg.optimizer.grad_clip_val,
        callbacks=callbacks
    )
    
    # Training with timing
    start_time = time.time()
    trainer.fit(
        predictor,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )
    training_time = time.time() - start_time
    
    # Testing
    predictor.freeze()
    test_results = trainer.test(predictor, dataloaders=data_module.test_dataloader())
    
    exp_logger.finalize('success')
    
    if run is not None:
        wandb.finish()
    
    # Collect results
    results = {
        'parallel_chunking': parallel_chunking,
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'test_metrics': test_results[0] if test_results else {},
        'best_val_metric': trainer.callback_metrics.get(cfg.optimizer.monitor, None),
        'epochs_trained': trainer.current_epoch,
    }
    
    # Convert tensors to floats for JSON serialization
    for key, value in results['test_metrics'].items():
        if isinstance(value, torch.Tensor):
            results['test_metrics'][key] = value.item()
    
    if isinstance(results['best_val_metric'], torch.Tensor):
        results['best_val_metric'] = results['best_val_metric'].item()
    
    print(Fore.GREEN + f"\nResults for parallel_chunking = {parallel_chunking}:" + Fore.RESET)
    print(f"  Training time: {training_time:.2f}s ({training_time/60:.2f}min)")
    print(f"  Test metrics: {results['test_metrics']}")
    
    return results


def plot_results(results: list, output_dir: str):
    """
    Generate comparative plots for benchmark results.
    
    Args:
        results: List of result dictionaries from experiments
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    parallel_chunkings = [r['parallel_chunking'] for r in results]
    training_times = [r['training_time_minutes'] for r in results]
    
    # Extract test metrics (handle different possible metric names)
    test_metrics_keys = list(results[0]['test_metrics'].keys()) if results[0]['test_metrics'] else []
    
    fig, axes = plt.subplots(1, 2 + len(test_metrics_keys), figsize=(6 * (2 + len(test_metrics_keys)), 5))
    if len(test_metrics_keys) == 0:
        axes = [axes] if not hasattr(axes, '__len__') else list(axes)
    
    # Plot 1: Training Time
    ax = axes[0]
    bars = ax.bar(parallel_chunkings, training_times, color=['#2ecc71', '#3498db'], edgecolor='black', linewidth=1.2)
    ax.set_xlabel('parallel_chunking', fontsize=12)
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(parallel_chunkings)
    
    for bar, t in zip(bars, training_times):
        ax.annotate(f'{t:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Speedup (if both values exist)
    ax = axes[1]
    if len(training_times) == 2 and training_times[0] > 0:
        speedup = training_times[0] / training_times[1] if training_times[1] > 0 else 0
        ax.bar(['Speedup'], [speedup], color='#e74c3c', edgecolor='black', linewidth=1.2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title(f'Speedup (pc=0 → pc=1): {speedup:.2f}x', fontsize=14, fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor speedup calculation', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Speedup', fontsize=14, fontweight='bold')
    
    # Plot test metrics
    for i, metric_key in enumerate(test_metrics_keys):
        ax = axes[2 + i]
        metric_values = [r['test_metrics'].get(metric_key, 0) for r in results]
        bars = ax.bar(parallel_chunkings, metric_values, color=['#9b59b6', '#f39c12'], edgecolor='black', linewidth=1.2)
        ax.set_xlabel('parallel_chunking', fontsize=12)
        ax.set_ylabel(metric_key, fontsize=12)
        ax.set_title(f'Test {metric_key}', fontsize=14, fontweight='bold')
        ax.set_xticks(parallel_chunkings)
        
        for bar, v in zip(bars, metric_values):
            ax.annotate(f'{v:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_path / 'parallel_chunking_benchmark.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(Fore.GREEN + f"\nPlot saved to: {plot_path}" + Fore.RESET)
    return str(plot_path)


def save_results_json(results: list, output_dir: str):
    """Save results to JSON file."""
    output_path = Path(output_dir) / 'benchmark_results.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(Fore.GREEN + f"Results saved to: {output_path}" + Fore.RESET)
    return str(output_path)


def load_existing_results(output_dir: str) -> list:
    """Load existing results from JSON file if it exists."""
    output_path = Path(output_dir) / 'benchmark_results.json'
    
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            print(Fore.YELLOW + f"Loaded {len(results)} existing results from: {output_path}" + Fore.RESET)
            return results
        except Exception as e:
            print(Fore.YELLOW + f"Could not load existing results: {e}" + Fore.RESET)
            return []
    return []


def is_experiment_completed(results: list, parallel_chunking_value: float) -> bool:
    """Check if experiment with given parallel_chunking value is already completed."""
    for result in results:
        if abs(result['parallel_chunking'] - parallel_chunking_value) < 1e-6:
            return True
    return False


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    """
    Main benchmark function.
    
    Iterates through parallel_chunking values [0, 1], runs training,
    and generates comparative visualizations.
    """
    print(Fore.CYAN + "\n" + "="*70 + Fore.RESET)
    print(Fore.CYAN + "  parallel_chunking BENCHMARK" + Fore.RESET)
    print(Fore.CYAN + "="*70 + "\n" + Fore.RESET)
    
    # Setup output directory
    benchmark_dir = Path(cfg.logging.base_dir) / "benchmarks" / f"parallel_chunking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing results (if resuming)
    all_results = load_existing_results(str(benchmark_dir))
    
    parallel_chunking_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for pc_value in parallel_chunking_values:
        # Check if this experiment is already completed
        if is_experiment_completed(all_results, pc_value):
            print(Fore.YELLOW + f"\nSkipping parallel_chunking={pc_value} (already completed)" + Fore.RESET)
            continue
        
        try:
            result = run_single_experiment(cfg, pc_value)
            all_results.append(result)
            
            # Save results immediately after each successful experiment
            save_results_json(all_results, str(benchmark_dir))
            print(Fore.GREEN + f"Progress: {len(all_results)}/{len(parallel_chunking_values)} experiments completed\n" + Fore.RESET)
            
        except Exception as e:
            print(Fore.RED + f"Error running experiment with parallel_chunking={pc_value}: {e}" + Fore.RESET)
            import traceback
            traceback.print_exc()
    
    if all_results:
        # Save results and generate plots
        json_path = save_results_json(all_results, str(benchmark_dir))
        plot_path = plot_results(all_results, str(benchmark_dir))
        
        # Print summary
        print(Fore.CYAN + "\n" + "="*70 + Fore.RESET)
        print(Fore.CYAN + "  BENCHMARK SUMMARY" + Fore.RESET)
        print(Fore.CYAN + "="*70 + "\n" + Fore.RESET)
        
        for r in all_results:
            print(f"parallel_chunking = {r['parallel_chunking']}:")
            print(f"  Training time: {r['training_time_minutes']:.2f} minutes")
            print(f"  Epochs trained: {r['epochs_trained']}")
            print(f"  Test metrics: {r['test_metrics']}")
            print()
        
        if len(all_results) == 2:
            t0, t1 = all_results[0]['training_time_minutes'], all_results[1]['training_time_minutes']
            if t1 > 0:
                print(Fore.GREEN + f"Speedup (pc=0 → pc=1): {t0/t1:.2f}x" + Fore.RESET)
        
        print(Fore.GREEN + f"\nAll outputs saved to: {benchmark_dir}" + Fore.RESET)
    else:
        print(Fore.RED + "No experiments completed successfully." + Fore.RESET)


if __name__ == "__main__":
    main()
