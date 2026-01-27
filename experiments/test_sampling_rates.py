"""
Zero-shot evaluation of pretrained time series models under varying sampling rates.

This script loads a pretrained model and evaluates its robustness to temporal
resolution degradation via resampling. No retraining is performed.

Usage:
    python test_sampling_rates.py load_model_path=/path/to/weights.pth
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from models.gru import GRU
from models.dmixer import DMixer
from models.timemixer import TimeMixer
from models.s5_model import S5

from extras.data_loader import convert_tsf_to_dataframe
from extras.predictor import WrapPredictor
from extras.metrics_logging import MetricsLogger
from extras.timeseriesdatamodule_resampled import TimeSeriesDataModuleResampled

from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

from colorama import Fore


# Sampling rate multipliers: 1.0 = original, 2.0 = half the samples, etc.
SAMPLING_RATES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def get_model(name: str):
    models = {
        'gru': GRU,
        'dmixer': DMixer,
        'timemixer': TimeMixer,
        's5': S5,
    }
    if name not in models:
        raise NotImplementedError(f"Model '{name}' not implemented. Available: {list(models.keys())}")
    return models[name]


def build_predictor(cfg: DictConfig, data_module) -> WrapPredictor:
    """Instantiate the predictor with model and loss configuration."""
    
    model_class = get_model(cfg.model.name)
    
    model_kwargs = dict(
        input_size=1,
        exog_size=0,
        output_size=1,
        weighted_graph=None,
        embedding_cfg=cfg.get('embedding'),
        horizon=cfg.dataset.horizon
    )
    model_class.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    # Loss function
    if cfg.optimizer.loss_fn == "mae":
        loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)
    elif cfg.optimizer.loss_fn == "mse":
        loss_fn = torch_metrics.MaskedMSE(compute_on_step=True)
    else:
        raise ValueError(f"Unknown loss: {cfg.optimizer.loss_fn}")

    log_metrics = MetricsLogger()
    metrics = log_metrics.filter_metrics(cfg.dataset.log_metrics)

    scheduler_class = scheduler_kwargs = None
    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)

    predictor = WrapPredictor(
        model_class=model_class,
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
    
    return predictor


def plot_results(results: dict, save_path: Path, model_name: str, dataset_name: str):
    """Generate and save plots for MSE/MAE vs sampling rate."""
    
    rates = results['sampling_rate']
    mse_vals = results['mse']
    mae_vals = results['mae']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Zero-Shot Robustness: {model_name} on {dataset_name}', fontsize=14, fontweight='bold')
    
    # MSE plot
    axes[0].plot(rates, mse_vals, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sampling Rate Multiplier', fontsize=11)
    axes[0].set_ylabel('MSE', fontsize=11)
    axes[0].set_title('Mean Squared Error vs Temporal Resolution')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(rates)
    
    # MAE plot
    axes[1].plot(rates, mae_vals, 's-', color='#A23B72', linewidth=2, markersize=8)
    axes[1].set_xlabel('Sampling Rate Multiplier', fontsize=11)
    axes[1].set_ylabel('MAE', fontsize=11)
    axes[1].set_title('Mean Absolute Error vs Temporal Resolution')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(rates)
    
    # Baseline reference line
    for ax, vals in zip(axes, [mse_vals, mae_vals]):
        if 1.0 in rates:
            idx = rates.index(1.0)
            ax.axhline(y=vals[idx], color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
            ax.legend()
    
    plt.tight_layout()
    
    plot_path = save_path / 'sampling_rate_robustness.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path / 'sampling_rate_robustness.pdf', bbox_inches='tight')
    print(Fore.GREEN + f"Plots saved to {plot_path}" + Fore.RESET)
    
    plt.show()
    

def save_results_csv(results: dict, save_path: Path):
    """Save numerical results to CSV."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    csv_path = save_path / 'sampling_rate_results.csv'
    df.to_csv(csv_path, index=False)
    print(Fore.GREEN + f"Results saved to {csv_path}" + Fore.RESET)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    
    # Validate pretrained weights path
    load_model_path = cfg.get('load_model_path')
    if load_model_path is None:
        raise ValueError("cfg.load_model_path must be specified for zero-shot testing.")
    
    print(Fore.CYAN + f"Loading pretrained weights from: {load_model_path}" + Fore.RESET)
    
    # Dataset paths
    dataset_paths = {
        "Wind": "datasets/Wind/wind_farms_minutely_dataset_with_missing_values.tsf",
        "Solar": "datasets/Solar/solar_10_minutes_dataset.tsf",
        "Electricity": "datasets/Electricity/australian_electricity_demand_dataset.tsf",
    }
    
    if cfg.dataset.name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    data_path = dataset_paths[cfg.dataset.name]
    print(f"Loading data from {data_path}...")
    
    (loaded_data, frequency, 
     forecast_horizon, contain_missing_values, 
     contain_equal_length) = convert_tsf_to_dataframe(data_path)
    
    # Transform configuration
    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {'target': StandardScaler(axis=scale_axis)}
    
    # Output directory
    output_dir = Path(cfg.run_dir) / 'sampling_rate_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = {
        'sampling_rate': [],
        'mse': [],
        'mae': [],
        'num_test_samples': []
    }
    
    # Trainer configuration (testing only)
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False,
        enable_progress_bar=True,
    )
    
    print(Fore.YELLOW + "\n" + "="*60)
    print("Starting Zero-Shot Evaluation Across Sampling Rates")
    print("="*60 + Fore.RESET)
    
    for rate in SAMPLING_RATES:
        print(Fore.CYAN + f"\n[Rate {rate:.1f}x] Preparing data module..." + Fore.RESET)
        
        # Deep copy to avoid mutations across iterations
        data_copy = copy.deepcopy(loaded_data)
        
        # Create data module with resampling
        data_module = TimeSeriesDataModuleResampled(
            data=data_copy,
            window_size=cfg.dataset.window_size,
            transform=transform,
            batch_size=cfg.optimizer.batch_size,
            frequency=frequency,
            forecast_horizon=cfg.dataset.horizon,
            contain_missing_values=contain_missing_values,
            contain_equal_length=contain_equal_length,
            workers=cfg.optimizer.num_workers,
            splits=cfg.dataset.splitting,
            resample_rate=rate,
        )
        data_module.setup()
        
        # Build predictor and load weights
        predictor = build_predictor(cfg, data_module)
        predictor.load_model(load_model_path)
        predictor.freeze()
        
        # Test
        print(f"[Rate {rate:.1f}x] Running test...")
        test_results = trainer.test(predictor, dataloaders=data_module.test_dataloader(), verbose=False)
        
        # Extract metrics
        test_metrics = test_results[0]
        mse = test_metrics.get('test_mse', test_metrics.get('test_MaskedMSE', np.nan))
        mae = test_metrics.get('test_mae', test_metrics.get('test_MaskedMAE', np.nan))
        
        results['sampling_rate'].append(rate)
        results['mse'].append(mse)
        results['mae'].append(mae)
        results['num_test_samples'].append(len(data_module.test_dataset))
        
        print(Fore.GREEN + f"[Rate {rate:.1f}x] MSE: {mse:.6f} | MAE: {mae:.6f}" + Fore.RESET)
    
    # Summary
    print(Fore.YELLOW + "\n" + "="*60)
    print("Evaluation Complete - Summary")
    print("="*60 + Fore.RESET)
    
    for i, rate in enumerate(results['sampling_rate']):
        print(f"  Rate {rate:.1f}x: MSE={results['mse'][i]:.6f}, MAE={results['mae'][i]:.6f}")
    
    # Save and plot
    save_results_csv(results, output_dir)
    plot_results(results, output_dir, cfg.model.name, cfg.dataset.name)
    
    # Save config
    cfg_path = output_dir / 'config.yaml'
    with open(cfg_path, 'w') as f:
        OmegaConf.save(cfg, f)
    print(Fore.GREEN + f"Config saved to {cfg_path}" + Fore.RESET)


if __name__ == "__main__":
    main()
