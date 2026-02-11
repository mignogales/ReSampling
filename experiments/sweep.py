import torch
import hydra
from zoneinfo import ZoneInfo
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path
from datetime import datetime
import gc

from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

from numpy import concatenate, isnan, nan_to_num
from colorama import Fore, Style, init
init()

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.gru import GRU
from models.dmixer import DMixer
from models.timemixer import TimeMixer
from models.s5_model import S5
from models.mlp import SimpleTemporalMLP
from models.dlinear import LTSFDLinear
from models.patch_tst import PatchTST
from models.flowstate import FlowState
from models.s5_model_no_D import S5 as S5_no_D

from extras.data_loader import convert_tsf_to_dataframe
from extras.predictor import WrapPredictor
from extras.metrics_logging import MetricsLogger
from extras.callbacks import Wandb_callback, BestModelTracker, ReshuffleTrainData, ScheduledTeacherForcing
from extras.timeseriesdataset import TimeSeriesDataset
from extras.timeseriesdatamodule import TimeSeriesDataModule
from extras.notifications import notify_update
from extras.sweep_manager import SweepManager

import yaml
import sys

import os

def get_model(name):
    if name == 'gru':
        return GRU
    elif name == 'dmixer':
        return DMixer
    elif name == 'timemixer':
        return TimeMixer
    elif name == 's5':
        return S5
    elif name == 'mlp':
        return SimpleTemporalMLP
    elif name == 'dlinear':
        return LTSFDLinear
    elif name == 'patch_tst':
        return PatchTST
    elif name == 'flowstate':
        return FlowState
    elif name == 's5_no_D':
        return S5_no_D
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")


def build_cfg(cfg: DictConfig):
    model_name = cfg.model.name
    model_config_path = f"./models/{model_name}.yaml"
    model_cfg = hydra.compose(config_name=model_config_path)
    print(f"Loaded configuration for {model_name}: {model_cfg}")

    optimizer_name = cfg.optimizer.name
    optimizer_config_path = f"./optimizers/{optimizer_name}.yaml"
    optimizer_cfg = hydra.compose(config_name=optimizer_config_path)
    print(f"Loaded configuration for optimizer {optimizer_name}: {optimizer_cfg}")


def setup_sweep_directories(cfg: DictConfig) -> str:
    """
    DEPRECATED: This function is no longer needed.
    The SweepManager creates all necessary directories automatically.
    Keeping for backwards compatibility only.
    """
    # This function is now a no-op - SweepManager handles directory creation
    return str(cfg.run_dir)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):

    # Get or create the global sweep manager for this model+dataset combination
    sweep_manager = get_or_create_sweep_manager(cfg.model.name, cfg.dataset.name, cfg)

    # Set run_dir to the sweep-specific directory with window_size and horizon
    sweep_key = f"{cfg.model.name}_{cfg.dataset.name}"
    window_horizon_subfolder = f"{cfg.dataset.window_size}_{cfg.dataset.horizon}"
    base_logs_dir = Path(cfg.logging.base_dir)
    sweep_dir = base_logs_dir / "sweeps" / f"sweep_{sweep_key}" / window_horizon_subfolder

    # Create run name using model + dataset + wandb ID
    custom_run_name = f"sweep_{cfg.model.name}_{cfg.dataset.name}_{wandb.util.generate_id()}"

    # Create run-specific subdirectory for this sweep iteration
    run_dir = sweep_dir / custom_run_name  # Use sweep_dir directly to avoid nesting
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Update cfg.run_dir to point to sweep base directory (for SweepManager)
    cfg.run_dir = str(sweep_dir)
    
    print(Fore.CYAN + f"Created run directory: {run_dir}" + Fore.RESET)

    # CRITICAL: Initialize wandb FIRST before accessing wandb.config
    # Build initial config dict
    initial_config = {
        "model": cfg.model.name,
        "dataset": cfg.dataset.name,
        "optimizer": cfg.optimizer.name,
        "batch_size": cfg.optimizer.batch_size,
        "epochs": cfg.optimizer.epochs,
        "loss_fn": cfg.optimizer.loss_fn,
    }
    
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=initial_config,
        name=custom_run_name,
        dir=str(run_dir),  # Use run-specific directory
        settings=wandb.Settings(start_method="thread")
    )
    
    print(Fore.CYAN + f"Initialized W&B run: {custom_run_name}" + Fore.RESET)

    #######################################
    # Dataset Initialization
    #######################################
    if cfg.dataset.name == "Wind":
        data_path = "datasets/Wind/wind_farms_minutely_dataset_with_missing_values.tsf"
    elif cfg.dataset.name == "Solar":
        data_path = "datasets/Solar/solar_10_minutes_dataset.tsf"
    elif cfg.dataset.name == "Electricity":
        data_path = "datasets/Electricity/australian_electricity_demand_dataset.tsf"
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    print(f"Loading data from {data_path}...")

    (loaded_data, frequency, 
        forecast_horizon, contain_missing_values, 
        contain_equal_length) = convert_tsf_to_dataframe(data_path)

    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis),
    }

    # Update config with wandb sweep parameters
    cfg = notify_update(key='batch_size', wandb_keys=['batch_size'], wandb_config=wandb.config, cfg=cfg, sub_attr='optimizer')
    cfg = notify_update(key='epochs', wandb_keys=['epochs'], wandb_config=wandb.config, cfg=cfg, sub_attr='optimizer')
    cfg = notify_update(key='loss_fn', wandb_keys=['loss_fn'], wandb_config=wandb.config, cfg=cfg, sub_attr='optimizer')

    # check if model.training_mode exists in cfg.model. cfg.model.training_mode could not exist
    if 'training_mode' in cfg.model:
        training_mode = cfg.model.training_mode
        print(Fore.CYAN + f"Using training mode from config: {training_mode}" + Fore.RESET)
    else:
        training_mode = "normal"
        print(Fore.CYAN + f"No training mode specified in config, using default: {training_mode}" + Fore.RESET)

    data_module = TimeSeriesDataModule(
        data=loaded_data,
        window_size=cfg.dataset.window_size,
        transform=transform,
        batch_size=cfg.dataset.batch_size,
        frequency=frequency,
        forecast_horizon=cfg.dataset.horizon,
        contain_missing_values=contain_missing_values,
        contain_equal_length=contain_equal_length,
        workers=cfg.optimizer.num_workers,
        splits=cfg.dataset.splitting,
        training_mode=training_mode
    )

    data_module.setup()

    ######################################
    # Model Initialization
    ######################################
    model = get_model(cfg.model.name)

    # check if parallel_chunking exists in cfg.model
    if 'parallel_chunking' in cfg.model:
        parallel_chunking = cfg.model.parallel_chunking
    else:
        parallel_chunking = 0.0

    model_kwargs = dict(
        input_size=1,
        exog_size=0,
        output_size=1,
        weighted_graph=None,
        embedding_cfg=cfg.get('embedding'),
        horizon=cfg.dataset.horizon,
        window_size=cfg.dataset.window_size,
        parallel_chunking=parallel_chunking,
        scan_method = cfg.model.get('scan_method', 'auto'),
    )

    model.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    # Update model kwargs with wandb config
    # for key in wandb.config.keys():
    #     if key in model_kwargs.keys():
    #         model_kwargs[key] = wandb.config[key]
    #         print(Fore.GREEN + f"Updated model kwargs: {key} = {wandb.config[key]}")
    #     elif key in cfg.model.hparams.keys():
    #         model_kwargs[key] = wandb.config[key]
    #         cfg.model.hparams[key] = wandb.config[key]
    #         print(Fore.GREEN + f"Updated model hparams: {key} = {wandb.config[key]}")

    # use notify_update to update model hyperparameters from wandb config
    for key in cfg.model.hparams.keys():
        cfg = notify_update(key=key, wandb_keys=[key], wandb_config=wandb.config, cfg=cfg, sub_attr='model.hparams')
        model_kwargs[key] = cfg.model.hparams[key]

    ########################################
    # Predictor Setup
    ########################################
    if cfg.optimizer.loss_fn == "mae":
        base_loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)
    elif cfg.optimizer.loss_fn == "mse":
        base_loss_fn = torch_metrics.MaskedMSE(compute_on_step=True)
    else:
        raise ValueError(f"Unknown loss type: {cfg.optimizer.loss_fn}")

    loss_fn = base_loss_fn

    log_list = cfg.dataset.log_metrics
    log_metrics = MetricsLogger()
    metrics = log_metrics.filter_metrics(log_list)

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    optimizer_kwargs = dict(cfg.optimizer.hparams)

    # Update optimizer kwargs with wandb config
    if 'optimizer' in wandb.config.keys():
        cfg.optimizer.name = wandb.config['optimizer']
        print(Fore.GREEN + f"Updated optimizer: {cfg.optimizer.name}" + Fore.RESET)

    for key in wandb.config.keys():
        if key in optimizer_kwargs.keys():
            optimizer_kwargs[key] = wandb.config[key]
            cfg.optimizer.hparams[key] = wandb.config[key]
            print(Fore.GREEN + f"Updated optimizer kwargs: {key} = {wandb.config[key]}" + Fore.RESET)

    predictor_fn = WrapPredictor

    predictor = predictor_fn(
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
        batch_size=cfg.dataset.batch_size,
    )

    exp_logger = TensorBoardLogger(save_dir=cfg.run_dir, name=f"sweep_{cfg.model.name}_{cfg.dataset.name}")

    ######################################
    # Training Setup
    ######################################
    early_stop_callback = EarlyStopping(
        monitor=cfg.optimizer.monitor,
        patience=cfg.optimizer.patience,
        mode='min'
    )

    # Use our custom BestModelTracker with SweepManager integration
    run_id = wandb.run.id if wandb.run else custom_run_name
    best_model_tracker = BestModelTracker(
        dirpath=str(run_dir),  # Save to run-specific directory
        monitor=cfg.optimizer.monitor,
        mode='min',
        save_top_k=1,
        sweep_manager=sweep_manager,
        run_id=run_id,
        run_name=custom_run_name
    )

    wandb_logger_callback = Wandb_callback(
        log_dir=cfg.run_dir,
        run=run,
        log_metrics=log_list,
    )

    reshuffle_train_data = ReshuffleTrainData()

    if model == S5:
        schedule_teacher_forcing_config = cfg.model.get('schedule_teacher_forcing', False)

        schedule_teacher_forcing = ScheduledTeacherForcing(total_epochs=cfg.optimizer.epochs, **schedule_teacher_forcing_config) if schedule_teacher_forcing_config else ScheduledTeacherForcing(total_epochs=cfg.optimizer.epochs, mode="None")  

        callbacks = [early_stop_callback, wandb_logger_callback, reshuffle_train_data, schedule_teacher_forcing]
    else:
        callbacks = [early_stop_callback, wandb_logger_callback, reshuffle_train_data]

    if cfg.optimizer.get('grad_clip_val') is not None:
        print(Fore.GREEN + f"Using gradient clipping with value {cfg.optimizer.grad_clip_val}" + Fore.RESET)
    else:
        print(Fore.YELLOW + "Not using gradient clipping" + Fore.RESET)

    trainer = Trainer(
        max_epochs=cfg.optimizer.epochs,
        limit_train_batches=100,
        default_root_dir=str(run_dir),  # Use run-specific directory
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # devices=1 if torch.cuda.is_available() else None,
        gradient_clip_val=cfg.optimizer.grad_clip_val,
        callbacks=callbacks  # Removed checkpoint_callback
    )

    ########################################
    # Training
    ########################################
    
    # Build the actual model config that will be saved with the best model
    # This contains everything needed to reproduce/load this exact model
    # Use OmegaConf.to_container() to convert DictConfig to plain Python dicts
    model_config = {
        'metadata': {
            'run_id': run_id,
            'run_name': custom_run_name,
            'model_name': cfg.model.name,
            'dataset_name': cfg.dataset.name,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'model': {
            'name': cfg.model.name,
            'hparams': OmegaConf.to_container(cfg.model.hparams, resolve=True),
            'input_size': model_kwargs.get('input_size', 1),
            'output_size': model_kwargs.get('output_size', 1),
            'horizon': cfg.dataset.horizon,
            'window_size': cfg.dataset.window_size,
        },
        'training': {
            'optimizer': cfg.optimizer.name,
            'optimizer_hparams': OmegaConf.to_container(cfg.optimizer.hparams, resolve=True),
            'loss_fn': cfg.optimizer.loss_fn,
            'batch_size': cfg.optimizer.batch_size,
            'max_epochs': cfg.optimizer.epochs,
            'monitor': cfg.optimizer.monitor,
            'patience': cfg.optimizer.patience,
            'grad_clip_val': cfg.optimizer.get('grad_clip_val'),
        },
        'dataset': {
            'name': cfg.dataset.name,
            'window_size': cfg.dataset.window_size,
            'horizon': cfg.dataset.horizon,
            'splitting': OmegaConf.to_container(cfg.dataset.splitting, resolve=True) if cfg.dataset.get('splitting') else None,
        },
    }
    
    # Add lr_scheduler if present
    if cfg.get('lr_scheduler') is not None:
        model_config['training']['lr_scheduler'] = {
            'name': cfg.lr_scheduler.name,
            'hparams': OmegaConf.to_container(cfg.lr_scheduler.hparams, resolve=True),
        }
    
    # Save the model config for this run in the all_configs folder
    sweep_base_dir = Path(cfg.run_dir)
    all_configs_dir = sweep_base_dir / "all_configs"
    config_save_path = all_configs_dir / f"{custom_run_name}_config.yaml"
    
    # Ensure directory exists
    all_configs_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_save_path, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)
    
    # Set config path for the model tracker
    best_model_tracker.set_config_path(str(config_save_path))
    
    trainer.fit(
        predictor,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )

    # Load the best model from this run before testing
    # if best_model_tracker.best_model_path is not None:
    #     print(Fore.GREEN + f"Loading best model: {best_model_tracker.best_model_path}" + Fore.RESET)
    #     predictor.load_model(best_model_tracker.best_model_path)
    # else:
    #     print(Fore.YELLOW + "No best model found, using final model" + Fore.RESET)

    ########################################
    # Testing
    ########################################
    predictor.freeze()
    trainer.test(predictor, dataloaders=data_module.test_dataloader())

    exp_logger.finalize('success')

    # Get global sweep results
    best_models = sweep_manager.get_best_models()
    
    # Log sweep summary information
    sweep_summary = {
        'run_id': run_id,
        'run_name': custom_run_name,
        'model_name': cfg.model.name,
        'dataset_name': cfg.dataset.name,
        'local_best_model_path': best_model_tracker.best_model_path,
        'local_best_metric_value': best_model_tracker.best_metric_value,
        'monitored_metric': cfg.optimizer.monitor,
        'total_epochs': trainer.current_epoch + 1,
        'global_rank': None,  # Will be updated if this model made it to global top-k
        'global_top_k': len(best_models),
        'sweep_config': OmegaConf.to_container(cfg, resolve=True),
        'wandb_config': dict(wandb.config) if wandb.run else {},
        'hyperparameters': {
            'learning_rate': cfg.optimizer.hparams.get('lr', 'N/A'),
            'batch_size': cfg.optimizer.batch_size,
            'optimizer': cfg.optimizer.name,
            'loss_function': cfg.optimizer.loss_fn,
            **cfg.model.hparams
        }
    }
    
    # Check if our model made it to global top-k
    for i, model in enumerate(best_models):
        if model.run_id == run_id:
            sweep_summary['global_rank'] = i + 1
            break
    
    # Save local sweep summary in the sweep_data folder
    sweep_data_dir = Path(cfg.run_dir) / "sweep_data"
    sweep_data_dir.mkdir(parents=True, exist_ok=True)
    
    local_summary_path = sweep_data_dir / f"{custom_run_name}_summary.yaml"
    with open(local_summary_path, 'w') as f:
        yaml.dump(sweep_summary, f, default_flow_style=False)
    
    # Also save a global sweep status file
    global_sweep_status = {
        'sweep_key': sweep_key,
        'total_runs': len(sweep_manager.get_all_models()) if hasattr(sweep_manager, 'get_all_models') else 'N/A',
        'best_models': [
            {
                'rank': i + 1,
                'run_id': model.run_id,
                'metric_value': model.metric_value,
                'run_name': getattr(model, 'run_name', 'N/A')
            }
            for i, model in enumerate(best_models)
        ],
        'last_updated': datetime.fromtimestamp(wandb.run.start_time).strftime('%Y-%m-%d %H:%M:%S') if wandb.run and wandb.run.start_time else 'N/A'
    }
    
    global_status_path = sweep_data_dir / f"global_sweep_status.yaml"
    with open(global_status_path, 'w') as f:
        yaml.dump(global_sweep_status, f, default_flow_style=False)
    
    print(Fore.CYAN + f"Run Summary:" + Fore.RESET)
    print(f"  Run: {custom_run_name}")
    print(f"  Local best {cfg.optimizer.monitor}: {best_model_tracker.best_metric_value:.6f}")
    if sweep_summary['global_rank']:
        print(f"  🎉 Global rank: {sweep_summary['global_rank']}/{sweep_manager.top_k}")
    else:
        print(f"  Global rank: Not in top-{sweep_manager.top_k}")
    print(f"  Summary saved to: {local_summary_path}")

    # Log to wandb
    if best_model_tracker.best_model_path:
        wandb.log({
            "local_best_model_path": best_model_tracker.best_model_path,
            "local_best_metric_value": best_model_tracker.best_metric_value,
            "global_rank": sweep_summary['global_rank'] or 999,  # Use high number if not in top-k
            "is_global_top_k": sweep_summary['global_rank'] is not None
        })
        
        # Only save model to wandb if it's in global top-k (saves space)
        if sweep_summary['global_rank']:
            wandb.save(best_model_tracker.best_model_path)
            print(f"🎉 Model saved to W&B (global rank {sweep_summary['global_rank']})")

    # Optional: Save the final resolved config to the run directory
    wandb.save(local_summary_path)
    wandb.save(config_save_path)
    
    # Print global sweep status
    print(f"\n{Fore.YELLOW}Global Sweep Status:{Fore.RESET}")
    sweep_manager.print_summary()
    
    # Clean up orphaned run directories in the sweep root
    try:
        sweep_root = Path(cfg.run_dir)
        orphaned_count = 0
        
        # Clean up orphaned run directories (subdirectories starting with sweep_)
        for run_subdir in sweep_root.iterdir():
            if run_subdir.is_dir() and run_subdir.name.startswith('sweep_'):
                # Skip the special directories we want to keep
                if run_subdir.name in ['best_models', 'all_configs', 'sweep_data']:
                    continue
                
                # Check if this directory is referenced by a top-k model
                is_best = any(
                    str(run_subdir) in str(model.model_path) 
                    for model in sweep_manager.get_best_models()
                )
                
                if not is_best:
                    import shutil
                    shutil.rmtree(run_subdir)
                    orphaned_count += 1
                    print(f"🗑️  Cleaned up orphaned run directory: {run_subdir.name}")
        
        # Also clean up any loose checkpoint files in the root (shouldn't exist but just in case)
        for ckpt_file in sweep_root.glob("*.ckpt"):
            is_best = any(
                str(ckpt_file) in str(model.model_path) 
                for model in sweep_manager.get_best_models()
            )
            if not is_best:
                ckpt_file.unlink()
                orphaned_count += 1
                print(f"🗑️  Cleaned up orphaned checkpoint: {ckpt_file.name}")
        
        # Also clean up .pth files
        for pth_file in sweep_root.glob("*.pth"):
            is_best = any(
                str(pth_file) in str(model.model_path) 
                for model in sweep_manager.get_best_models()
            )
            if not is_best:
                pth_file.unlink()
                orphaned_count += 1
                print(f"🗑️  Cleaned up orphaned checkpoint: {pth_file.name}")
        
        if orphaned_count > 0:
            print(Fore.GREEN + f"✅ Cleaned up {orphaned_count} orphaned item(s)" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + f"⚠️  Error during cleanup: {e}" + Fore.RESET)
    
    # CRITICAL: Always finish wandb run to prevent storage leaks
    wandb.finish()

    # Force cleanup of data module resources
    # if hasattr(data_module, 'teardown'):
    #     data_module.teardown()
    
    # Force garbage collection to release file handles
    gc.collect()
    
    # close any other thing that needs closing
    print(Fore.CYAN + "Finished W&B run and cleanup." + Fore.RESET)



# Global sweep managers dictionary to track different dataset+model combinations
_global_sweep_managers = {}

def get_or_create_sweep_manager(model_name: str, dataset_name: str, cfg: DictConfig) -> SweepManager:
    """Get or create a sweep manager for a specific model+dataset combination."""
    window_size = cfg.dataset.window_size
    horizon = cfg.dataset.horizon
    key = f"{model_name}_{dataset_name}_{window_size}_{horizon}"
    
    if key not in _global_sweep_managers:
        # Create sweep-specific directory based on model+dataset combination
        base_logs_dir = Path(cfg.logging.base_dir) 
        window_horizon_subfolder = f"{window_size}_{horizon}"
        sweep_dir = base_logs_dir / "sweeps" / f"sweep_{model_name}_{dataset_name}" / window_horizon_subfolder
        sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized subdirectories
        (sweep_dir / "all_configs").mkdir(exist_ok=True)
        (sweep_dir / "best_models").mkdir(exist_ok=True)
        (sweep_dir / "best_configs").mkdir(exist_ok=True)
        (sweep_dir / "sweep_data").mkdir(exist_ok=True)
        
        _global_sweep_managers[key] = SweepManager(
            sweep_dir=sweep_dir,
            top_k=3,
            metric_name=cfg.optimizer.monitor,
            mode='min' if 'loss' in cfg.optimizer.monitor.lower() else 'max',
            cleanup_intermediate=True
        )
        print(Fore.CYAN + f"Created sweep manager for {key} in {sweep_dir}" + Fore.RESET)
        print(f"  📁 Subdirectories: all_configs, best_models, best_configs, sweep_data")
    
    return _global_sweep_managers[key]


def wandb_sweep():
    """
    Run a hyperparameter sweep with wandb.
    """
    # Get model name and dataset name from command line arguments
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'gru'
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'Wind'

    # Select sweep config based on model
    if 'dmixer' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_dmixer.yaml"
    elif 'timemixer' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_timemixer.yaml"
    elif 's5' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_s5.yaml"
    elif 'dlinear' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_dlinear.yaml"
    elif 'patch_tst' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_patch_tst.yaml"
    elif 'flowstate' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_flowstate.yaml"
    else:
        wandb_sweep_path = "./config/wandb/sweep.yaml"

    wandb_keys_path = "./config/wandb/keys.yaml"

    with open(wandb_sweep_path, 'r') as f:
        dict_sweep = yaml.safe_load(f)
    with open(wandb_keys_path, 'r') as f:
        dict_keys = yaml.safe_load(f)

    dict_sweep = dict_sweep['sweep']

    wandb.login(key=dict_keys['key'])

    for key in dict_sweep.keys():
        print(f"Setting sweep parameter {key} to {dict_sweep[key]}")

    # Name sweep with model name and dataset name
    dict_sweep['name'] = f"sweep_{model_name}_{dataset_name}"

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=dict_sweep,
        project=dict_keys['project'],
        entity=dict_keys['entity'],
    )

    print(Fore.GREEN + f"Started sweep {sweep_id} for {model_name} on {dataset_name}" + Fore.RESET)

    # Start the sweep agent
    wandb.agent(sweep_id, function=main, count=dict_sweep['count'])


if __name__ == "__main__":
    wandb_sweep()
