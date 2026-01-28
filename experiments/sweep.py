import torch
import hydra
from zoneinfo import ZoneInfo
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path

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

from extras.data_loader import convert_tsf_to_dataframe
from extras.predictor import WrapPredictor
from extras.metrics_logging import MetricsLogger
from extras.callbacks import Wandb_callback, BestModelTracker
from extras.timeseriesdataset import TimeSeriesDataset
from extras.timeseriesdatamodule import TimeSeriesDataModule
from extras.notifications import notify_update
from extras.sweep_manager import SweepManager

from datetime import datetime
import yaml
import sys

import os

# Note: Using Hydra's built-in 'now' resolver instead of custom registration
# Hydra automatically provides ${now:%Y-%m-%d_%H-%M-%S} functionality

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
    """Setup and create sweep directories."""
    # Create base logs directory if it doesn't exist
    base_dir = Path(cfg.logging.base_dir)
    base_dir.mkdir(exist_ok=True)
    
    # Create sweep subdirectory
    sweep_dir = base_dir / "sweeps"
    sweep_dir.mkdir(exist_ok=True)
        
    # Create a marker file to indicate the experiment type
    experiment_dir = Path(cfg.run_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Write experiment type marker
    marker_file = experiment_dir / ".experiment_type_sweep"
    marker_file.write_text(f"Experiment type: sweep\nCreated: {datetime.now()}\n")
    
    print(Fore.GREEN + f"Created sweep directory: {experiment_dir}" + Fore.RESET)
    return str(experiment_dir)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):

    # Set run_dir to sweeps directory
    cfg.run_dir = cfg.logging.sweeps_dir

    # Setup sweep directory structure
    setup_sweep_directories(cfg)

    # Initialize global sweep manager for tracking best models across all runs
    # Extract base sweep directory (parent of individual run directories)
    base_sweep_dir = Path(cfg.run_dir).parent
    sweep_manager = SweepManager(
        sweep_dir=base_sweep_dir,
        top_k=3,  # Keep only top 3 models globally
        metric_name=cfg.optimizer.monitor,
        mode='min' if 'loss' in cfg.optimizer.monitor else 'min',  # Assume loss metrics should be minimized
        cleanup_intermediate=True
    )

    # 1. Create your custom timestamped folder name
    now = datetime.now(tz=ZoneInfo('Europe/Berlin')).strftime("%Y%m%d_%H%M%S")
    custom_run_name = f"sweep_{cfg.model.name}_{now}_{wandb.util.generate_id()}"

    # 2. Initialize wandb with the 'dir' argument
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        # config=config,
        name=custom_run_name, 
        dir=cfg.run_dir,  # This dictates where the 'wandb' folder is created
        settings=wandb.Settings(start_method="thread") # Recommended for sweeps
    )

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
    )

    data_module.setup()

    ######################################
    # Model Initialization
    ######################################
    model = get_model(cfg.model.name)

    model_kwargs = dict(
        input_size=1,
        exog_size=0,
        output_size=1,
        weighted_graph=None,
        embedding_cfg=cfg.get('embedding'),
        horizon=cfg.dataset.horizon,
        window_size=cfg.dataset.window_size,
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
        batch_size=cfg.optimizer.batch_size,
    )

    exp_logger = TensorBoardLogger(save_dir=cfg.run_dir, name=cfg.run_name)

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
        dirpath=cfg.run_dir,
        monitor=cfg.optimizer.monitor,
        mode='min',
        save_top_k=1,
        sweep_manager=sweep_manager,
        run_id=run_id,
        run_name=custom_run_name
    )

    # Keep the original ModelCheckpoint for PyTorch Lightning compatibility (optional)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run_dir,
        save_top_k=1,
        monitor=cfg.optimizer.monitor,
        mode='min',
    )

    # Build config dict for wandb logging
    config = {}
    for key, value in cfg.optimizer.hparams.items():
        config[key] = value
    for key, value in cfg.dataset.items():
        config[key] = value
    config.update({
        "model": cfg.model.name,
        "optimizer": cfg.optimizer.name,
        "batch_size": cfg.optimizer.batch_size,
        "epochs": cfg.optimizer.epochs,
    })

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=config,
    )

    wandb_logger_callback = Wandb_callback(
        log_dir=cfg.run_dir,
        run=run,
        log_metrics=log_list,
    )

    if cfg.optimizer.get('grad_clip_val') is not None:
        print(Fore.GREEN + f"Using gradient clipping with value {cfg.optimizer.grad_clip_val}" + Fore.RESET)
    else:
        print(Fore.YELLOW + "Not using gradient clipping" + Fore.RESET)

    trainer = Trainer(
        max_epochs=cfg.optimizer.epochs,
        limit_train_batches=None,
        default_root_dir=cfg.run_dir,
        logger=exp_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # devices=1 if torch.cuda.is_available() else None,
        gradient_clip_val=cfg.optimizer.grad_clip_val,
        callbacks=[early_stop_callback, best_model_tracker, wandb_logger_callback]
    )

    ########################################
    # Training
    ########################################
    
    # Save the resolved config for this run
    config_save_path = os.path.join(cfg.run_dir, "resolved_config.yaml")
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, default_flow_style=False)
    
    # Set config path for the model tracker
    best_model_tracker.set_config_path(config_save_path)
    
    trainer.fit(
        predictor,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )
    # Load the best model from our custom tracker
    if best_model_tracker.best_model_path is not None:
        print(Fore.GREEN + f"Loading best model: {best_model_tracker.best_model_path}" + Fore.RESET)
        predictor.load_model(best_model_tracker.best_model_path)
    else:
        print(Fore.YELLOW + "No best model found, using final model" + Fore.RESET)

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
        'local_best_model_path': best_model_tracker.best_model_path,
        'local_best_metric_value': best_model_tracker.best_metric_value,
        'monitored_metric': cfg.optimizer.monitor,
        'total_epochs': trainer.current_epoch + 1,
        'global_rank': None,  # Will be updated if this model made it to global top-k
        'global_top_k': len(best_models),
        'sweep_config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    # Check if our model made it to global top-k
    for i, model in enumerate(best_models):
        if model.run_id == run_id:
            sweep_summary['global_rank'] = i + 1
            break
    
    # Save local sweep summary
    local_summary_path = os.path.join(cfg.run_dir, "run_summary.yaml")
    with open(local_summary_path, 'w') as f:
        yaml.dump(sweep_summary, f, default_flow_style=False)
    
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


def wandb_sweep():
    """
    Run a hyperparameter sweep with wandb.
    """
    # Get model name from command line arguments
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'gru'

    # Select sweep config based on model
    if 'dmixer' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_dmixer.yaml"
    elif 'timemixer' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_timemixer.yaml"
    elif 's5' in model_name:
        wandb_sweep_path = "./config/wandb/sweep_s5.yaml"
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

    # Name sweep with model name and timestamp
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    dict_sweep['name'] = f"sweep_{model_name}_{date_str}"

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=dict_sweep,
        project=dict_keys['project'],
        entity=dict_keys['entity'],
    )

    # Start the sweep agent
    wandb.agent(sweep_id, function=main, count=dict_sweep['count'])


if __name__ == "__main__":
    wandb_sweep()
