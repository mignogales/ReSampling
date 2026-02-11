import torch
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
import os
from pathlib import Path
from datetime import datetime

from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

from numpy import concatenate, isnan, nan_to_num
from colorama import Fore

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
from models.s4_model import S4

from extras.data_loader import convert_tsf_to_dataframe
from extras.predictor import WrapPredictor
from extras.metrics_logging import MetricsLogger
from extras.callbacks import Wandb_callback, ReshuffleTrainData, CheckGrads, ScheduledTeacherForcing
from extras.timeseriesdataset import TimeSeriesDataset
from extras.timeseriesdatamodule import TimeSeriesDataModule
from extras.notifications import notify_update

def setup_experiment_directories(cfg: DictConfig, experiment_type: str = "experiment") -> str:
    """
    Setup and create experiment directories based on the type of experiment.
    
    Args:
        cfg: Configuration object
        experiment_type: Type of experiment ('experiment', 'sweep', 'frequency_test')
    
    Returns:
        Path to the created experiment directory
    """
    # Create base logs directory if it doesn't exist
    base_dir = Path(cfg.logging.base_dir)
    base_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different experiment types
    experiment_dirs = {
        "experiment": base_dir / "experiments",
        "sweep": base_dir / "sweeps", 
        "frequency_test": base_dir / "frequency_testing"
    }
    
    for exp_type, exp_dir in experiment_dirs.items():
        exp_dir.mkdir(exist_ok=True)
        
    # Create a marker file to indicate the experiment type
    experiment_dir = Path(cfg.run_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Write experiment type marker
    marker_file = experiment_dir / f".experiment_type_{experiment_type}"
    marker_file.write_text(f"Experiment type: {experiment_type}\nCreated: {datetime.now()}\n")
    
    print(Fore.GREEN + f"Created {experiment_type} directory: {experiment_dir}" + Fore.RESET)
    return str(experiment_dir)

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
    elif name == 's4':
        return S4
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
    
def build_cfg(cfg: DictConfig):

    model_name = cfg.model.name  # Get the model name from the config
    model_config_path = f"./models/{model_name}.yaml"  # Construct the model-specific config path

    # Load the model-specific config
    model_cfg = hydra.compose(config_name=model_config_path)
    print(f"Loaded configuration for {model_name}: {model_cfg}")

    # Optimizer configuration
    optimizer_name = cfg.optimizer.name  # Get the optimizer name from the config
    optimizer_config_path = f"./optimizers/{optimizer_name}.yaml"  # Construct the optimizer-specific config path
    optimizer_cfg = hydra.compose(config_name=optimizer_config_path)
    print(f"Loaded configuration for optimizer {optimizer_name}: {optimizer_cfg}")

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):

    # Set run_dir to experiments directory
    cfg.run_dir = cfg.logging.experiments_dir

    # Setup experiment directory structure
    setup_experiment_directories(cfg, experiment_type="experiment")

    # check if save_model_weights is in cfg, if not set to False
    if 'save_model_weights' not in cfg:
        with open_dict(cfg):
            cfg.save_model_weights = False

    # which info to send to wandb. open cfg.model, cfg.optimizer, cfg.dataset and put them in a dict. open everything so nothing is nested
    config_wandb = {}
    for main_key in ['model', 'optimizer', 'dataset']:
        for key, value in cfg[main_key].items():
            if isinstance(value, dict) or isinstance(value, DictConfig):
                for sub_key, sub_value in value.items():
                    config_wandb[f"{main_key}_{sub_key}"] = sub_value
            else:
                config_wandb[f"{main_key}_{key}"] = value

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=config_wandb,
        name=cfg.wandb.name,
        # mode=cfg.wandb.mode
    )

    #######################################
    # dataset Initialization
    #######################################
    if cfg.dataset.name == "Wind":
        data_path = "datasets/Wind/wind_farms_minutely_dataset_with_missing_values.tsf"
    elif cfg.dataset.name == "Solar":
        data_path = "datasets/Solar/solar_10_minutes_dataset.tsf"
    elif cfg.dataset.name == "Electricity":
        data_path = "datasets/Electricity/australian_electricity_demand_dataset.tsf"
    elif cfg.dataset.name == "Weather":
        data_path = "datasets/Weather_austria/weather_dataset.tsf"
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    
    print(f"Loading data from {data_path}...")

    # Note: Ensure convert_tsf_to_dataframe is imported or defined
    (loaded_data, frequency, 
        forecast_horizon, contain_missing_values, 
        contain_equal_length) = convert_tsf_to_dataframe(data_path)
        
    
    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis),
    }

    cfg = notify_update(key='batch_size', wandb_keys=['batch_size'], wandb_config=wandb.config, cfg=cfg, sub_attr='optimizer')
    cfg = notify_update(key='epochs', wandb_keys=['epochs'], wandb_config=wandb.config, cfg=cfg, sub_attr='optimizer')

    # check if model.training_mode exists in cfg.model. cfg.model.training_mode could not exist
    if getattr(cfg, 'model', None) is not None and 'training_mode' in cfg.model:
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

    model_kwargs = dict(input_size = 1,
                        exog_size = 0,
                        output_size = 1,
                        weighted_graph = None,
                        embedding_cfg = cfg.get('embedding'), #### changed from None to embedding_cfg
                        horizon = cfg.dataset.horizon,
                        window_size = cfg.dataset.window_size,
                        parallel_chunking = cfg.model.parallel_chunking if 'parallel_chunking' in cfg.model else 0.0,
                        scan_method = cfg.model.get('scan_method', 'auto'),
                        )
    
    if model == S5:
        # add discretization_method from cfg.model to model_kwargs
        model_kwargs['discretization_method'] = cfg.model.get('discretization_method', 'zoh')

    model.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    cfg = notify_update(key='loss_fn', wandb_keys=['loss_fn'], wandb_config=wandb.config, cfg=cfg)

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
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

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

    exp_logger = TensorBoardLogger(save_dir=cfg.run_dir, name=cfg.run_name)
    
    ######################################
    # Training and Setting up
    ######################################

    early_stop_callback = EarlyStopping(
        monitor=cfg.optimizer.monitor,
        patience=cfg.optimizer.patience,
        mode='min'
    )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=cfg.run_dir,
    #     save_top_k=1,
    #     monitor=cfg.optimizer.monitor,
    #     mode='min',
    # )

    config = {}
    for key, value in cfg.optimizer.hparams.items():
        config[key] = value
    for key, value in cfg.dataset.items():
        config[key] = value

    wandb_logger_callback = Wandb_callback(
        log_dir=cfg.run_dir,
        run=run,
        log_metrics=log_list,
    )

    reshuffle_train_data = ReshuffleTrainData()
    
    # check_grads = CheckGrads(log_dir=cfg.run_dir)

    if model == S5:
        schedule_teacher_forcing_config = cfg.model.get('schedule_teacher_forcing', False)

        schedule_teacher_forcing = ScheduledTeacherForcing(total_epochs=cfg.optimizer.epochs, **schedule_teacher_forcing_config) if schedule_teacher_forcing_config else None

        callbacks = [early_stop_callback, wandb_logger_callback, reshuffle_train_data, schedule_teacher_forcing]
    else:
        callbacks = [early_stop_callback, wandb_logger_callback, reshuffle_train_data]

    trainer = Trainer(max_epochs=cfg.optimizer.epochs,
                        limit_train_batches=100,
                        default_root_dir=cfg.run_dir,
                        logger=exp_logger,  # Disable default logger
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        gradient_clip_val=cfg.optimizer.grad_clip_val,
                        callbacks=callbacks)

    ########################################
    # training                             #
    ########################################

    ## I comment this part due to this script being used for training only, loading models is done in test_sampling_rates.py

    # load_model_path = cfg.get('load_model_path')
    # if load_model_path is not None:
    #     predictor.load_model(load_model_path)
    # elif cfg.model.name == 'persistent':
    #     pass
    # else:

    # compile model
    predictor.compile()

    trainer.fit(predictor, train_dataloaders=data_module.train_dataloader(),
                    val_dataloaders=data_module.val_dataloader())

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()

    trainer.test(predictor, dataloaders=data_module.test_dataloader())
    
    exp_logger.finalize('success')

    if cfg.save_model_weights:
        save_path = f"{cfg.run_dir}/model_weights.pth"
        predictor.save_model(save_path)
        print(Fore.GREEN + f"Model weights saved to {save_path}" + Fore.RESET)
        # save also the config used
        cfg_path = f"{cfg.run_dir}/config.yaml"
        with open(cfg_path, 'w') as f:
            OmegaConf.save(cfg, f)
        print(Fore.GREEN + f"Config saved to {cfg_path}" + Fore.RESET)
    

if __name__ == "__main__":
    main()