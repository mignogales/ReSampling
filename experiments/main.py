import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

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

from extras.data_loader import convert_tsf_to_dataframe
from extras.predictor import WrapPredictor
from extras.metrics_logging import MetricsLogger
from extras.callbacks import Wandb_callback
from extras.timeseriesdataset import TimeSeriesDataset
from extras.timeseriesdatamodule import TimeSeriesDataModule
from extras.notifications import notify_update

def get_model(name):
    if name == 'gru':
        return GRU
    elif name == 'dmixer':
        return DMixer
    elif name == 'timemixer':
        return TimeMixer
    elif name == 's5':
        return S5
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

    # check if save_model_weights is in cfg, if not set to False
    if 'save_model_weights' not in cfg:
        cfg.save_model_weights = False

    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg.model, resolve=True),
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

    model_kwargs = dict(input_size = 1,
                        exog_size = 0,
                        output_size = 1,
                        weighted_graph = None,
                        embedding_cfg = cfg.get('embedding'), #### changed from None to embedding_cfg
                        horizon = cfg.dataset.horizon
                        )

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
        batch_size=cfg.optimizer.batch_size,
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

    run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=cfg.wandb.entity,
                # Set the wandb project where this run will be logged.
                project=cfg.wandb.project,
                # Track all the hyperparameters and run metadata.
                config = config,
            )

    
    wandb_logger_callback = Wandb_callback(
        log_dir=cfg.run_dir,
        run=run,
        log_metrics=log_list,
    )

    trainer = Trainer(max_epochs=cfg.optimizer.epochs,
                        limit_train_batches=None,
                        default_root_dir=cfg.run_dir,
                        logger=exp_logger,  # Disable default logger
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        gradient_clip_val=cfg.optimizer.grad_clip_val,
                        callbacks=[early_stop_callback, wandb_logger_callback])
    
    ########################################
    # training                             #
    ########################################


    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    elif cfg.model.name == 'persistent':
        pass
    else:
        trainer.fit(predictor, train_dataloaders=data_module.train_dataloader(),
                    val_dataloaders=data_module.val_dataloader())
        # predictor.load_model(checkpoint_callback.best_model_path)

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