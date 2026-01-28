from pytorch_lightning.callbacks import Callback
from pathlib import Path

class MetricsHistory(Callback):
    def __init__(self, log_metrics, log_dir=None):
        super().__init__()

        self.log_dir = log_dir
        self.log_metrics = log_metrics

        # Initialize lists to store metrics
        for metric in log_metrics:
            setattr(self, f'train_{metric}', [0])
            setattr(self, f'val_{metric}', [])
            setattr(self, f'test_{metric}', [])

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        for metric in self.log_metrics:
            if f'train_{metric}' in m:
                getattr(self, f'train_{metric}').append(m[f'train_{metric}'].cpu().item())

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        for metric in self.log_metrics:
            if f'val_{metric}' in m:
                getattr(self, f'val_{metric}').append(m[f'val_{metric}'].cpu().item())

    def on_test_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        for metric in self.log_metrics:
            if f'test_{metric}' in m:
                getattr(self, f'test_{metric}').append(m[f'test_{metric}'].cpu().item())


class BestModelTracker(Callback):
    """
    Enhanced callback to track and save the best model during training.
    
    Features:
    - Integrates with SweepManager for global top-k tracking across sweeps
    - Saves models using custom predictor.save_model() method
    - Supports both local (per-run) and global (per-sweep) tracking
    - Thread-safe operations for concurrent sweep runs
    """
    
    def __init__(
        self,
        dirpath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        sweep_manager=None,
        run_id: str = None,
        run_name: str = None
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.sweep_manager = sweep_manager
        self.run_id = run_id or f"run_{hash(str(dirpath))}"
        self.run_name = run_name or f"run_{self.run_id[:8]}"
        
        # Track local best models for this run
        self.best_models = []  # List of (metric_value, epoch, model_path)
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = None
        self.config_path = None
        
        print(f"BestModelTracker initialized: monitoring '{monitor}' (mode: {mode})")
        if sweep_manager:
            print(f"  -> Connected to SweepManager for global top-{sweep_manager.top_k} tracking")
    
    def set_config_path(self, config_path: str):
        """Set the path to the configuration file for this run."""
        self.config_path = config_path
    
    def on_validation_end(self, trainer, pl_module):
        """Check if current model is the best and save it if needed."""
        metrics = trainer.callback_metrics
        
        if self.monitor not in metrics:
            return
            
        current_metric = float(metrics[self.monitor].cpu().item())
        current_epoch = trainer.current_epoch
        
        # Check if this is a new local best model
        is_local_best = False
        if self.mode == 'min':
            is_local_best = current_metric < self.best_metric_value
        else:
            is_local_best = current_metric > self.best_metric_value
            
        if is_local_best:
            self.best_metric_value = current_metric
            
            # Save the model using our custom save method
            model_filename = f"best_model_epoch_{current_epoch:03d}_{self.monitor}_{current_metric:.6f}.pth"
            model_path = self.dirpath / model_filename
            
            # Save using predictor's custom save method
            pl_module.save_model(str(model_path))
            
            # Update best model path
            self.best_model_path = str(model_path)
            
            # Add to local best models list
            self.best_models.append((current_metric, current_epoch, str(model_path)))
            
            # Keep only top_k models locally
            if self.mode == 'min':
                self.best_models.sort(key=lambda x: x[0])  # Sort by metric (ascending for min)
            else:
                self.best_models.sort(key=lambda x: x[0], reverse=True)  # Sort by metric (descending for max)
                
            # Remove excess local models
            if len(self.best_models) > self.save_top_k:
                # Delete old model files
                for _, _, old_path in self.best_models[self.save_top_k:]:
                    try:
                        Path(old_path).unlink()
                    except FileNotFoundError:
                        pass
                        
                # Keep only top_k
                self.best_models = self.best_models[:self.save_top_k]
            
            print(f"New best model saved: {model_filename} ({self.monitor}: {current_metric:.6f})")
            
            # Also save a symlink or copy as "best_model.pth" for easy access
            best_link_path = self.dirpath / "best_model.pth"
            try:
                if best_link_path.exists():
                    best_link_path.unlink()
                # Copy the best model to a standard name
                import shutil
                shutil.copy2(model_path, best_link_path)
                print(f"Best model also saved as: {best_link_path}")
            except Exception as e:
                print(f"Warning: Could not create best_model.pth link: {e}")
            
            # Register with global sweep manager if available
            if self.sweep_manager is not None:
                try:
                    # Collect additional metrics
                    additional_metrics = {}
                    for key, value in metrics.items():
                        if key != self.monitor and hasattr(value, 'cpu'):
                            try:
                                additional_metrics[key] = float(value.cpu().item())
                            except:
                                pass
                    
                    # Register with sweep manager
                    is_global_best = self.sweep_manager.register_model(
                        run_id=self.run_id,
                        run_name=self.run_name,
                        metric_value=current_metric,
                        model_path=str(model_path),
                        config_path=self.config_path or str(self.dirpath / "config.yaml"),
                        epoch=current_epoch,
                        wandb_run_id=getattr(trainer.logger, 'run_id', None) if hasattr(trainer, 'logger') else None,
                        additional_metrics=additional_metrics
                    )
                    
                    if is_global_best:
                        print(f"🎉 Model registered as global top-{self.sweep_manager.top_k}!")
                    else:
                        print(f"Model registered but not in global top-{self.sweep_manager.top_k}")
                        
                except Exception as e:
                    print(f"Warning: Could not register model with sweep manager: {e}")


class Wandb_callback(Callback):
    def __init__(self, log_metrics, log_dir=None, run=None):
        super().__init__()
        self.log_dir = log_dir
        self.run = run
        self.epoch = 0
        self.log_metrics = log_metrics

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        log_dict = {"train_"+k: m["train_"+k].cpu().item() for k in self.log_metrics}

        self.run.log(log_dict, step = self.epoch)
    
    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        log_dict = {"val_"+k: m["val_"+k].cpu().item() for k in self.log_metrics}

        self.run.log(log_dict, step = self.epoch)

        self.epoch += 1

    def on_test_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        log_dict = {"test_"+k: m["test_"+k].cpu().item() for k in self.log_metrics}

        self.run.log(log_dict, step = self.epoch)
