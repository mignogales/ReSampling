"""
Sweep management utilities for handling model checkpoints across entire sweeps.

This module provides functionality to:
1. Track the best models globally across a sweep
2. Clean up intermediate checkpoints 
3. Maintain only the top-k best models
4. Organize sweep results and metadata
"""

import os
import json
try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Some features may not work.")
    yaml = None
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
try:
    import fcntl
except ImportError:
    # fcntl is not available on Windows
    fcntl = None
import time


@dataclass
class ModelResult:
    """Container for model run results."""
    run_id: str
    run_name: str
    metric_value: float
    metric_name: str
    model_path: str
    config_path: str
    epoch: int
    timestamp: str
    wandb_run_id: Optional[str] = None
    additional_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class SweepManager:
    """
    Global sweep manager that tracks the best models across all runs in a sweep.
    
    This class:
    - Maintains a global ranking of the best models
    - Handles cleanup of intermediate checkpoints
    - Saves only the top-k best models globally
    - Provides thread-safe operations for concurrent sweep runs
    """
    
    def __init__(
        self,
        sweep_dir: Union[str, Path],
        top_k: int = 3,
        metric_name: str = "val_loss",
        mode: str = "min",
        cleanup_intermediate: bool = True
    ):
        """
        Initialize the sweep manager.
        
        Args:
            sweep_dir: Directory where sweep results are stored
            top_k: Number of best models to keep globally
            metric_name: Metric to optimize
            mode: "min" or "max" for optimization direction
            cleanup_intermediate: Whether to clean up intermediate checkpoints
        """
        self.sweep_dir = Path(sweep_dir)
        self.top_k = top_k
        self.metric_name = metric_name
        self.mode = mode
        self.cleanup_intermediate = cleanup_intermediate
        
        # Create sweep directory structure
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.sweep_dir / "sweep_results.json"
        self.best_models_dir = self.sweep_dir / "best_models"
        self.best_models_dir.mkdir(exist_ok=True)
        self.best_configs_dir = self.sweep_dir / "best_configs"
        self.best_configs_dir.mkdir(exist_ok=True)
        self.lock_file = self.sweep_dir / ".sweep.lock"
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load existing results
        self.results = self._load_results()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the sweep manager."""
        logger = logging.getLogger(f"SweepManager_{self.sweep_dir.name}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.sweep_dir / "sweep_manager.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger (avoid duplicates)
        if not logger.handlers:
            logger.addHandler(handler)
            
        return logger
    
    def _load_results(self) -> List[ModelResult]:
        """Load existing sweep results from file."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                return [ModelResult(**item) for item in data]
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Could not load existing results: {e}")
        return []
    
    def _save_results(self):
        """Save current results to file."""
        try:
            with open(self.results_file, 'w') as f:
                data = [asdict(result) for result in self.results]
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save results: {e}")
    
    def _acquire_lock(self, timeout: float = 30.0) -> bool:
        """Acquire a file lock for thread-safe operations."""
        if fcntl is None:
            # Fallback for systems without fcntl (e.g., Windows)
            self.logger.warning("File locking not available on this platform")
            return True
            
        try:
            self.lock_fd = open(self.lock_file, 'w')
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    if time.time() - start_time > timeout:
                        return False
                    time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Could not acquire lock: {e}")
            return False
    
    def _release_lock(self):
        """Release the file lock."""
        if fcntl is None:
            return
            
        try:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
        except Exception as e:
            self.logger.error(f"Could not release lock: {e}")
    
    def register_model(
        self,
        run_id: str,
        run_name: str,
        metric_value: float,
        model_path: str,
        config_path: str,
        epoch: int,
        wandb_run_id: Optional[str] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Register a model result and update global top-k ranking.
        
        Args:
            run_id: Unique identifier for this run
            run_name: Human-readable run name
            metric_value: Value of the monitored metric
            model_path: Path to the saved model
            config_path: Path to the run configuration
            epoch: Epoch number when this model was saved
            wandb_run_id: W&B run ID if available
            additional_metrics: Additional metrics to store
            
        Returns:
            True if this model is in the top-k, False otherwise
        """
        if not self._acquire_lock():
            self.logger.error("Could not acquire lock for model registration")
            return False
        
        try:
            # Reload results to get latest state
            self.results = self._load_results()
            
            # Create new result
            result = ModelResult(
                run_id=run_id,
                run_name=run_name,
                metric_value=metric_value,
                metric_name=self.metric_name,
                model_path=model_path,
                config_path=config_path,
                epoch=epoch,
                timestamp=datetime.now().isoformat(),
                wandb_run_id=wandb_run_id,
                additional_metrics=additional_metrics or {}
            )
            
            # Add to results
            self.results.append(result)
            
            # Sort results based on metric
            reverse = (self.mode == "max")
            self.results.sort(key=lambda x: x.metric_value, reverse=reverse)
            
            # Check if this model made it to top-k
            is_top_k = result in self.results[:self.top_k]
            
            if is_top_k:
                # Copy model and config to best_models directory
                self._save_best_model(result)
                self.logger.info(f"Model {run_name} added to top-{self.top_k} "
                               f"({self.metric_name}: {metric_value:.6f})")
            else:
                # Model didn't make top-k, delete the run directory
                self.logger.info(f"Model {run_name} ({self.metric_name}: {metric_value:.6f}) "
                               f"did not make top-{self.top_k}")
                if self.cleanup_intermediate:
                    self._delete_run_directory(model_path)
            
            # Clean up models outside top-k
            if self.cleanup_intermediate:
                self._cleanup_old_models()
            
            # Save updated results
            self._save_results()
            
            # Generate summary report
            self._generate_summary()
            
            return is_top_k
            
        finally:
            self._release_lock()
    
    def _save_best_model(self, result: ModelResult):
        """Save a model and its config to the best_models directory."""
        try:
            # Create unique filename with ranking
            current_rank = self.results.index(result) + 1
            safe_name = "".join(c for c in result.run_name if c.isalnum() or c in "._-")
            
            # Copy model file
            model_src = Path(result.model_path)
            if model_src.exists():
                # Use a cleaner filename: rank_run_name_metric_value.pth
                model_ext = model_src.suffix  # .ckpt or .pth
                model_dst = self.best_models_dir / f"rank_{current_rank:02d}_{safe_name}_{self.metric_name}_{result.metric_value:.6f}{model_ext}"
                shutil.copy2(model_src, model_dst)
                self.logger.info(f"Copied model to best_models: {model_dst.name}")
                
                result.model_path = str(model_dst)  # Update path
            
            # Copy config file (contains model hyperparameters)
            config_src = Path(result.config_path)
            if config_src.exists():
                config_dst = self.best_configs_dir / f"rank_{current_rank:02d}_{safe_name}_config.yaml"
                shutil.copy2(config_src, config_dst)
                self.logger.info(f"Copied config to best_configs: {config_dst.name}")
                result.config_path = str(config_dst)  # Update path
            
            # Delete the run directory after successful copy
            if self.cleanup_intermediate and model_src.exists():
                self._delete_run_directory(str(model_src))
            
        except Exception as e:
            self.logger.error(f"Could not save best model {result.run_name}: {e}")
    
    def _cleanup_old_models(self):
        """Remove models that are no longer in top-k from best_models directory."""
        try:
            # Get the list of files currently in best_models directory
            if not self.best_models_dir.exists():
                return
            
            # Collect run_ids that should be kept (top-k)
            top_k_run_ids = {result.run_id for result in self.results[:self.top_k]}
            
            # Scan best_models directory and remove files not in top-k
            for file_path in self.best_models_dir.iterdir():
                if file_path.is_file():
                    # Check if this file belongs to a top-k model
                    should_keep = False
                    for result in self.results[:self.top_k]:
                        if (result.model_path and Path(result.model_path) == file_path) or \
                           (result.config_path and Path(result.config_path) == file_path):
                            should_keep = True
                            break
                    
                    # Remove file if it's not in top-k
                    if not should_keep:
                        try:
                            file_path.unlink()
                            self.logger.info(f"Cleaned up old file: {file_path.name}")
                        except Exception as e:
                            self.logger.warning(f"Could not remove {file_path}: {e}")
            
            # Keep only top-k results in memory
            self.results = self.results[:self.top_k]
            
            self.logger.info(f"Cleanup complete - keeping top {self.top_k} models")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _delete_checkpoint(self, checkpoint_path: str):
        """Delete a checkpoint file that didn't make top-k."""
        try:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists() and checkpoint_path.is_file():
                # Only delete if it's in the sweep directory (safety check)
                if str(self.sweep_dir) in str(checkpoint_path.resolve()):
                    checkpoint_path.unlink()
                    self.logger.info(f"Deleted non-top-k checkpoint: {checkpoint_path.name}")
                else:
                    self.logger.warning(
                        f"Checkpoint outside sweep directory, skipping deletion: {checkpoint_path}"
                    )
        except Exception as e:
            self.logger.error(f"Error deleting checkpoint {checkpoint_path}: {e}")
    
    def _delete_run_directory(self, model_path: str):
        """Delete the entire run directory for a model that didn't make top-k."""
        try:
            model_path = Path(model_path)
            run_dir = model_path.parent
            
            # Safety checks
            if not run_dir.exists():
                return
            
            if not run_dir.is_dir():
                return
            
            # Only delete if it's a subdirectory of sweep_dir
            if str(self.sweep_dir) not in str(run_dir.resolve()):
                self.logger.warning(
                    f"Run directory outside sweep directory, skipping deletion: {run_dir}"
                )
                return
            
            # Don't delete the sweep root itself
            if run_dir == self.sweep_dir:
                self.logger.warning("Attempted to delete sweep root directory, skipping")
                return
            
            # Check if this looks like a run-specific directory
            if run_dir.name.startswith('sweep_'):
                shutil.rmtree(run_dir)
                self.logger.info(f"Deleted run directory: {run_dir.name}")
            else:
                self.logger.warning(
                    f"Directory doesn't match run pattern, skipping deletion: {run_dir.name}"
                )
                
        except Exception as e:
            self.logger.error(f"Error deleting run directory: {e}")
    
    def _generate_summary(self):
        """Generate a summary report of the sweep."""
        try:
            summary = {
                "sweep_info": {
                    "sweep_dir": str(self.sweep_dir),
                    "top_k": self.top_k,
                    "metric_name": self.metric_name,
                    "mode": self.mode,
                    "total_runs": len(self.results),
                    "last_updated": datetime.now().isoformat()
                },
                "best_models": []
            }
            
            for i, result in enumerate(self.results[:self.top_k], 1):
                model_info = {
                    "rank": i,
                    "run_name": result.run_name,
                    "run_id": result.run_id,
                    "metric_value": result.metric_value,
                    "epoch": result.epoch,
                    "model_path": result.model_path,
                    "config_path": result.config_path,
                    "timestamp": result.timestamp,
                    "wandb_run_id": result.wandb_run_id,
                    "additional_metrics": result.additional_metrics
                }
                summary["best_models"].append(model_info)
            
            # Save summary as JSON (always available)
            summary_json_path = self.sweep_dir / "sweep_summary.json"
            with open(summary_json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save as YAML if available
            if yaml is not None:
                summary_path = self.sweep_dir / "sweep_summary.yaml"
                with open(summary_path, 'w') as f:
                    yaml.dump(summary, f, default_flow_style=False)
                self.logger.info(f"Generated sweep summary: {summary_path}")
            else:
                self.logger.info(f"Generated sweep summary: {summary_json_path}")
                
        except Exception as e:
            self.logger.error(f"Could not generate summary: {e}")
    
    def get_best_models(self, k: Optional[int] = None) -> List[ModelResult]:
        """
        Get the top-k best models.
        
        Args:
            k: Number of models to return (defaults to self.top_k)
            
        Returns:
            List of ModelResult objects
        """
        if k is None:
            k = self.top_k
        return self.results[:k]
    
    def get_best_model_path(self) -> Optional[str]:
        """Get the path to the best model."""
        if self.results:
            return self.results[0].model_path
        return None
    
    def print_summary(self):
        """Print a summary of the current top models."""
        print(f"\n{'='*60}")
        print(f"SWEEP SUMMARY - Top {self.top_k} Models")
        print(f"{'='*60}")
        print(f"Metric: {self.metric_name} (mode: {self.mode})")
        print(f"Total runs: {len(self.results)}")
        print()
        
        for i, result in enumerate(self.results[:self.top_k], 1):
            print(f"{i:2d}. {result.run_name}")
            print(f"    {self.metric_name}: {result.metric_value:.6f}")
            print(f"    Epoch: {result.epoch}")
            print(f"    Model: {result.model_path}")
            if result.wandb_run_id:
                print(f"    W&B: {result.wandb_run_id}")
            print()
    
    def cleanup_run_directories(self, keep_last_n: int = 10):
        """
        Clean up old run directories to save disk space.
        Keeps only the last N runs' temporary files.
        
        Args:
            keep_last_n: Number of recent runs to keep temp files for
        """
        try:
            # Get all run directories sorted by timestamp
            run_dirs = []
            for result in self.results:
                model_path = Path(result.model_path)
                # Look for the run's temporary directory (not in best_models)
                if 'best_models' not in str(model_path):
                    run_dir = model_path.parent
                    if run_dir.exists():
                        run_dirs.append((result.timestamp, run_dir, result.run_id))
            
            # Sort by timestamp (newest first)
            run_dirs.sort(reverse=True)
            
            # Remove old run directories (keep only last N)
            for i, (timestamp, run_dir, run_id) in enumerate(run_dirs):
                if i >= keep_last_n:
                    # Check if this run is in top-k (don't delete those)
                    is_top_k = any(r.run_id == run_id for r in self.results[:self.top_k])
                    if not is_top_k:
                        try:
                            if run_dir.exists() and run_dir != self.sweep_dir:
                                shutil.rmtree(run_dir)
                                self.logger.info(f"Cleaned up old run directory: {run_dir}")
                        except Exception as e:
                            self.logger.warning(f"Could not remove {run_dir}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error during run directory cleanup: {e}")


def create_sweep_manager(
    sweep_dir: Union[str, Path],
    top_k: int = 3,
    metric_name: str = "val_loss",
    mode: str = "min"
) -> SweepManager:
    """
    Factory function to create a sweep manager.
    
    Args:
        sweep_dir: Directory where sweep results are stored
        top_k: Number of best models to keep
        metric_name: Metric to optimize
        mode: "min" or "max"
        
    Returns:
        Configured SweepManager instance
    """
    return SweepManager(
        sweep_dir=sweep_dir,
        top_k=top_k,
        metric_name=metric_name,
        mode=mode
    )


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create a test sweep manager
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = create_sweep_manager(
            sweep_dir=temp_dir,
            top_k=3,
            metric_name="val_loss"
        )
        
        # Simulate some model results
        models = [
            ("run_1", "test_model_1", 0.5, "model1.pth", "config1.yaml", 10),
            ("run_2", "test_model_2", 0.3, "model2.pth", "config2.yaml", 15),
            ("run_3", "test_model_3", 0.7, "model3.pth", "config3.yaml", 8),
            ("run_4", "test_model_4", 0.2, "model4.pth", "config4.yaml", 20),
            ("run_5", "test_model_5", 0.4, "model5.pth", "config5.yaml", 12),
        ]
        
        for run_id, run_name, metric_val, model_path, config_path, epoch in models:
            # Create dummy files
            Path(temp_dir, model_path).touch()
            Path(temp_dir, config_path).touch()
            
            is_top_k = manager.register_model(
                run_id=run_id,
                run_name=run_name,
                metric_value=metric_val,
                model_path=str(Path(temp_dir, model_path)),
                config_path=str(Path(temp_dir, config_path)),
                epoch=epoch
            )
            print(f"Model {run_name} (loss: {metric_val:.3f}) "
                  f"{'✓' if is_top_k else '✗'} top-{manager.top_k}")
        
        # Print final summary
        manager.print_summary()
