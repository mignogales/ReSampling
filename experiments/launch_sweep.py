#!/usr/bin/env python3
"""
Enhanced sweep launcher with automatic cleanup and monitoring.

This script:
- Launches sweeps with the new SweepManager integration
- Monitors sweep progress
- Provides real-time updates on best models
- Automatically cleans up intermediate files
"""

import sys
import yaml
import wandb
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import time
import signal
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from extras.sweep_manager import SweepManager


def load_sweep_config(model_name: str, enhanced: bool = False) -> dict:
    """Load sweep configuration based on model name."""
    
    # Determine config file
    if enhanced and model_name == 's5':
        config_path = "config/wandb/sweep_s5_enhanced.yaml"
    elif model_name in ['dmixer']:
        config_path = "config/wandb/sweep_dmixer.yaml"
    elif model_name in ['timemixer']:
        config_path = "config/wandb/sweep_timemixer.yaml"
    elif model_name in ['s5']:
        config_path = "config/wandb/sweep_s5.yaml"
    else:
        config_path = "config/wandb/sweep.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['sweep']
    except FileNotFoundError:
        print(f"❌ Sweep config not found: {config_path}")
        print("Available configs:")
        config_dir = Path("config/wandb")
        for cfg_file in config_dir.glob("sweep*.yaml"):
            print(f"  - {cfg_file.name}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading sweep config: {e}")
        sys.exit(1)


def load_wandb_keys() -> dict:
    """Load W&B API keys and project info."""
    try:
        with open("config/wandb/keys.yaml", 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ W&B keys file not found: config/wandb/keys.yaml")
        print("Please create this file with your W&B credentials")
        sys.exit(1)


def setup_sweep_directory(model_name: str, dataset_name: str = "default") -> Path:
    """Setup sweep directory structure."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = Path(f"logs/sweeps/{model_name}/{dataset_name}/{timestamp}")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a marker file
    marker = sweep_dir / ".sweep_info"
    info = {
        'model': model_name,
        'dataset': dataset_name,
        'started': timestamp,
        'status': 'running'
    }
    with open(marker, 'w') as f:
        yaml.dump(info, f)
    
    return sweep_dir


def monitor_sweep(sweep_id: str, sweep_manager: SweepManager, max_runs: int = None):
    """Monitor sweep progress and provide updates."""
    print(f"\n🔍 Monitoring sweep {sweep_id}")
    print("Press Ctrl+C to stop monitoring (sweep will continue)")
    
    last_run_count = 0
    
    try:
        while True:
            # Get current best models
            best_models = sweep_manager.get_best_models()
            current_run_count = len(sweep_manager.results)
            
            if current_run_count != last_run_count:
                print(f"\n📊 Sweep Update (Run {current_run_count})")
                if best_models:
                    best = best_models[0]
                    print(f"   Best {best.metric_name}: {best.metric_value:.6f}")
                    print(f"   Best model: {best.run_name}")
                    print(f"   Total runs: {current_run_count}")
                
                # Show top-3 if available
                if len(best_models) > 1:
                    print("   Top models:")
                    for i, model in enumerate(best_models[:3], 1):
                        print(f"     {i}. {model.run_name} ({model.metric_value:.6f})")
                
                last_run_count = current_run_count
            
            # Check if sweep is complete
            if max_runs and current_run_count >= max_runs:
                print(f"\n✅ Sweep completed! {current_run_count}/{max_runs} runs finished")
                break
                
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print(f"\n👋 Stopped monitoring (sweep continues running)")
        print(f"   Sweep ID: {sweep_id}")
        print(f"   Check progress at: https://wandb.ai/")


def launch_sweep(
    model_name: str,
    dataset_name: str = "Solar",
    enhanced: bool = False,
    count: int = None,
    monitor: bool = True,
    dry_run: bool = False
):
    """Launch a hyperparameter sweep."""
    
    print(f"🚀 Launching {'enhanced ' if enhanced else ''}sweep for {model_name}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Enhanced: {enhanced}")
    
    # Load configurations
    sweep_config = load_sweep_config(model_name, enhanced)
    wandb_keys = load_wandb_keys()
    
    # Override count if specified
    if count:
        sweep_config['count'] = count
        print(f"   Run count: {count}")
    
    # Setup directories
    sweep_dir = setup_sweep_directory(model_name, dataset_name)
    print(f"   Sweep directory: {sweep_dir}")
    
    # Initialize sweep manager
    sweep_manager = SweepManager(
        sweep_dir=sweep_dir,
        top_k=3,
        metric_name=sweep_config['metric']['name'],
        mode='min' if sweep_config['metric']['goal'] == 'minimize' else 'max'
    )
    
    if dry_run:
        print("\n🔍 DRY RUN - Would launch sweep with config:")
        print(yaml.dump(sweep_config, default_flow_style=False))
        return
    
    # Login to W&B
    wandb.login(key=wandb_keys['key'])
    
    # Add timestamp to sweep name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_config['name'] = f"sweep_{model_name}_{dataset_name}_{timestamp}"
    
    print(f"   Sweep name: {sweep_config['name']}")
    
    # Initialize the sweep
    try:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=wandb_keys['project'],
            entity=wandb_keys['entity'],
        )
        print(f"✅ Sweep created: {sweep_id}")
        print(f"   View at: https://wandb.ai/{wandb_keys['entity']}/{wandb_keys['project']}/sweeps/{sweep_id}")
        
        # Save sweep info
        sweep_info = {
            'sweep_id': sweep_id,
            'sweep_config': sweep_config,
            'wandb_keys': {k: v for k, v in wandb_keys.items() if k != 'key'},  # Don't save the key
            'sweep_dir': str(sweep_dir),
            'model_name': model_name,
            'dataset_name': dataset_name
        }
        
        with open(sweep_dir / "sweep_info.yaml", 'w') as f:
            yaml.dump(sweep_info, f, default_flow_style=False)
        
        # Start the sweep agent
        print(f"\n🤖 Starting sweep agent...")
        
        if monitor:
            print(f"\n🤖 Starting sweep agent...")
        print(f"   Use this command to start agents:")
        print(f"   wandb agent {wandb_keys['entity']}/{wandb_keys['project']}/{sweep_id}")
        print(f"   Or run: python experiments/sweep.py")
        
        # Don't start agent automatically - let user control it
        input("\nPress Enter when ready to start monitoring...")
        
        if monitor:
            monitor_sweep(sweep_id, sweep_manager, sweep_config.get('count'))
        
        print(f"\n🎉 Sweep completed!")
        
        # Final summary
        sweep_manager.print_summary()
        
    except Exception as e:
        print(f"❌ Error launching sweep: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Enhanced sweep launcher")
    
    parser.add_argument('model', help='Model name (s5, dmixer, timemixer, etc.)')
    parser.add_argument('--dataset', default='Solar', help='Dataset name')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced sweep configuration')
    parser.add_argument('--count', type=int, help='Number of runs (overrides config)')
    parser.add_argument('--no-monitor', action='store_true', help='Disable progress monitoring')
    parser.add_argument('--dry-run', action='store_true', help='Show config without launching')
    
    args = parser.parse_args()
    
    launch_sweep(
        model_name=args.model,
        dataset_name=args.dataset,
        enhanced=args.enhanced,
        count=args.count,
        monitor=not args.no_monitor,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
