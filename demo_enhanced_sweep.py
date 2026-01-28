#!/usr/bin/env python3
"""
Quick demonstration of the enhanced sweep management system.
Run this to see how the system works without a full sweep.
"""

import tempfile
import json
from pathlib import Path

# Import our enhanced system
import sys
sys.path.append('.')
from extras.sweep_manager import create_sweep_manager
from extras.callbacks import BestModelTracker


def demo_sweep_system():
    """Demonstrate the enhanced sweep system capabilities."""
    print("🎯 ENHANCED SWEEP SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create a temporary demo directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Demo directory: {temp_dir}")
        
        # 1. Create a sweep manager (this is what happens automatically during sweeps)
        print("\n1️⃣ Creating SweepManager...")
        sweep_manager = create_sweep_manager(
            sweep_dir=temp_dir,
            top_k=3,  # Keep only top 3 models
            metric_name="val_loss",
            mode="min"
        )
        print(f"✅ SweepManager created (top-{sweep_manager.top_k} tracking)")
        
        # 2. Simulate multiple training runs with different performance
        print("\n2️⃣ Simulating sweep runs...")
        simulated_runs = [
            # (run_name, final_loss, config_summary)
            ("s5_run_001", 0.45, "hidden_size=32, lr=0.001"),
            ("s5_run_002", 0.32, "hidden_size=64, lr=0.0001"), # Good
            ("s5_run_003", 0.67, "hidden_size=16, lr=0.01"),
            ("s5_run_004", 0.28, "hidden_size=128, lr=0.0001"), # Better  
            ("s5_run_005", 0.52, "hidden_size=32, lr=0.01"),
            ("s5_run_006", 0.23, "hidden_size=256, lr=0.0001"), # Best
            ("s5_run_007", 0.38, "hidden_size=64, lr=0.001"),
        ]
        
        for i, (run_name, loss, config_desc) in enumerate(simulated_runs, 1):
            print(f"\n   🏃 Run {i}: {run_name}")
            print(f"       Config: {config_desc}")
            print(f"       Final val_loss: {loss:.3f}")
            
            # Create dummy model and config files
            model_path = Path(temp_dir) / f"{run_name}_model.pth"
            config_path = Path(temp_dir) / f"{run_name}_config.yaml"
            model_path.write_text(f"# Model for {run_name}\nval_loss: {loss}")
            config_path.write_text(f"# Config for {run_name}\n{config_desc}")
            
            # Register model with sweep manager (this happens automatically via BestModelTracker)
            is_top_k = sweep_manager.register_model(
                run_id=f"run_{i:03d}",
                run_name=run_name,
                metric_value=loss,
                model_path=str(model_path),
                config_path=str(config_path),
                epoch=30 + i,  # Simulate different training lengths
                additional_metrics={
                    'train_loss': loss + 0.05,
                    'val_accuracy': max(0.6, 1.0 - loss)
                }
            )
            
            status = "✅ TOP-3" if is_top_k else "❌ Pruned"
            print(f"       Result: {status}")
        
        # 3. Show final results
        print("\n3️⃣ Final Results Summary:")
        print("-" * 40)
        sweep_manager.print_summary()
        
        # 4. Demonstrate file organization
        print("\n4️⃣ File Organization:")
        print("-" * 40)
        print("📁 Created files:")
        for file in sorted(sweep_manager.sweep_dir.rglob("*")):
            if file.is_file():
                relative_path = file.relative_to(sweep_manager.sweep_dir)
                size = file.stat().st_size
                print(f"   {relative_path} ({size} bytes)")
        
        # 5. Show JSON summary
        print("\n5️⃣ JSON Summary Preview:")
        print("-" * 40)
        summary_file = sweep_manager.sweep_dir / "sweep_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            print(f"📊 Sweep Info:")
            print(f"   Total runs: {summary['sweep_info']['total_runs']}")
            print(f"   Metric: {summary['sweep_info']['metric_name']}")
            print(f"   Best models:")
            for model in summary['best_models']:
                print(f"     {model['rank']}. {model['run_name']} ({model['metric_value']:.3f})")
        
        print(f"\n✨ DEMONSTRATION COMPLETE!")
        print(f"   - Simulated {len(simulated_runs)} training runs")
        print(f"   - Automatically kept only top {sweep_manager.top_k} models")
        print(f"   - Generated organized file structure")
        print(f"   - Created rich metadata and summaries")


def show_current_setup():
    """Show the current state of your sweep system."""
    print("\n" + "="*60)
    print("📋 YOUR CURRENT SWEEP SYSTEM STATUS")
    print("="*60)
    
    # Check existing sweeps
    logs_dir = Path("logs")
    if logs_dir.exists():
        sweep_dirs = list((logs_dir / "sweeps").rglob("sweep_results.json"))
        print(f"🔍 Found {len(sweep_dirs)} existing enhanced sweeps")
        
        # Check old W&B sweeps
        wandb_dir = Path("wandb")
        if wandb_dir.exists():
            wandb_sweeps = [d for d in wandb_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("sweep-")]
            print(f"📁 Found {len(wandb_sweeps)} legacy W&B sweep directories")
            if len(wandb_sweeps) > 5:
                print("   💡 Consider using sweep_utils.py cleanup to manage old sweeps")
    
    # Show available tools
    print(f"\n🛠️  Available Tools:")
    tools = [
        ("experiments/sweep.py", "Launch sweeps (automatically enhanced)"),
        ("experiments/sweep_utils.py", "Analyze and manage sweeps"),
        ("experiments/test_sweep_system.py", "Test the sweep system"),
        ("ENHANCED_SWEEP_GUIDE.md", "Complete usage guide"),
    ]
    
    for tool, description in tools:
        exists = "✅" if Path(tool).exists() else "❌"
        print(f"   {exists} {tool:<35} - {description}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Test the system: python experiments/test_sweep_system.py")
    print(f"   2. Run a sweep: python experiments/sweep.py s5")
    print(f"   3. Analyze results: python experiments/sweep_utils.py analyze")
    

if __name__ == "__main__":
    demo_sweep_system()
    show_current_setup()
