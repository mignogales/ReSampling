#!/usr/bin/env python3
"""
Test script for the new sweep management system.
This demonstrates the key features without needing full W&B integration.
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

from extras.sweep_manager import create_sweep_manager, ModelResult, SweepManager
from extras.callbacks import BestModelTracker


def test_sweep_manager():
    """Test the SweepManager functionality."""
    print("🧪 Testing SweepManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sweep manager
        manager = create_sweep_manager(
            sweep_dir=temp_dir,
            top_k=3,
            metric_name="val_loss"
        )
        
        print(f"✅ Created SweepManager in {temp_dir}")
        
        # Simulate some model results
        test_models = [
            ("run_1", "s5_model_1", 0.5),
            ("run_2", "s5_model_2", 0.3),  
            ("run_3", "s5_model_3", 0.7),
            ("run_4", "s5_model_4", 0.2),  # Best
            ("run_5", "s5_model_5", 0.4),
            ("run_6", "s5_model_6", 0.8),
        ]
        
        # Create dummy files and register models
        for run_id, run_name, metric_val in test_models:
            # Create dummy files
            model_path = Path(temp_dir) / f"{run_name}.pth"
            config_path = Path(temp_dir) / f"{run_name}_config.yaml"
            model_path.write_text(f"# Dummy model for {run_name}")
            config_path.write_text(f"config: {run_name}")
            
            is_top_k = manager.register_model(
                run_id=run_id,
                run_name=run_name,
                metric_value=metric_val,
                model_path=str(model_path),
                config_path=str(config_path),
                epoch=10,
                additional_metrics={"train_loss": metric_val + 0.1}
            )
            
            print(f"{'✅' if is_top_k else '❌'} {run_name} (loss: {metric_val:.1f}) -> {'Top-3' if is_top_k else 'Not top-3'}")
        
        # Check final results
        best_models = manager.get_best_models()
        print(f"\n🏆 Final Top-3 Models:")
        for i, model in enumerate(best_models, 1):
            print(f"  {i}. {model.run_name} ({model.metric_value:.1f})")
        
        # Verify we have exactly 3 models
        assert len(best_models) == 3, f"Expected 3 models, got {len(best_models)}"
        
        # Verify they're in the right order (best first)
        expected_order = [0.2, 0.3, 0.4]
        actual_order = [m.metric_value for m in best_models]
        assert actual_order == expected_order, f"Wrong order: {actual_order} != {expected_order}"
        
        print("✅ SweepManager test passed!")
        
        # Show summary
        manager.print_summary()


def test_model_result():
    """Test ModelResult dataclass."""
    print("\n🧪 Testing ModelResult...")
    
    result = ModelResult(
        run_id="test_run",
        run_name="test_model",
        metric_value=0.123,
        metric_name="val_loss",
        model_path="/path/to/model.pth",
        config_path="/path/to/config.yaml",
        epoch=25,
        timestamp="2024-01-01T12:00:00",
        wandb_run_id="wandb_123",
        additional_metrics={"accuracy": 0.95}
    )
    
    # Test serialization
    data = result.__dict__
    assert data["run_id"] == "test_run"
    assert data["additional_metrics"]["accuracy"] == 0.95
    
    print("✅ ModelResult test passed!")


def main():
    """Run all tests."""
    print("🚀 Running Enhanced Sweep Management System Tests")
    print("=" * 60)
    
    try:
        test_model_result()
        test_sweep_manager()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Setup your W&B keys in config/wandb/keys.yaml")
        print("  2. Launch a sweep with improved tracking")
        print("  3. Monitor progress: python experiments/sweep_utils.py analyze")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
