# Enhanced Sweep Management System

This directory contains an enhanced sweep management system that automatically tracks and saves only the top-3 best models across entire hyperparameter sweeps, helping you maintain a clean codebase while ensuring reliable experiment tracking.

## Key Features

✅ **Global Top-K Tracking**: Only saves the best 3 models across all runs in a sweep  
✅ **Automatic Cleanup**: Removes intermediate checkpoints that don't make the top-K  
✅ **Thread-Safe Operations**: Safe for concurrent sweep runs  
✅ **Rich Monitoring**: Real-time progress tracking and summaries  
✅ **Integration with W&B**: Seamless Weights & Biases integration  
✅ **Export & Analysis**: Comprehensive utilities for result analysis  

## Quick Start

### 1. Launch a Sweep

```bash
# Launch an enhanced S5 sweep
./experiments/launch_sweep.py s5 --dataset Solar --enhanced --count 20

# Launch a basic sweep for DMixer
./experiments/launch_sweep.py dmixer --dataset Wind --count 15

# Dry run to see configuration
./experiments/launch_sweep.py s5 --dry-run
```

### 2. Monitor Progress

The sweep launcher automatically monitors progress and shows real-time updates:

```
🔍 Monitoring sweep sweep_abc123
📊 Sweep Update (Run 5)
   Best val_loss: 0.234567
   Best model: sweep_s5_20260128_143022_run_5
   Total runs: 5
   Top models:
     1. sweep_s5_20260128_143022_run_5 (0.234567)
     2. sweep_s5_20260128_143022_run_3 (0.245123)
     3. sweep_s5_20260128_143022_run_1 (0.256789)
```

### 3. Analyze Results

```bash
# Analyze the most recent sweep
./experiments/sweep_utils.py analyze

# Compare all sweeps
./experiments/sweep_utils.py compare

# List all sweep directories
./experiments/sweep_utils.py list

# Export results to CSV
./experiments/sweep_utils.py export --output my_results.csv
```

### 4. Clean Up Old Sweeps

```bash
# Dry run cleanup (shows what would be deleted)
./experiments/sweep_utils.py cleanup --days 7

# Actually delete old sweeps
./experiments/sweep_utils.py cleanup --days 7 --live
```

## Directory Structure

After running sweeps, your directory structure will look like:

```
logs/
├── sweeps/
│   ├── s5/
│   │   └── Solar/
│   │       └── 2026-01-28_14-30-22/          # Sweep directory
│   │           ├── best_models/               # Top-3 models only
│   │           │   ├── rank_01_model_run_5_best_model.pth
│   │           │   ├── rank_01_model_run_5_config.yaml
│   │           │   ├── rank_02_model_run_3_best_model.pth
│   │           │   └── rank_03_model_run_1_best_model.pth
│   │           ├── sweep_results.json         # All run metadata
│   │           ├── sweep_summary.yaml         # Human-readable summary
│   │           └── sweep_manager.log          # Detailed logs
│   └── dmixer/
│       └── Wind/
└── experiments/                               # Individual run logs
    └── ...
```

## Configuration Files

### Enhanced S5 Sweep Configuration

The system includes an enhanced sweep configuration for S5 models (`config/wandb/sweep_s5_enhanced.yaml`) with:

- **Bayesian optimization** for efficient hyperparameter search
- **Early termination** with Hyperband for faster convergence
- **S5-specific parameters**: state_size, num_blocks, dt_min/dt_max, scan_method
- **Optimized ranges** based on S5 architecture requirements

### Custom Sweep Configurations

You can create custom sweep configurations by adding new YAML files to `config/wandb/`. The system automatically detects and uses appropriate configurations based on model names.

## SweepManager API

For programmatic access, you can use the SweepManager directly:

```python
from extras.sweep_manager import SweepManager

# Create a sweep manager
manager = SweepManager(
    sweep_dir="logs/sweeps/my_experiment",
    top_k=3,
    metric_name="val_loss",
    mode="min"
)

# Register a model result
is_top_k = manager.register_model(
    run_id="run_123",
    run_name="my_model_run",
    metric_value=0.234,
    model_path="path/to/model.pth",
    config_path="path/to/config.yaml",
    epoch=25
)

# Get best models
best_models = manager.get_best_models(k=3)
```

## Integration with Existing Code

The new system is backward-compatible with your existing sweep infrastructure:

1. **BestModelTracker**: Enhanced to work with SweepManager while maintaining local tracking
2. **sweep.py**: Updated to use SweepManager for global coordination
3. **W&B Integration**: Seamless logging and model artifact management

## Benefits

### Before (Old System)
- ❌ All models saved regardless of quality
- ❌ Manual cleanup required
- ❌ Difficult to compare across sweeps
- ❌ Large disk usage
- ❌ Hard to find best models

### After (Enhanced System)
- ✅ Only top-3 models preserved automatically
- ✅ Automatic cleanup of intermediate files
- ✅ Easy cross-sweep comparison
- ✅ Minimal disk usage
- ✅ Best models clearly identified and accessible

## Advanced Features

### Thread-Safe Operations
The system uses file locking to ensure safe operation during concurrent sweep runs, preventing race conditions and data corruption.

### Rich Metadata Tracking
Each model result includes:
- Performance metrics (primary + additional)
- Training metadata (epoch, timestamp)
- Configuration snapshots
- W&B run IDs for cross-referencing

### Extensible Architecture
The system is designed to be easily extended for:
- Different optimization metrics
- Custom cleanup policies
- Additional metadata tracking
- Integration with other experiment tracking systems

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure scripts are executable (`chmod +x`)
2. **Missing dependencies**: Install required packages (`pip install pandas fcntl`)
3. **W&B authentication**: Check `config/wandb/keys.yaml` exists and is valid
4. **Disk space**: Use cleanup utilities to manage storage

### Debug Mode

Enable verbose logging by setting the environment variable:
```bash
export SWEEP_DEBUG=1
./experiments/launch_sweep.py s5 --dataset Solar
```

## Contributing

When adding new features:
1. Update the SweepManager class for core functionality
2. Add corresponding utilities to sweep_utils.py
3. Update sweep configurations as needed
4. Add tests and documentation

---

*This enhanced sweep system helps maintain a clean, efficient, and reproducible hyperparameter optimization workflow.*
