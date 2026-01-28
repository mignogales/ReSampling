# ReSampling: Time Series Forecasting with Arbitrary Temporal Resolution

A comprehensive research framework for evaluating time series forecasting models under different temporal resolutions and resampling strategies. This project explores zero-shot robustness of forecasting models when applied to data with degraded temporal resolution.

## 🎯 Overview

This project investigates how time series forecasting models perform when evaluated on data with different temporal resolutions than their training data. Using sophisticated resampling techniques, we can simulate real-world scenarios where data quality degrades or collection frequencies change.

**Data Source**: [Forecasting Data Repository](https://forecastingdata.org/)

## ✨ Key Features

### 🔄 Advanced Resampling System
- **Arbitrary resampling rates** via `scipy.signal.resample`
- **Zero-shot evaluation** of models on degraded temporal resolution
- **Multiple resampling methods**: Fourier-based and interpolation
- **Flexible temporal resolution control**: 1.0 (original), 2.0 (half samples), 0.5 (double samples)

### 🧠 Multiple Model Architectures
- **GRU**: Gated Recurrent Units for sequential modeling
- **DMixer**: Deep mixing architecture for time series
- **TimeMixer**: Advanced temporal mixing model
- **S5**: State Space Sequence model
- **MLP**: Simple temporal Multi-Layer Perceptron

### 🔬 Enhanced Experiment Management
- **Top-K Model Tracking**: Automatically saves only the best 3 models across sweeps
- **Hyperparameter Sweeps**: Comprehensive grid search with W&B integration
- **Real-time Model Ranking**: Thread-safe operations for concurrent runs
- **Automatic Cleanup**: Removes intermediate checkpoints to save disk space

### 📊 Comprehensive Analysis Tools
- **Rich Metrics Logging**: Detailed performance tracking
- **Cross-sweep Comparison**: Compare results across different experiments  
- **CSV Export**: Export results for further analysis
- **Command-line Utilities**: Easy-to-use analysis tools

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ReSampling
```

2. **Create the conda environment**:
```bash
conda env create -f env.yml
conda activate ReSampling
```

### Basic Usage

1. **Configure your experiment** in `config/default.yaml`:
```yaml
dataset: your_dataset_name
model: gru  # or dmixer, timemixer, s5, mlp
```

2. **Run a single experiment**:
```bash
cd experiments
python main.py
```

3. **Launch a hyperparameter sweep**:
```bash
python launch_sweep.py
```

4. **Test resampling effects**:
```bash
python test_sampling_rates.py
```

## 📁 Project Structure

```
ReSampling/
├── config/                    # Configuration files
│   ├── default.yaml          # Main configuration
│   ├── dataset/              # Dataset configurations
│   ├── model/                # Model configurations
│   └── optimizer/            # Optimizer settings
├── experiments/              # Main experiment scripts
│   ├── main.py              # Single experiment runner
│   ├── launch_sweep.py      # Sweep launcher
│   └── sweep_utils.py       # Sweep utilities
├── models/                  # Model implementations
│   ├── gru.py              # GRU model
│   ├── dmixer.py           # DMixer model
│   ├── timemixer.py        # TimeMixer model
│   ├── s5_model.py         # S5 model
│   └── mlp.py              # MLP model
├── extras/                 # Additional utilities
│   ├── timeseriesdatamodule_resampled.py  # Resampling data module
│   ├── sweep_manager.py    # Enhanced sweep management
│   ├── callbacks.py        # Training callbacks
│   └── metrics_logging.py  # Metrics tracking
├── datasets/               # Dataset storage
├── logs/                  # Experiment logs
│   ├── experiments/       # Single experiment logs
│   ├── sweeps/           # Sweep logs
│   └── frequency_testing/ # Resampling test logs
└── wandb/                # Weights & Biases logs
```

## 🔬 Research Focus

### Temporal Resolution Robustness
- Evaluate model performance under different sampling frequencies
- Understand degradation patterns when temporal resolution decreases
- Identify models most robust to temporal resolution changes

### Zero-Shot Transfer
- Train models on high-resolution data
- Test on lower-resolution versions without retraining
- Measure performance degradation across resolution scales

### Resampling Strategies
- Compare Fourier-based vs. interpolation resampling
- Analyze the impact of different resampling rates
- Optimize resampling parameters for minimal performance loss

## 🔧 Configuration

The project uses Hydra for configuration management. Key configuration areas:

- **Dataset**: Configure data source, preprocessing, and splits
- **Model**: Select and configure model architecture
- **Training**: Set training parameters, optimizers, and schedules
- **Resampling**: Control temporal resolution and resampling methods
- **Logging**: Configure experiment tracking and output directories

## 📈 Experiment Types

1. **Single Experiments**: Test individual model-dataset combinations
2. **Hyperparameter Sweeps**: Comprehensive parameter exploration
3. **Frequency Testing**: Systematic resampling rate evaluation
4. **Cross-model Comparison**: Compare architectures under resampling

## 🎯 Enhanced Sweep Management

The project includes a sophisticated sweep management system:

- **Automatic Top-K Selection**: Only saves the best performing models
- **Resource Optimization**: Minimal disk usage through intelligent cleanup
- **Concurrent Safety**: Thread-safe operations for parallel sweep execution
- **Rich Analytics**: Built-in tools for sweep result analysis

### Running Enhanced Sweeps

```bash
# Launch a sweep with automatic top-3 model tracking
python launch_sweep.py

# Analyze sweep results
python -m extras.sweep_manager analyze --sweep-dir logs/sweeps/your_sweep

# Export results to CSV
python -m extras.sweep_manager export --sweep-dir logs/sweeps/your_sweep --output results.csv
```

## 📊 Metrics and Analysis

The framework tracks comprehensive metrics:

- **Forecast Accuracy**: MAE, MSE, RMSE, MAPE
- **Temporal Robustness**: Performance across resampling rates
- **Model Efficiency**: Training time, memory usage, convergence
- **Zero-shot Transfer**: Original vs. resampled performance ratios

## 🤝 Contributing

This is a research project focused on time series forecasting robustness. Contributions related to:
- New model architectures
- Novel resampling strategies  
- Enhanced evaluation metrics
- Dataset integrations

are welcome!

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- Data source: [Forecasting Data Repository](https://forecastingdata.org/)
- Built with PyTorch Lightning, Hydra, and Weights & Biases
- Uses TSL (Time Series Library) for data handling