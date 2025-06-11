# Hypertuning Machine Learning Models

This repository contains implementations of various machine learning models (NN, CNN, DNN, RNN) with optimal hyperparameters determined through experimentation. The project is structured to provide a clean interface for training and evaluating these models.

## Project Structure

```
.
├── dev/                   # Development utilities
├── img/                   # Images and visualizations
├── networks/              # Model implementations
│   ├── CNN.py             # Convolutional Neural Network
│   ├── NN.py              # Dense Neural Networks
│   └── RNN.py             # Recurrent Neural Network implementations
├── notebooks/             # Jupyter notebooks for experimentation
├── main.py                # Main script for training models
├── config.toml            # Configuration file for customizing parameters
├── run_all_networks.sh    # Script to run all network types for testing
└── README.md              # This file
```

## Configuration

The project uses a `config.toml` file to customize all parameters for models, datasets, and training settings. This allows for easy experimentation without modifying the code.

### Configuration Structure

The configuration file is organized into sections:

```toml
[general]
# General settings like save_model

[training]
# Training parameters like epochs, batch_size, etc.

[early_stopping]
# Early stopping parameters

[logging]
# Logging settings

[models.cnn]
# CNN model parameters

[models.nn]
# Neural Network parameters

[models.dnn]
# Deep Neural Network parameters

[models.rnn]
# RNN model parameters

[datasets.fashion]
# Fashion MNIST dataset parameters

[datasets.gestures]
# Gestures dataset parameters
```

You can modify any parameter in the config file to customize the behavior of the models and training process. Command-line arguments will override the values in the config file.

## Models and Optimal Hyperparameters

### CNN (Convolutional Neural Network)

The CNN model is designed for image classification tasks, particularly for the Fashion MNIST dataset.

**Optimal Hyperparameters:**
- Filters: 32
- Units: 128
- Num classes: 10
- Optimizer: Adam
- Epochs: 10
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=5

### DNN (Dense Neural Network)

Two variants of DNN are implemented:

1. **NeuralNetwork** (2 hidden layers)
   - Units1: 300
   - Units2: 100
   - Num classes: 10
   - Optimizer: Adam
   - Epochs: 10
   - Scheduler: ReduceLROnPlateau

2. **DeepNeuralNetwork** (3 hidden layers)
   - Units1: 512
   - Units2: 256
   - Units3: 128
   - Num classes: 10
   - Optimizer: Adam
   - Epochs: 10
   - Scheduler: ReduceLROnPlateau

### RNN (Recurrent Neural Network)

Four RNN model variants are implemented for sequence classification tasks, particularly for the Gestures dataset:

1. **RecurrentNeuralNetworkWithGRU** (GRU-based)
   - Uses Gated Recurrent Unit cells
   - Layer normalization after RNN layer

2. **RecurrentNeuralNetwork** (Basic RNN)
   - Uses basic RNN cells with tanh nonlinearity
   - Layer normalization after RNN layer

3. **GRUWithAttention** (GRU with Attention)
   - Uses Gated Recurrent Unit cells
   - Includes an attention mechanism to focus on important time steps
   - Layer normalization after RNN layer

4. **RecurrentNeuralNetworkWithAttention** (RNN with Attention)
   - Uses basic RNN cells with tanh nonlinearity
   - Includes an attention mechanism to focus on important time steps
   - Layer normalization after RNN layer

**Optimal Hyperparameters for all RNN models:**
- Input size: 3
- Hidden size: 128
- Number of layers: 2
- Dropout: 0.4
- Output size: 20
- Optimizer: Adam
- Epochs: 10
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=5

## Usage

The `main.py` script provides a unified interface for training the different model types with their optimal hyperparameters. All parameters can be customized either through the `config.toml` file or via command-line arguments.

Run `bash run_all_networks.sh` to verify all networks with minimal training for testing purposes.

### Command-line Arguments

- `--config`: Path to configuration file (default: config.toml)
- `--model`: Type of model to train (cnn, dnn, nn, rnn_gru, rnn_basic, gru_attention, or rnn_attention)
- `--epochs`: Number of training epochs (default: from config.toml)
- `--batch-size`: Batch size for data loading (default: from config.toml)
- `--no-save`: Do not save the trained model

### Examples

Train a CNN model with default hyperparameters:
```bash
python main.py --model cnn
```

Train a DNN model with 30 epochs:
```bash
python main.py --model dnn --epochs 30
```

Train a Deep Neural Network model with a batch size of 64:
```bash
python main.py --model nn --batch-size 64
```

Train a GRU model without saving:
```bash
python main.py --model rnn_gru --no-save
```

Train a basic RNN model:
```bash
python main.py --model rnn_basic
```

Train a GRU model with attention:
```bash
python main.py --model gru_attention
```

Train an RNN model with attention:
```bash
python main.py --model rnn_attention
```

## Development

To extend this project with new models or datasets:

1. Add new model implementations in the `networks/` directory
2. Update the `main.py` script to include functions for creating and training the new models
3. Determine optimal hyperparameters through experimentation in Jupyter notebooks

## Requirements

- Python 3.6+
- PyTorch
- Additional dependencies as specified in the project

