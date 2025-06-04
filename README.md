# Machine Learning Models with Optimal Hyperparameters

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
└── README.md              # This file
```

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

The `main.py` script provides a unified interface for training the different model types with their optimal hyperparameters.

### Command-line Arguments

- `--model`: Type of model to train (cnn, dnn, nn, rnn_gru, rnn_basic, gru_attention, or rnn_attention)
- `--epochs`: Number of training epochs (default: model-specific)
- `--batch-size`: Batch size for data loading (default: 32)
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
