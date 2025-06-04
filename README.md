# Machine Learning Models with Optimal Hyperparameters

This repository contains implementations of various machine learning models (CNN, DNN, RNN) with optimal hyperparameters determined through experimentation. The project is structured to provide a clean interface for training and evaluating these models.

## Project Structure

```
.
├── data/                  # Data directory
│   ├── external/          # External data sources
│   ├── processed/         # Processed data ready for modeling
│   └── raw/               # Raw data
├── dev/                   # Development utilities
├── img/                   # Images and visualizations
├── models/                # Model implementations
│   ├── CNN.py             # Convolutional Neural Network
│   ├── DNN.py             # Dense Neural Networks
│   └── RNN.py             # Recurrent Neural Network (GRU)
├── notebooks/             # Jupyter notebooks for experimentation
├── presentations/         # Presentation materials
├── references/            # Reference materials
├── reports/               # Generated reports
│   └── figures/           # Figures for reports
├── main.py                # Main script for training models
└── README.md              # This file
```

## Models and Optimal Hyperparameters

### CNN (Convolutional Neural Network)

The CNN model is designed for image classification tasks, particularly for the Fashion MNIST dataset.

**Optimal Hyperparameters:**
- Filters: 16
- Units: 128
- Optimizer: Adam with learning rate 0.01
- Epochs: 100
- Scheduler: ReduceLROnPlateau with factor=0.5, patience=10
- Early stopping: patience=100

### DNN (Dense Neural Network)

Two variants of DNN are implemented:

1. **DenseNeuralNetwork** (2 hidden layers)
   - Units1: 1024
   - Units2: 512
   - Optimizer: Adam
   - Epochs: 20
   - Scheduler: ReduceLROnPlateau

2. **DeepNeuralNetwork** (3 hidden layers)
   - Units1: 1024
   - Units2: 512
   - Units3: 256
   - Optimizer: Adam
   - Epochs: 20
   - Scheduler: ReduceLROnPlateau

### RNN (Recurrent Neural Network)

The RNN model is a GRU-based implementation designed for sequence classification tasks, particularly for the Gestures dataset.

**Optimal Hyperparameters:**
- Hidden size: 64
- Number of layers: 2
- Dropout: 0.4
- Optimizer: Adam
- Epochs: 10
- Scheduler: ReduceLROnPlateau with factor=0.5, patience=5
- Early stopping: patience=5

## Usage

The `main.py` script provides a unified interface for training the different model types with their optimal hyperparameters.

### Command-line Arguments

- `--model`: Type of model to train (cnn, dnn, deep_dnn, or gru)
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

Train a Deep DNN model with a batch size of 64:
```bash
python main.py --model deep_dnn --batch-size 64
```

Train a GRU model without saving:
```bash
python main.py --model gru --no-save
```

## Development

To extend this project with new models or datasets:

1. Add new model implementations in the `models/` directory
2. Update the `main.py` script to include functions for creating and training the new models
3. Determine optimal hyperparameters through experimentation in Jupyter notebooks

## Requirements

- Python 3.6+
- PyTorch
- Additional dependencies as specified in the project

## License

This project is licensed under the MIT License - see the LICENSE file for details.