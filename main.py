#!/usr/bin/env python3
"""
Main script for training machine learning models with optimal hyperparameters.
This script provides a unified interface for training different model types
(CNN, DNN, RNN) with their optimal hyperparameters.

Configuration is loaded from config.toml file, which can be overridden by command-line arguments.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import toml
import torch
import torch.optim as optim
import torch.nn as nn

# Add models and dev directories to path
sys.path.append(os.path.abspath('./networks'))
sys.path.append(os.path.abspath('./dev'))

# Load configuration from config.toml
def load_config(config_path="config.toml"):
    """
    Load configuration from a TOML file.
    
    Args:
        config_path: Path to the configuration file (default: config.toml)
        
    Returns:
        Configuration dictionary
    """
    try:
        config = toml.load(config_path)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default values.")
        return {}
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}

# Global configuration
CONFIG = load_config()

# Import models
from networks.CNN import CNN
from NN import NeuralNetwork, DeepNeuralNetwork
from networks.RNN import (
    RecurrentNeuralNetworkWithGRU, 
    RecurrentNeuralNetwork, 
    GRUWithAttention, 
    RecurrentNeuralNetworkWithAttention, 
    ModelConfig
)

# Import dataset and training utilities
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor, PaddedPreprocessor
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics


def create_cnn_model(filters=None, units=None, num_classes=None):
    """
    Create a CNN model with optimal hyperparameters.
    
    Args:
        filters: Number of filters in convolutional layers
        units: Number of units in dense layer
        num_classes: Number of output classes
        
    Returns:
        Configured CNN model
    """
    # Get parameters from config or use defaults
    if filters is None:
        filters = CONFIG.get("models", {}).get("cnn", {}).get("filters", 32)
    if units is None:
        units = CONFIG.get("models", {}).get("cnn", {}).get("units", 128)
    if num_classes is None:
        num_classes = CONFIG.get("models", {}).get("cnn", {}).get("num_classes", 10)
    
    return CNN(filters=filters, units=units, num_classes=num_classes)


def create_neural_network(num_classes=None, units1=None, units2=None):
    """
    Create a DenseNeuralNetwork model with optimal hyperparameters.
    
    Args:
        num_classes: Number of output classes
        units1: Number of units in first dense layer
        units2: Number of units in second dense layer
        
    Returns:
        Configured DenseNeuralNetwork model
    """
    # Get parameters from config or use defaults
    if num_classes is None:
        num_classes = CONFIG.get("models", {}).get("nn", {}).get("num_classes", 10)
    if units1 is None:
        units1 = CONFIG.get("models", {}).get("nn", {}).get("units1", 300)
    if units2 is None:
        units2 = CONFIG.get("models", {}).get("nn", {}).get("units2", 100)
    
    return NeuralNetwork(num_classes=num_classes, units1=units1, units2=units2)


def create_deep_neural_network(num_classes=None, units1=None, units2=None, units3=None):
    """
    Create a DeepNeuralNetwork model with optimal hyperparameters.
    
    Args:
        num_classes: Number of output classes
        units1: Number of units in first dense layer
        units2: Number of units in second dense layer
        units3: Number of units in third dense layer
        
    Returns:
        Configured DeepNeuralNetwork model
    """
    # Get parameters from config or use defaults
    if num_classes is None:
        num_classes = CONFIG.get("models", {}).get("dnn", {}).get("num_classes", 10)
    if units1 is None:
        units1 = CONFIG.get("models", {}).get("dnn", {}).get("units1", 512)
    if units2 is None:
        units2 = CONFIG.get("models", {}).get("dnn", {}).get("units2", 256)
    if units3 is None:
        units3 = CONFIG.get("models", {}).get("dnn", {}).get("units3", 128)
    
    return DeepNeuralNetwork(num_classes=num_classes, units1=units1, units2=units2, units3=units3)


def create_gru_model(input_size=None, hiddensize=None, num_layers=None, dropout=None, output_size=None):
    """
    Create a GRU model with optimal hyperparameters.
    
    Args:
        input_size: Size of input features
        hiddensize: Size of hidden layer
        num_layers: Number of recurrent layers
        dropout: Dropout rate
        output_size: Number of output classes
        
    Returns:
        Configured GRU model
    """
    # Get parameters from config or use defaults
    if input_size is None:
        input_size = CONFIG.get("models", {}).get("rnn", {}).get("input_size", 3)
    if hiddensize is None:
        hiddensize = CONFIG.get("models", {}).get("rnn", {}).get("hidden_size", 128)
    if num_layers is None:
        num_layers = CONFIG.get("models", {}).get("rnn", {}).get("num_layers", 2)
    if dropout is None:
        dropout = CONFIG.get("models", {}).get("rnn", {}).get("dropout", 0.4)
    if output_size is None:
        output_size = CONFIG.get("models", {}).get("rnn", {}).get("output_size", 20)
    
    config = ModelConfig(input_size, hiddensize, num_layers, output_size, dropout)
    return RecurrentNeuralNetworkWithGRU(config=config)


def create_rnn_model(input_size=None, hiddensize=None, num_layers=None, dropout=None, output_size=None):
    """
    Create a basic RNN model with optimal hyperparameters.
    
    Args:
        input_size: Size of input features
        hiddensize: Size of hidden layer
        num_layers: Number of recurrent layers
        dropout: Dropout rate
        output_size: Number of output classes
        
    Returns:
        Configured RNN model
    """
    # Get parameters from config or use defaults
    if input_size is None:
        input_size = CONFIG.get("models", {}).get("rnn", {}).get("input_size", 3)
    if hiddensize is None:
        hiddensize = CONFIG.get("models", {}).get("rnn", {}).get("hidden_size", 128)
    if num_layers is None:
        num_layers = CONFIG.get("models", {}).get("rnn", {}).get("num_layers", 2)
    if dropout is None:
        dropout = CONFIG.get("models", {}).get("rnn", {}).get("dropout", 0.4)
    if output_size is None:
        output_size = CONFIG.get("models", {}).get("rnn", {}).get("output_size", 20)
    
    config = ModelConfig(input_size, hiddensize, num_layers, output_size, dropout)
    return RecurrentNeuralNetwork(config=config)


def create_gru_attention_model(input_size=None, hiddensize=None, num_layers=None, dropout=None, output_size=None):
    """
    Create a GRU model with attention mechanism and optimal hyperparameters.
    
    Args:
        input_size: Size of input features
        hiddensize: Size of hidden layer
        num_layers: Number of recurrent layers
        dropout: Dropout rate
        output_size: Number of output classes
        
    Returns:
        Configured GRU with Attention model
    """
    # Get parameters from config or use defaults
    if input_size is None:
        input_size = CONFIG.get("models", {}).get("rnn", {}).get("input_size", 3)
    if hiddensize is None:
        hiddensize = CONFIG.get("models", {}).get("rnn", {}).get("hidden_size", 128)
    if num_layers is None:
        num_layers = CONFIG.get("models", {}).get("rnn", {}).get("num_layers", 2)
    if dropout is None:
        dropout = CONFIG.get("models", {}).get("rnn", {}).get("dropout", 0.4)
    if output_size is None:
        output_size = CONFIG.get("models", {}).get("rnn", {}).get("output_size", 20)
    
    config = ModelConfig(input_size, hiddensize, num_layers, output_size, dropout)
    return GRUWithAttention(config=config)


def create_rnn_attention_model(input_size=None, hiddensize=None, num_layers=None, dropout=None, output_size=None):
    """
    Create a RNN model with attention mechanism and optimal hyperparameters.
    
    Args:
        input_size: Size of input features
        hiddensize: Size of hidden layer
        num_layers: Number of recurrent layers
        dropout: Dropout rate
        output_size: Number of output classes
        
    Returns:
        Configured RNN with Attention model
    """
    # Get parameters from config or use defaults
    if input_size is None:
        input_size = CONFIG.get("models", {}).get("rnn", {}).get("input_size", 3)
    if hiddensize is None:
        hiddensize = CONFIG.get("models", {}).get("rnn", {}).get("hidden_size", 128)
    if num_layers is None:
        num_layers = CONFIG.get("models", {}).get("rnn", {}).get("num_layers", 2)
    if dropout is None:
        dropout = CONFIG.get("models", {}).get("rnn", {}).get("dropout", 0.4)
    if output_size is None:
        output_size = CONFIG.get("models", {}).get("rnn", {}).get("output_size", 20)
    
    config = ModelConfig(input_size, hiddensize, num_layers, output_size, dropout)
    return RecurrentNeuralNetworkWithAttention(config=config)


def load_fashion_dataset(batch_size=None):
    """
    Load the Fashion MNIST dataset.
    
    Args:
        batch_size: Batch size for data loading (default: 32)
        
    Returns:
        Training and validation data streamers
    """
    # Get batch size from config or use default
    if batch_size is None:
        batch_size = CONFIG.get("datasets", {}).get("fashion", {}).get("batch_size", 32)
    
    factory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    
    streamers = factory.create_datastreamer(batchsize=batch_size, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]
    
    return train.stream(), valid.stream(), len(train), len(valid)

def load_gestures_dataset(batch_size=None):
    """
    Load the Gestures dataset.
    
    Args:
        batch_size: Batch size for data loading (default: 32)
        
    Returns:
        Training and validation data streamers
    """
    # Get batch size from config or use default
    if batch_size is None:
        batch_size = CONFIG.get("datasets", {}).get("gestures", {}).get("batch_size", 32)
    
    factory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
    preprocessor = PaddedPreprocessor()
    
    streamers = factory.create_datastreamer(batchsize=batch_size, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]
    
    return train.stream(), valid.stream(), len(train), len(valid)

def create_trainer_settings(model_type, epochs=None, train_steps=None, valid_steps=None):
    """
    Create trainer settings with appropriate configuration for the model type.
    
    Args:
        model_type: Type of model ('cnn', 'nn', 'dnn', or 'rnn')
        epochs: Number of training epochs
        train_steps: Number of training steps per epoch
        valid_steps: Number of validation steps per epoch
        
    Returns:
        Configured TrainerSettings object
    """
    # Get parameters from config or use defaults
    if epochs is None:
        epochs = CONFIG.get("training", {}).get("epochs", 10)
    if train_steps is None:
        train_steps = CONFIG.get("training", {}).get("train_steps", 100)
    if valid_steps is None:
        valid_steps = CONFIG.get("training", {}).get("valid_steps", 100)
    
    accuracy = metrics.Accuracy()
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    base_log_dir = CONFIG.get("logging", {}).get("log_dir", "logs")
    log_dir = Path(f"{base_log_dir}/{model_type}/{timestamp}").resolve()
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    
    # Configure settings based on model type
    if model_type == 'cnn':
        # Settings from les2.ipynb
        # Get report types from config or use defaults
        report_types_str = CONFIG.get("logging", {}).get("report_types", ["TENSORBOARD", "TOML"])
        report_types = [getattr(ReportTypes, rt) for rt in report_types_str if hasattr(ReportTypes, rt)]
        
        # Get early stopping parameters from config
        early_stop_enabled = CONFIG.get("early_stopping", {}).get("enabled", True)
        early_stop_kwargs = {}
        if early_stop_enabled:
            early_stop_kwargs = {
                "save": CONFIG.get("early_stopping", {}).get("save", True),
                "verbose": CONFIG.get("early_stopping", {}).get("verbose", True),
                "patience": CONFIG.get("early_stopping", {}).get("patience", 5),
            }
        
        settings = TrainerSettings(
            epochs=epochs,
            metrics=[accuracy],
            logdir=log_dir,
            train_steps=train_steps,
            valid_steps=valid_steps,
            reporttypes=report_types,
            earlystop_kwargs=early_stop_kwargs if early_stop_enabled else None,
        )
    elif model_type in ['dnn', 'nn']:
        # Settings from les1.ipynb
        # Get report types from config or use defaults
        report_types_str = CONFIG.get("logging", {}).get("report_types", ["TENSORBOARD", "TOML"])
        report_types = [getattr(ReportTypes, rt) for rt in report_types_str if hasattr(ReportTypes, rt)]
        
        settings = TrainerSettings(
            epochs=epochs,
            metrics=[accuracy],
            logdir=log_dir,
            train_steps=train_steps,
            valid_steps=valid_steps,
            reporttypes=report_types,
        )
    elif model_type in ['rnn_gru', 'rnn_basic', 'gru_attention', 'rnn_attention']:
        # Settings from les3.ipynb
        # Get report types from config or use defaults
        report_types_str = CONFIG.get("logging", {}).get("report_types", ["TOML", "TENSORBOARD", "MLFLOW"])
        report_types = [getattr(ReportTypes, rt) for rt in report_types_str if hasattr(ReportTypes, rt)]
        
        # Get early stopping parameters from config
        early_stop_enabled = CONFIG.get("early_stopping", {}).get("enabled", True)
        early_stop_kwargs = {}
        if early_stop_enabled:
            early_stop_kwargs = {
                "save": CONFIG.get("early_stopping", {}).get("save", False),
                "verbose": CONFIG.get("early_stopping", {}).get("verbose", True),
                "patience": CONFIG.get("early_stopping", {}).get("patience", 5),
            }
        
        settings = TrainerSettings(
            epochs=epochs,
            metrics=[accuracy],
            logdir=log_dir,
            train_steps=train_steps,
            valid_steps=valid_steps,
            reporttypes=report_types,
            earlystop_kwargs=early_stop_kwargs if early_stop_enabled else None,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return settings


def train_model(model_type, epochs=None, batch_size=None, save_model=None):
    """
    Train a model with optimal hyperparameters.
    
    Args:
        model_type: Type of model ('cnn', 'dnn', 'deep_dnn', or 'gru')
        epochs: Number of training epochs (if None, use default for model type)
        batch_size: Batch size for data loading (default: 32)
        save_model: Whether to save the trained model (default: True)
        
    Returns:
        Trained model and training history
    """
    # Get parameters from config or use defaults
    if epochs is None:
        epochs = CONFIG.get("training", {}).get("epochs", 10)
    if batch_size is None:
        batch_size = CONFIG.get("training", {}).get("batch_size", 32)
    if save_model is None:
        save_model = CONFIG.get("general", {}).get("save_model", True)
    
    # Create model
    if model_type == 'cnn':
        model = create_cnn_model()
        train_loader, valid_loader, train_steps, valid_steps = load_fashion_dataset(batch_size)
    elif model_type == 'dnn':
        model = create_deep_neural_network()
        train_loader, valid_loader, train_steps, valid_steps = load_fashion_dataset(batch_size)
    elif model_type == 'nn':
        model = create_neural_network()
        train_loader, valid_loader, train_steps, valid_steps = load_fashion_dataset(batch_size)
    elif model_type == 'rnn_gru':
        model = create_gru_model()
        train_loader, valid_loader, train_steps, valid_steps = load_gestures_dataset(batch_size)
    elif model_type == 'rnn_basic':
        model = create_rnn_model()
        train_loader, valid_loader, train_steps, valid_steps = load_gestures_dataset(batch_size)
    elif model_type == 'gru_attention':
        model = create_gru_attention_model()
        train_loader, valid_loader, train_steps, valid_steps = load_gestures_dataset(batch_size)
    elif model_type == 'rnn_attention':
        model = create_rnn_attention_model()
        train_loader, valid_loader, train_steps, valid_steps = load_gestures_dataset(batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create trainer settings
    settings = create_trainer_settings(model_type, epochs, train_steps, valid_steps)
    
    # Create loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train_loader,
        validdataloader=valid_loader,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    )
    
    # Train model
    print(f"Training {model_type} model for {epochs} epochs...")
    trainer.loop()
    
    # Save model if requested
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        model_dir = Path(f"saved_models/{model_type}").resolve()
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        
        model_path = model_dir / f"{timestamp}_model.pt"
        torch.save(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model, trainer


def main():
    """
    Main function to parse arguments and train models.
    """

    global CONFIG

    # Load configuration
    config = CONFIG
    
    parser = argparse.ArgumentParser(description='Train machine learning models with optimal hyperparameters.')
    parser.add_argument('--config', type=str, default='config.toml',
                        help='Path to configuration file (default: config.toml)')
    parser.add_argument('--model', type=str, 
                        choices=['cnn', 'dnn', 'nn', 'rnn_gru', 'rnn_basic', 'gru_attention', 'rnn_attention'], 
                        required=True,
                        help='Type of model to train (cnn, dnn, nn, rnn_gru, rnn_basic, gru_attention, or rnn_attention)')
    parser.add_argument('--epochs', type=int,
                        help=f'Number of training epochs (default: {config.get("training", {}).get("epochs", 10)})')
    parser.add_argument('--batch-size', type=int,
                        help=f'Batch size for data loading (default: {config.get("training", {}).get("batch_size", 32)})')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the trained model')
    
    args = parser.parse_args()
    
    # If a different config file is specified, reload the configuration
    if args.config != 'config.toml':
        CONFIG = load_config(args.config)
    
    # Train model
    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_model=not args.no_save if args.no_save else None
    )


if __name__ == '__main__':
    main()