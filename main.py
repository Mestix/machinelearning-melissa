#!/usr/bin/env python3
"""
Main script for training machine learning models with optimal hyperparameters.
This script provides a unified interface for training different model types
(CNN, DNN, RNN) with their optimal hyperparameters.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn

# Add models and dev directories to path
sys.path.append(os.path.abspath('./networks'))
sys.path.append(os.path.abspath('./dev'))

# Import models
from networks.CNN import CNN
from NN import NeuralNetwork, DeepNeuralNetwork
from networks.RNN import RecurrentNeuralNetworkWithGRU, ModelConfig

# Import dataset and training utilities
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor, PaddedPreprocessor
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics


def create_cnn_model(filters=32, units=128, num_classes=10):
    """
    Create a CNN model with optimal hyperparameters.
    
    Args:
        num_classes: Number of output classes (default: 10 for Fashion MNIST)
        
    Returns:
        Configured CNN model
    """
    # Optimal hyperparameters from les2.ipynb
    return CNN(filters=filters, units=units, num_classes=num_classes)


def create_neural_network(num_classes=10, units1=300, units2=100):
    """
    Create a DenseNeuralNetwork model with optimal hyperparameters.
    
    Args:
        num_classes: Number of output classes (default: 10 for Fashion MNIST)
        
    Returns:
        Configured DenseNeuralNetwork model
    """
    # Optimal hyperparameters from les1.ipynb
    return NeuralNetwork(num_classes=num_classes, units1=units1, units2=units2)


def create_deep_neural_network(num_classes=10, units1=512, units2=256, units3=128):
    """
    Create a DeepNeuralNetwork model with optimal hyperparameters.
    
    Args:
        num_classes: Number of output classes (default: 10 for Fashion MNIST)
        
    Returns:
        Configured DeepNeuralNetwork model
    """
    # Optimal hyperparameters from les1.ipynb
    return DeepNeuralNetwork(num_classes=num_classes, units1=units1, units2=units2, units3=units3)


def create_gru_model(input_size=3, hiddensize=128, num_layers=2, dropout=0.4, output_size=20):
    """
    Create a GRU model with optimal hyperparameters.
    
    Args:
        input_size: Size of input features (default: 3 for Gestures dataset)
        output_size: Number of output classes (default: 20 for Gestures dataset)
        
    Returns:
        Configured GRU model
    """
    # Optimal hyperparameters from les3.ipynb
    config = ModelConfig(input_size, hiddensize, num_layers, output_size, dropout)
    return RecurrentNeuralNetworkWithGRU(config=config)


def load_fashion_dataset(batch_size=32):
    """
    Load the Fashion MNIST dataset.
    
    Args:
        batch_size: Batch size for data loading (default: 32)
        
    Returns:
        Training and validation data streamers
    """
    factory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    
    streamers = factory.create_datastreamer(batchsize=batch_size, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]
    
    return train.stream(), valid.stream(), len(train), len(valid)


def load_gestures_dataset(batch_size=32):
    """
    Load the Gestures dataset.
    
    Args:
        batch_size: Batch size for data loading (default: 32)
        
    Returns:
        Training and validation data streamers
    """
    factory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
    preprocessor = PaddedPreprocessor()
    
    streamers = factory.create_datastreamer(batchsize=batch_size, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]
    
    return train.stream(), valid.stream(), len(train), len(valid)


def create_trainer_settings(model_type, epochs=10, train_steps=100, valid_steps=100):
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
    accuracy = metrics.Accuracy()
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = Path(f"logs/{model_type}/{timestamp}").resolve()
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    
    # Configure settings based on model type
    if model_type == 'cnn':
        # Settings from les2.ipynb
        settings = TrainerSettings(
            epochs=epochs,
            metrics=[accuracy],
            logdir=log_dir,
            train_steps=train_steps,
            valid_steps=valid_steps,
            reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
            earlystop_kwargs={
                "save": True,
                "verbose": True,
                "patience": 100,
            },
            scheduler_kwargs={
                "factor": 0.5,
                "patience": 10
            }
        )
    elif model_type in ['dnn', 'nn']:
        # Settings from les1.ipynb
        settings = TrainerSettings(
            epochs=epochs,
            metrics=[accuracy],
            logdir=log_dir,
            train_steps=train_steps,
            valid_steps=valid_steps,
            reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
        )
    elif model_type == 'rnn':
        # Settings from les3.ipynb
        settings = TrainerSettings(
            epochs=epochs,
            metrics=[accuracy],
            logdir=log_dir,
            train_steps=train_steps,
            valid_steps=valid_steps,
            reporttypes=[ReportTypes.TOML, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
            scheduler_kwargs={"factor": 0.5, "patience": 5},
            earlystop_kwargs={
                "save": False,
                "verbose": True,
                "patience": 5,
            }
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return settings


def train_model(model_type, epochs=10, batch_size=32, save_model=True):
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

    
    # Create model
    if model_type == 'cnn':
        model = create_cnn_model()
        train_loader, valid_loader, train_steps, valid_steps = load_fashion_dataset(batch_size)
    elif model_type == 'dnn':
        model = create_neural_network()
        train_loader, valid_loader, train_steps, valid_steps = load_fashion_dataset(batch_size)
    elif model_type == 'nn':
        model = create_deep_neural_network()
        train_loader, valid_loader, train_steps, valid_steps = load_fashion_dataset(batch_size)
    elif model_type == 'rnn':
        model = create_gru_model()
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
    parser = argparse.ArgumentParser(description='Train machine learning models with optimal hyperparameters.')
    parser.add_argument('--model', type=str, choices=['cnn', 'dnn', 'nn', 'rnn'], required=True,
                        help='Type of model to train (cnn, dnn, nn, or rnn)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for data loading (default: 32)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the trained model')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_model=not args.no_save
    )


if __name__ == '__main__':
    main()