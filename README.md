# EMNIST Letters Classification

This project implements a neural network to classify handwritten letters from the EMNIST (Extended MNIST) dataset.

## Project Overview

The goal of this project is to train a multi-layer perceptron (MLP) to recognize handwritten letters (A-Z). It uses PyTorch for building the model and torchvision for handling the EMNIST dataset.

## Key Features

- **Data Preprocessing**: Includes normalization and custom image orientation correction.
- **Model Architecture**: A simple neural network with:
  - Input layer (784 neurons for 28x28 images)
  - Two hidden layers (256 and 128 neurons with ReLU activation)
  - Output layer (26 neurons for letters A-Z)
- **Training Pipeline**: Supports Adam optimizer, Cross-Entropy Loss, and Early Stopping.
- **Evaluation**: Includes per-batch and per-image evaluation functions.



## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

### Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

### Running the Code

Open the `letters.ipynb` notebook in PyCharm or any Jupyter environment and run the cells sequentially.

- **Data Loading**: The dataset will be downloaded automatically to the `./data` directory on the first run.
- **Training**: Adjust `num_epochs` as needed. The default is set to 10 for a balance between speed and performance.
- **Evaluation**: The final cell evaluates the model on the full test set.

## Code Structure

- `correct_image_orientation(image)`: Rotates and flips the EMNIST tensors to the correct human-readable orientation.
- `initialize_emnist_model()`: Defines the network architecture, loss function, and optimizer.
- `train_epoch()`: Handles the training loop for a single epoch.
- `evaluate()`: Computes accuracy on the test loader.
- `train_and_evaluate()`: Orchestrates the full training process with early stopping logic.
- `predict_letter()`: Helper for predicting a single character from a tensor or image.
