# Fashion-MNIST Classification Project

## Overview
This project implements multi-class classification on the Fashion-MNIST dataset using two approaches built from scratch with NumPy:
1. **Logistic Regression Classifier** with L2 regularization
2. **Neural Network with One Hidden Layer** featuring various activation functions and dropout

The Fashion-MNIST dataset consists of 70,000 grayscale images (28x28 pixels) across 10 clothing categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

## Project Structure
- `Ex2.py` - Main implementation file containing all classifiers and utilities
- `train.csv` - Training dataset (56,000 examples with labels)
- `test.csv` - Test dataset (14,000 examples without labels)
- `Report_208520262_208980888.pdf` - Detailed analysis and results report

## Requirements
```
numpy
pandas
matplotlib
scikit-learn
tqdm
```

## How to Run

### Prerequisites
1. Ensure you have Python 3.x installed
2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib scikit-learn tqdm
   ```
3. Place `train.csv` and `test.csv` in the same directory as `Ex2.py`

### Running the Complete Pipeline
Execute the main script:
```bash
python Ex2.py
```

This will automatically run through all three parts:

### Part 1: Data Visualization
- Displays a 10x4 grid showing 4 examples from each of the 10 fashion categories
- Each row represents a different clothing class with proper labels

### Part 2: Logistic Regression Classification
**Hyperparameter Search:**
- Batch sizes: [128, 256, 512]
- Learning rates: [0.001, 0.01, 0.05]  
- Regularization coefficients: [1e-07, 0.001]

**Process:**
1. Automatically splits training data (80% train, 20% validation)
2. Normalizes pixel values using min-max normalization
3. Applies one-hot encoding to labels
4. Tests all hyperparameter combinations
5. Selects best model based on validation accuracy
6. Generates predictions on test set

**Output:** `lr_pred.csv` - Contains predictions for test dataset

### Part 3: Neural Network Classification
**Hyperparameter Search:**
- Batch size: 128
- Learning rate: 0.5
- Regularization coefficient: 1e-08
- Activation functions: [ReLU, Sigmoid, Tanh]
- Hidden layer sizes: [256, 128, 10]
- Dropout probabilities: [1.0, 0.9, 0.8, 0.5] (keep probability)

**Process:**
1. Implements forward and backward propagation from scratch
2. Tests all hyperparameter combinations with progress bars
3. Selects best model based on validation accuracy
4. Applies dropout during training for regularization
5. Generates predictions on test set

**Output:** `NN_pred.csv` - Contains predictions for test dataset

## Key Features

### Implementation Details
- **Numerically Stable Softmax:** Uses `softmax(z - max(z))` to prevent overflow
- **Vectorized Operations:** Efficient NumPy implementations for all computations
- **Mini-batch Gradient Descent:** Configurable batch sizes for optimization
- **L2 Regularization:** Prevents overfitting in both models
- **Dropout:** Neural network includes dropout for additional regularization

### Model Architecture
**Logistic Regression:**
- Multi-class classification using softmax activation
- Cross-entropy loss function
- L2 regularization term

**Neural Network:**
- Input layer: 784 features (28x28 flattened images)
- Hidden layer: Variable size with selectable activation function
- Output layer: 10 classes with softmax activation
- Dropout applied to hidden layer during training

## Results Summary

### Logistic Regression
- **Best Configuration:** Batch size=128, Learning rate=0.001, Regularization=1e-07
- **Performance:** ~87% training accuracy, ~85% validation accuracy
- **Key Finding:** Lower learning rates (0.001) provided more stable training

### Neural Network  
- **Best Configuration:** Hidden size=256, ReLU activation, No dropout
- **Performance:** ~86% training accuracy, ~85% validation accuracy
- **Key Findings:** 
  - ReLU and Tanh outperformed Sigmoid activation
  - Larger hidden layers (256) achieved better performance
  - Dropout showed minimal impact on performance

## Output Files
- `lr_pred.csv` - Logistic regression predictions (one prediction per line, 0-9)
- `NN_pred.csv` - Neural network predictions (one prediction per line, 0-9)

## Notes
- The code includes comprehensive hyperparameter search with progress tracking
- All models are implemented from scratch using only NumPy for core computations
- Training includes real-time loss and accuracy monitoring
- Best models are automatically selected based on validation performance
- Results visualization includes training curves for the optimal configurations

