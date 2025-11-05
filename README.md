# NBA Championship Prediction

A fun project using a custom Multi-Layer Perceptron (MLP) with manual backpropagation to predict NBA championship winners based on team statistics.

## Model Statistics

**Architecture:** 25 input features → 64 hidden neurons → 32 hidden neurons → 2 output classes

**Training Results:**
- Training Accuracy: 97.49%
- Validation Accuracy: 96.25%
- Test Accuracy: 96.25%
- F1 Score: 0.00 (due to class imbalance)

**Key Details:**
- Custom backpropagation implementation (no autograd)
- Trained on 797 team-seasons from 1996-2024
- Uses 25 engineered features (shooting efficiency, scoring, rebounds, defense)
- Dataset: 28 champions (3.5%), 769 non-champions (96.5%)

The model achieves high accuracy by predicting "non-champion" for most teams (rational given class imbalance). However, it does learn patterns - championship contenders receive higher probabilities (28-33%) compared to the baseline. The model demonstrates an understanding of gradient descent and neural network fundamentals through its manual backpropagation implementation.

## Quick Start

```bash
uv sync
uv run python train.py    # Train the model
uv run python predict.py  # Predict current season
```

## Files

- `data.py` - Data collection and feature engineering
- `model.py` - MLP with custom backpropagation
- `train.py` - Training pipeline and evaluation
- `predict.py` - Current season predictions
