# MNIST Classification with CI/CD Pipeline

![ML Pipeline](https://github.com/{username}/{repository}/actions/workflows/ml-pipeline.yml/badge.svg)

## Project Overview
This project implements a lightweight Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model is designed to be efficient with less than 25,000 parameters while maintaining >95% accuracy on the test set.

### Model Architecture
- 3-layer CNN with batch normalization and dropout
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Parameters: <25,000
- Training: 1 epoch with data augmentation

### Data Augmentation Techniques
- Random rotation (±10 degrees)
- Random translation (up to 10%)
- Random scaling (90%-110%)
- Random erasing
- Normalization

## CI/CD Pipeline Features
- Automated model training
- Parameter count verification
- Input/output shape validation
- Accuracy testing (>95% threshold)
- Model artifact storage

## Project Structure 