# MNIST Classification with CI/CD Pipeline

[![ML Pipeline](https://github.com/ksharsha72/ERAV3/actions/workflows/ml-pipeline.yml/badge.svg?branch=main)](https://github.com/ksharsha72/ERAV3/actions/workflows/ml-pipeline.yml)

## Project Overview
This project implements a lightweight Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model is designed to be efficient with less than 25,000 parameters while maintaining >95% accuracy on the test set.

### Model Architecture
- 3-layer CNN with batch normalization and dropout
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Parameters: <25,000
- Training: 1 epoch with data augmentation

### Current Status
- Model Parameters: 21,544
- Target Accuracy: >95%
- Latest Build Status: [![Build Status](https://github.com/ksharsha72/ERAV3/actions/workflows/ml-pipeline.yml/badge.svg?event=push)](https://github.com/ksharsha72/ERAV3/actions)

### Data Augmentation Techniques
- Basic normalization
- No augmentation for better initial training

## CI/CD Pipeline Features
- Automated model training
- Parameter count verification (<25,000)
- Input/output shape validation
- Accuracy testing (>95% threshold)
- Model artifact storage

## Project Structure 