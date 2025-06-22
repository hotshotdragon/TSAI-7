# MNIST Classification with PyTorch 99.4% accuracy with ~8000 parameters under 15 epochs

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

This project implements a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch. The model is designed to achieve high accuracy with a minimal number of parameters.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hotshotdragon/TSAI-7.git
   cd mnist-classification
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train and test the model, run:
```bash
python main.py
```

## Model Architecture

The model is a simple CNN with the following layers:
- Convolutional layers with ReLU activation and Batch Normalization
- Max Pooling layers
- Global Average Pooling
- Fully connected layer for classification

## Results

The model achieves over 99.40% accuracy on the MNIST test set within training for 15 epochs.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
