# Deep Neural Network from Scratch

## Overview
This project implements a deep neural network (DNN) from scratch using NumPy. The network is designed for binary classification tasks, specifically distinguishing between cars and non-cars from image data.

## Features
- Implements a deep neural network with customizable layers.
- Uses ReLU activation for hidden layers and Sigmoid for the output layer.
- Includes forward and backward propagation.
- Implements cost computation using cross-entropy loss.
- Supports gradient descent for parameter updates.
- Loads and preprocesses image datasets.
- Provides model evaluation metrics including accuracy, precision, recall, and F1-score.
- Allows saving and loading trained parameters.
- Supports predictions on new images.

## Project Structure
```
├── deep_neural_network.py  # Main script implementing the neural network
├── README.md               # Project documentation
├── parameters.pkl          # Saved model parameters
├── carnoncar-dataset       # Dataset directory (downloaded via Kaggle)
│   ├── car_scaled         
│   ├── non_car_scaled     
```

## Installation
### Prerequisites
- Python 3.x
- NumPy
- Matplotlib
- PIL (Pillow)
- SciPy
- h5py

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/DebBidhi/Deep-Neural-Network-from-scratch.git
   cd deep-neural-network
   ```
2. Install dependencies:
   ```bash
   pip install numpy matplotlib pillow scipy h5py kaggle
   ```
3. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d izznfkhrlislm/carnoncar-dataset
   unzip carnoncar-dataset.zip -d carnoncar-dataset
   ```

## Usage
### Training the Model
Run the script to train the deep neural network:
```bash
python deep_neural_network.py
```
The model is trained using the `L_layer_model` function and the dataset is split into training and testing sets.

### Making Predictions
To make predictions on new images, use:
```python
from deep_neural_network import predict_from_png

predict_from_png("path/to/image.png", loaded_parameters)
```

## Model Details
### Network Architecture
The network uses the following structure:
```
Input Layer  → Fully Connected + ReLU  → Fully Connected + ReLU  → Output Layer + Sigmoid
```
Example layer dimensions:
```
[20736, 20, 7, 1]  # Input layer has 20736 features (72x72x4), 2 hidden layers, and 1 output neuron
```

### Training Hyperparameters
- Learning rate: `0.001`
- Number of iterations: `3000`
- Cost function: Cross-entropy loss

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## Improvements & Future Work
- Implement batch normalization for better training stability.
- Support different activation functions.
- Convert the model to use TensorFlow/PyTorch for efficiency.
- Extend dataset for better generalization.

## License
This project is open-source under the MIT License.

