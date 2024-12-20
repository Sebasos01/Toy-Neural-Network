# Toy Neural Network Implementation

A basic Python implementation of a feedforward neural network. It includes utilities for loading data, initializing parameters, training, and computing outputs.

## Files Overview

- **functions.py**  
  - `load_training_data(folder)`: Reads input/output sizes, labels, and training samples.
  - `set_random_parameters(all_layer_sizes, folder)`: Initializes random weights and biases.
  - `load_parameters(folder)`: Loads previously saved network parameters.
  - `update_parameters(new_parameters, folder)`: Saves updated parameters back to disk.
  - `run_neural_network(parameters, input_data)`: Performs a forward pass and returns output probabilities.
  - `loss_function(parameters, labels, training_data)`: Computes the total network error over training data.
  - `optimize_parameters(parameters, labels, training_data, learning_rate)`: Applies gradient descent to improve parameters.
  - `train_neural_network(parameters, labels, training_data)`: Iteratively trains the network, adjusting learning rate if stuck.
  - `assign_labels_output_layer(output_labels)`: Maps output indices to label names.
  - `clear_all(folder)`: Deletes all parameter files and resets the network.

- **matematica.py**  
  - `activation_function(value, type)`: Computes neuron activation (ReLu, SoftPlus, Sigmoid).
  - `softmax(output_layer)`: Converts a raw output vector into a probability distribution.
  - `error_function(output, type, labels, observed)`: Calculates network error (CRE or SSD).
  - `sum_vector(a, b)`, `dot_product(a, b)`, `linear_transformation(matrix, vector)`: Core math operations.
  - `feed_forward(matrix, biases, layer, type)`: Single-layer forward pass.
  - `check_matrix(matrix)`: Verifies matrix consistency.
  - `random_weight()`: Generates a random initial weight.

- **console.py**  
  Command-line interface allowing:
  1. Loading data
  2. Initializing parameters
  3. Running forward passes
  4. Training the network
  5. Viewing/updating parameters

- **parameters/**  
  Directory where generated parameters and lock file are stored.

- **training_data/**  
  Contains example training data (`metadata.txt`, `t_data.txt`).

## Usage

1. Place your training data in `training_data/`.
2. Run `console.py`:
   ```bash
   python console.py
