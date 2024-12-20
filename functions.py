from pathlib import Path
import glob
import matematica
import os
from typing import Tuple, List, Dict, Any

def load_training_data(folder: str) -> Tuple[int, int, List[str], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
    """
    Load training data from the specified folder.

    Args:
        folder (str): Directory path where training data files are located.

    Returns:
        Tuple containing input size, output size, output labels, and training data.
    """
    metadata_path = Path(folder) / "metadata.txt"
    with metadata_path.open("r") as f:
        lines = f.read().replace(" ", "").split("\n")
    metadata = lines[1].split(";")
    input_size = int(metadata[0])
    output_size = int(metadata[1])
    output_labels = metadata[3].split(",")

    training_data = []
    t_data_path = Path(folder) / "t_data.txt"
    with t_data_path.open("r") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(";")
            label = parts[-1]
            input_values = tuple(parts[:-1])
            training_data.append((label, input_values))

    return (input_size, output_size, output_labels, tuple(training_data))


def set_random_parameters(all_layer_sizes: Tuple[int, ...], folder: str) -> Tuple[int, int]:
    """
    Initialize random weights and biases for the neural network and save them to files.

    Args:
        all_layer_sizes (Tuple[int, ...]): Sizes of each layer in the neural network.
        folder (str): Directory path where parameter files will be stored.

    Returns:
        Tuple containing the number of weights and biases.
    """
    nweights, nbiases = 0, 0
    for i in range(1, len(all_layer_sizes)):
        layer_type = "hidden_layer" if i < len(all_layer_sizes) - 1 else "output_layer"
        parameter_file = Path(folder) / f"[{i}]parameters_{layer_type}_{i}.txt"
        with parameter_file.open("w") as handle:
            for _ in range(all_layer_sizes[i]):
                weights = [str(matematica.random_weight()) for _ in range(all_layer_sizes[i-1])]
                handle.write(";".join(weights) + ";0\n")
                nweights += all_layer_sizes[i-1]
                nbiases += 1

    # Create a lock file to indicate parameters are set
    lock_path = Path(folder) / "lock"
    lock_path.touch(exist_ok=True)

    return (nweights, nbiases)


def load_parameters(folder: str) -> List[List[Any]]:
    """
    Load neural network parameters (weights and biases) from parameter files.

    Args:
        folder (str): Directory path where parameter files are stored.

    Returns:
        List containing weight matrices and bias vectors for each layer.
    """
    parameters = []
    parameter_files = glob.glob(str(Path(folder) / "*.txt"))
    for file_path in parameter_files:
        file_name = Path(file_path).name
        if not file_name.startswith("[0]"):
            weight_matrix = []
            bias_vector = []
            with Path(file_path).open("r") as file:
                for line in file:
                    parts = line.strip().split(";")
                    weights = parts[:-1]
                    bias = parts[-1]
                    weight_matrix.append(weights)
                    bias_vector.append(bias)
            parameters.append([weight_matrix, bias_vector])
    return parameters


def update_parameters(new_parameters: List[List[Any]], folder: str) -> None:
    """
    Update neural network parameters by writing them back to parameter files.

    Args:
        new_parameters (List[List[Any]]): Updated weights and biases.
        folder (str): Directory path where parameter files are stored.
    """
    for idx, (matrix, biases) in enumerate(new_parameters, start=1):
        layer_type = "hidden_layer" if idx < len(new_parameters) else "output_layer"
        parameter_file = Path(folder) / f"[{idx}]parameters_{layer_type}_{idx}.txt"
        with parameter_file.open("w") as handle:
            for weight_row, bias in zip(matrix, biases):
                line = ";".join(map(str, weight_row)) + f";{bias}\n"
                handle.write(line)


def run_neural_network(parameters: List[List[Any]], input_data: Tuple[str, ...]) -> List[float]:
    """
    Run a forward pass of the neural network with the given input.

    Args:
        parameters (List[List[Any]]): Neural network parameters.
        input_data (Tuple[str, ...]): Input data for the neural network.

    Returns:
        List of output probabilities after softmax.
    """
    layer = [float(x) for x in input_data]
    for idx, (matrix, biases) in enumerate(parameters, start=1):
        if matematica.check_matrix(matrix):
            layer = matematica.feed_forward(matrix, biases, layer, "SoftPlus")
        else:
            print(f"Matrix error in layer {idx}")
            exit()
    layer = matematica.softmax(layer)
    return layer


def loss_function(parameters: List[List[Any]], labels: Dict[int, str], training_data: Tuple[Tuple[str, Tuple[str, ...]], ...]) -> float:
    """
    Calculate the total loss over the training data.

    Args:
        parameters (List[List[Any]]): Neural network parameters.
        labels (Dict[int, str]): Mapping of output indices to label names.
        training_data (Tuple[Tuple[str, Tuple[str, ...]], ...]): Training data.

    Returns:
        Total loss as a float.
    """
    total_error = 0.0
    for label, input_values in training_data:
        output = run_neural_network(parameters, input_values)
        total_error += matematica.error_function(output, "CRE", labels, label)
    return total_error


def optimize_parameters(parameters: List[List[Any]], labels: Dict[int, str], training_data: Tuple[Tuple[str, Tuple[str, ...]], ...], learning_rate: float) -> List[List[Any]]:
    """
    Optimize neural network parameters using gradient descent.

    Args:
        parameters (List[List[Any]]): Current neural network parameters.
        labels (Dict[int, str]): Mapping of output indices to label names.
        training_data (Tuple[Tuple[str, Tuple[str, ...]], ...]): Training data.
        learning_rate (float): Learning rate for gradient descent.

    Returns:
        List containing optimized weights and biases.
    """
    optimized_parameters = [[layer_weights.copy(), bias_vec.copy()] for layer_weights, bias_vec in parameters]
    dp = 0.0001
    current_loss = loss_function(parameters, labels, training_data)

    for i, (matrix, biases) in enumerate(parameters):
        for k, (weight_row, bias) in enumerate(zip(matrix, biases)):
            for l, weight in enumerate(weight_row):
                # Optimize weight
                modified_parameters = [[wts.copy(), bvs.copy()] for wts, bvs in parameters]
                modified_parameters[i][0][k][l] = str(float(weight) + dp)
                new_loss = loss_function(modified_parameters, labels, training_data)
                gradient = (new_loss - current_loss) / dp
                optimized_weight = float(weight) - gradient * learning_rate
                optimized_parameters[i][0][k][l] = str(optimized_weight)

            # Optimize bias
            modified_parameters = [[wts.copy(), bvs.copy()] for wts, bvs in parameters]
            modified_parameters[i][1][k] = str(float(bias) + dp)
            new_loss = loss_function(modified_parameters, labels, training_data)
            gradient = (new_loss - current_loss) / dp
            optimized_bias = float(bias) - gradient * learning_rate
            optimized_parameters[i][1][k] = str(optimized_bias)

    return optimized_parameters


def train_neural_network(parameters: List[List[Any]], labels: Dict[int, str], training_data: Tuple[Tuple[str, Tuple[str, ...]], ...]) -> List[List[Any]]:
    """
    Train the neural network by optimizing parameters over multiple steps.
    If the loss stops decreasing for a prolonged period, attempt to resolve by reducing the learning rate.

    Args:
        parameters (List[List[Any]]): Initial neural network parameters.
        labels (Dict[int, str]): Mapping of output indices to label names.
        training_data (Tuple[Tuple[str, Tuple[str, ...]], ...]): Training data.

    Returns:
        List containing trained weights and biases.
    """
    steps = 50_000
    learning_rate = 0.0003
    no_improvement_steps = 0
    improvement_threshold = 1e-7  # Minimum improvement in loss to consider progress
    stuck_limit = 2000  # Number of steps with no improvement allowed
    prev_loss = loss_function(parameters, labels, training_data)

    for step in range(steps):
        parameters = optimize_parameters(parameters, labels, training_data, learning_rate)
        current_loss = loss_function(parameters, labels, training_data)

        if step % 1000 == 0:
            print(f"Step {step}, Loss: {current_loss}")

        # Check if improvement is occurring
        if prev_loss - current_loss < improvement_threshold:
            no_improvement_steps += 1
        else:
            no_improvement_steps = 0

        # If stuck, reduce learning rate to attempt escaping plateau
        if no_improvement_steps > stuck_limit:
            learning_rate *= 0.5
            no_improvement_steps = 0
            print(f"Loss appears stuck. Reducing learning rate to {learning_rate} and continuing training.")

        prev_loss = current_loss

    return parameters


def assign_labels_output_layer(output_labels: List[str]) -> Dict[int, str]:
    """
    Assign labels to the output layer indices.

    Args:
        output_labels (List[str]): List of output label names.

    Returns:
        Dictionary mapping indices to label names.
    """
    return {i: label for i, label in enumerate(output_labels)}


def clear_all(folder: str) -> None:
    """
    Delete all parameter and lock files in the specified folder.

    Args:
        folder (str): Directory path where parameter files are stored.
    """
    lock_path = Path(folder) / "lock"
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass  # If lock file doesn't exist, do nothing

    parameter_files = glob.glob(str(Path(folder) / "*.txt"))
    for file_path in parameter_files:
        Path(file_path).unlink()
