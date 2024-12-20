import random
import math

def activation_function(neuron_value: float, type: str) -> float:
    """
    Compute the activation of a single neuron based on the specified activation function type.

    Args:
        neuron_value (float): The input value to the neuron.
        type (str): The type of activation function to use ("ReLu", "SoftPlus", "Sigmoid").

    Returns:
        float: The activated neuron value.
    """
    if type == "ReLu":
        return max(0, neuron_value)
    elif type == "SoftPlus":
        # Using math.exp(neuron_value) is numerically stable
        # math.log1p(math.exp(neuron_value)) is often more stable than log(1+e^x)
        return math.log1p(math.exp(neuron_value))
    elif type == "Sigmoid":
        # Sigmoid function
        return 1 / (1 + math.exp(-neuron_value))
    else:
        print("Invalid activation function type")
        int("a")  # forces a ValueError


def vector_activation_function(layer: list, type: str) -> list:
    """
    Apply the activation function element-wise to a layer of neurons.

    Args:
        layer (list): The input neuron values for the layer.
        type (str): The type of activation function to apply.

    Returns:
        list: The layer after activation is applied.
    """
    for i in range(len(layer)):
        layer[i] = activation_function(layer[i], type)
    return layer


def softmax(output_layer: list) -> list:
    """
    Apply the softmax function to transform a vector of values into probabilities.

    Args:
        output_layer (list): The output values from the final layer before softmax.

    Returns:
        list: A probability distribution of the same length as output_layer.
    """
    # Use math.exp for numerical stability
    exps = [math.exp(x) for x in output_layer]
    denom = sum(exps)
    return [val / denom for val in exps]


def error_function(output: list, type: str, labels: dict, observed: str) -> float:
    """
    Compute the error based on the specified error function type.

    Supported error functions:
    - "SSD": Sum of Squared Differences
    - "CRE": Cross-Entropy

    Args:
        output (list): The output probabilities/predictions from the network.
        type (str): The type of error function ("SSD" or "CRE").
        labels (dict): A dictionary mapping output indices to label names.
        observed (str): The observed/true label.

    Returns:
        float: The computed error.
    """
    if len(labels) != len(output):
        print("Major Error, labels output size != output size")

    if type == "SSD":
        error = 0.0
        for i in range(len(output)):
            target = 1.0 if labels[i] == observed else 0.0
            error += (target - output[i])**2
        return error

    elif type == "CRE":
        # Cross-entropy error
        for i in range(len(output)):
            if labels[i] == observed:
                # Avoid log(0) by ensuring output[i] > 0.
                # If output[i] is extremely small, math.log might produce a large error term.
                return -math.log(output[i])
        print("Error in error function, label == output not founded")
        int("a")  

    else:
        print("Invalid error function type")
        int("a")  


def sum_vector(vector1: list, vector2: list) -> list:
    """
    Compute the element-wise sum of two vectors.

    Args:
        vector1 (list): The first vector.
        vector2 (list): The second vector.

    Returns:
        list: The element-wise sum of vector1 and vector2.
    """
    if len(vector1) != len(vector2):
        print("Error in sum vector, different vector sizes")
        int("a")  
    return [float(a) + float(b) for a, b in zip(vector1, vector2)]


def dot_product(vector1: list, vector2: list) -> float:
    """
    Compute the dot product of two vectors.

    Args:
        vector1 (list): The first vector.
        vector2 (list): The second vector.

    Returns:
        float: The dot product of vector1 and vector2.
    """
    if len(vector1) != len(vector2):
        print("Error in dot product, different vector sizes")
        int("a")  
    return sum(float(v1) * float(v2) for v1, v2 in zip(vector1, vector2))


def linear_transformation(matrix: list, vector: list) -> list:
    """
    Perform a linear transformation using the given matrix and input vector: output = matrix * vector.

    Args:
        matrix (list): A 2D list representing the transformation matrix.
        vector (list): A 1D list representing the input vector.

    Returns:
        list: The transformed vector.
    """
    if len(matrix[0]) != len(vector):
        print("Error in linear transformation, matrix length != vector size")
        int("a") 
    return [dot_product(row, vector) for row in matrix]


def feed_forward(matrix: list, biases: list, layer: list, type: str) -> list:
    """
    Perform a single feed-forward step of the neural network layer.

    Args:
        matrix (list): The weight matrix for the current layer.
        biases (list): The bias vector for the current layer.
        layer (list): The input vector to the current layer.
        type (str): The activation function type.

    Returns:
        list: The output of the layer after activation.
    """
    layer = linear_transformation(matrix, layer)
    layer = sum_vector(layer, biases)
    layer = vector_activation_function(layer, type)
    return layer


def check_matrix(matrix: list) -> bool:
    """
    Check if all rows of the matrix have the same length.

    Args:
        matrix (list): The matrix to check.

    Returns:
        bool: True if all rows are of equal length, False otherwise.
    """
    return all(len(matrix[i]) == len(matrix[i - 1]) for i in range(1, len(matrix)))


def random_weight() -> float:
    """
    Generate a random weight value between -1 and 1.

    Returns:
        float: A random weight value.
    """
    # Choose sign randomly and multiply by a random float in [0, 1)
    sign = random.choice([1, -1])
    number = random.random() * sign
    return round(number, 4)
