import os
from pathlib import Path
import functions
import matematica

def init_neural_network(all_layer_sizes: tuple, folder: str) -> None:
    """
    Initialize the neural network by setting random parameters and saving general information.

    Args:
        all_layer_sizes (tuple): Sizes of each layer in the neural network.
        folder (str): Directory path where parameters will be stored.
    """
    nweights, nbiases = functions.set_random_parameters(all_layer_sizes, folder)
    general_information = ""

    for i, size in enumerate(all_layer_sizes):
        if i == 0:
            general_information += f"Input layer size: {size}\n"
        elif i < len(all_layer_sizes) - 1:
            general_information += f"Hidden layer {i} size: {size}\n"
        else:
            general_information += f"Output layer size: {size}\n"

    general_information += (
        f"Weights: {nweights}\n"
        f"Biases: {nbiases}\n"
        f"Total parameters: {nweights + nbiases}"
    )

    general_info_path = Path(folder) / "[0]general_information.txt"
    with general_info_path.open("w") as handle:
        handle.write(general_information)

    print(general_information)

# Relative paths using pathlib for better portability
BASE_DIR = Path(__file__).parent
PARAMETERS_FOLDER = BASE_DIR / "parameters"
TRAINING_DATA_FOLDER = BASE_DIR / "training_data"

# Example parameters and variables 
new_parameters = [
    [
        [
            [-1.3947, 3.1531],
            [2.1540, -0.3390]
        ],
        [-1.0819, 2.3651]
    ],
    [
        [
            [2.1683, 2.1189],
            [-0.6306, 3.5848],
            [4.1583, -0.3343]
        ],
        [5.0997, 3.6952, 1.2795]
    ]
]
f = [
    -0.0083, -0.0176, -0.0207,
    -0.0431, -0.5360, -0.4218
]

def run() -> None:
    """
    Runs the neural network menu interface, allowing users to perform various actions
    such as loading data, initializing parameters, training, and running the network.
    """
    all_layer_sizes = []
    hidden_layer_sizes = [2]
    parameters = None
    output_labels = None
    nparameters = None
    training_data = None

    menu = """
Neural Network Menu
 1. Load training data
 2. Set the size of the neural network layers and random parameters
 3. Load parameters
 4. Train neural network
 5. Update neural network parameters
 6. Run neural network
 7. Loss function
 8. Delete all data
 9. Done
>>> """

    while True:
        # Check for lock file existence
        lock_file = PARAMETERS_FOLDER / "lock"
        permission = not lock_file.exists()

        option = input(menu).strip()

        if option == "1":
            print("Loading training data...")
            data_info = functions.load_training_data(str(TRAINING_DATA_FOLDER))
            input_size, output_size, output_labels, training_data = data_info
            all_layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
            output_labels = functions.assign_labels_output_layer(output_labels)
            print(f"{len(training_data)} elements loaded")
            print(training_data)

        elif option == "2":
            if permission:
                print("Setting random parameters...")
                init_neural_network(tuple(all_layer_sizes), str(PARAMETERS_FOLDER))
            else:
                print("Prohibited option. To perform this action, delete all existing neural network data first.")

        elif option == "3":
            print("Loading parameters...")
            # Uncomment the line below if you want to load parameters from files
            parameters = functions.load_parameters(str(PARAMETERS_FOLDER))
            #parameters = new_parameters 
            print(parameters)

            try:
                general_info_path = PARAMETERS_FOLDER / "[0]general_information.txt"
                with general_info_path.open("r") as gi:
                    for line in gi:
                        if line.startswith("Total parameters"):
                            nparameters = int(line.split(":")[1].strip())
                            break
            except FileNotFoundError:
                print("General information file not found.")
            except ValueError:
                print("Error parsing the number of parameters.")

            print(f"{nparameters} parameters loaded" if nparameters else "Number of parameters not found.")

            for idx, layer in enumerate(parameters, start=1):
                matrix, biases = layer
                matrix_height = len(matrix)
                matrix_width = len(matrix[0]) if matrix else 0
                bias_length = len(biases)
                layer_type = "Hidden Layer" if idx < len(parameters) else "Output Layer"
                print(
                    f"{layer_type} {idx}: "
                    f"Matrix height-> {matrix_height}, "
                    f"Matrix width-> {matrix_width}, "
                    f"Bias vector length-> {bias_length}"
                )

        elif option == "4":
            if parameters and output_labels and training_data:
                print("Training neural network...")
                parameters = functions.train_neural_network(parameters, output_labels, training_data)
                print("Training completed. Updated parameters:")
                print(parameters)
            else:
                print("Parameters or training data not loaded. Please load them first.")

        elif option == "5":
            if parameters:
                print("Updating parameters...")
                functions.update_parameters(parameters, str(PARAMETERS_FOLDER))
                print("Parameters updated.")
            else:
                print("Parameters not loaded. Please load them first.")

        elif option == "6":
            if parameters and output_labels:
                input_vector = [1, 2]  # Example input; modify as needed
                print("Running neural network with input:", input_vector)
                output = functions.run_neural_network(parameters, input_vector)
                for label, value in zip(output_labels, output):
                    print(f"{label} -> {int(round(value * 100, 0))}%")
            else:
                print("Parameters or output labels not loaded. Please load them first.")

        elif option == "7":
            if parameters and output_labels and training_data:
                loss = functions.loss_function(parameters, output_labels, training_data)
                print(f"Current loss: {loss}")
            else:
                print("Parameters or training data not loaded. Please load them first.")

        elif option == "8":
            confirmation = input("Are you sure?\n 1. Yes\n 2. No\n>>> ").strip()
            if confirmation == "1":
                functions.clear_all(str(PARAMETERS_FOLDER))
                print("Neural network information deleted.")
            else:
                print("Deletion canceled.")

        elif option == "9":
            print("Exiting the neural network program.")
            break

        else:
            print("Invalid option. Please select a valid menu number.")

if __name__ == "__main__":
    run()
