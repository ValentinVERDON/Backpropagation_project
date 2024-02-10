# This file contains the functions used in the main file which are not related to a specific class

# ------------------- Imports ------------------- #

import numpy as np
from Doodler import *

# ------------------- Functions ------------------- #

# NOTE: we should add a function to check if the config file is valid 
#       (i.e. if the parameters are in the right range and if the types are correct)       

# Function to read the config file
def read_config_file(file_path):

    config = {"GLOBALS": {}, "LAYERS": []}
    current_section = None

    with open(file_path, 'r') as file:
        # we read the file line by line
        for line in file:

            # Ignore black line or line starting with #
            if line.strip() == '' or line.startswith('#'):
                continue

            # Set the actual section (GLOBALS or LAYERS)
            if line.startswith("GLOBALS"):
                current_section = "GLOBALS"
                continue
            elif line.startswith("LAYERS"):
                current_section = "LAYERS"
                continue

            # Analyse the line
            key_value_pairs = [pair.strip() for pair in line.split(',')]
            if current_section == "GLOBALS":
                for pair in key_value_pairs:
                    key, value = pair.split(':')
                    config["GLOBALS"][key.strip()] = value.strip()
            elif current_section == "LAYERS":
                layer_info = {}
                for pair in key_value_pairs:
                    key, value = pair.split(':')
                    layer_info[key.strip()] = value.strip()
                config["LAYERS"].append(layer_info)

            # Modify the type of the values
            for key in config["GLOBALS"]:
                if key in ["lrate","wreg"]:
                    config["GLOBALS"][key] = float(config["GLOBALS"][key])

            for layer_info in config["LAYERS"]:
                for key, value in layer_info.items():
                    if key in ["size"]:
                        layer_info[key] = int(value)
                    elif key in ["lrate"]:
                        layer_info[key] = float(value)
                    elif key in ["wr", "br"]:
                        if isinstance(value, str):
                            layer_info[key] = [float(val) for val in value.split()]
                
    if len(config["LAYERS"]) > 7:
        raise ValueError("Too many layers")

    return config

def vizualisation_example(dataset,number=5):
    
    # pic number random index 
    index = np.random.choice(range(len(dataset[0])),number)

    # plot the pictures
    for i in range(number):
        image = dataset[0][index[i]]
        label = dataset[2][index[i]]
        quickplot_matrix(image, fs=None, title='Class = {}'.format(label))