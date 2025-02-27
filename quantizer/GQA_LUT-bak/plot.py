import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from scipy import special

random.seed(42)
np.random.seed(42)

# Activation Functions
ACT_FUNCS = {
    "swish": lambda x: x / (1.0 + np.exp(-x)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1 + special.erf(x / np.sqrt(2))),
    "hswish": lambda x: x * np.clip(x + 3, 0, 6) / 6,
    "exp": lambda x: np.exp(x),
    "reci": lambda x: np.reciprocal(x),
    "sqrt_reci": lambda x: np.reciprocal(np.sqrt(x)),
    "silu": lambda x: x / (1.0 + np.exp(-x))
}

# Load JSON Data
def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

# Piecewise Linear Function based on JSON
def piecewise_linear(x, breakpoints, slopes, intercepts):
    for i in range(len(breakpoints) - 1):
        if breakpoints[i] <= x < breakpoints[i + 1]:
            return slopes[i] * x + intercepts[i]
    return slopes[-1] * x + intercepts[-1]

# Plot Activation Functions vs Piecewise Linear Approximation
def plot_activation_vs_pwl(json_data, act_func_name, output_dir):
    x_values = np.linspace(-5, 5, 1000)
    activation_func = ACT_FUNCS[act_func_name]
    y_activation = activation_func(x_values)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot for each set of parameters in the JSON file
    for key, params in json_data[act_func_name].items():
        breakpoints = params['breakpoints']
        slopes = params['slopes']
        intercepts = params['intercepts']

        y_pwl = [piecewise_linear(x, breakpoints, slopes, intercepts) for x in x_values]

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_activation, label=f'{act_func_name} Activation Function', color='b')
        plt.plot(x_values, y_pwl, label=f'PWL Approximation - Set {key}', color='r', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{act_func_name} vs PWL Approximation (Set {key})')
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(output_dir, f'{act_func_name}_pwl_set_{key}.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Plot saved: {output_path}')

if __name__ == "__main__":
    # Load the JSON file
    json_filepath = "./pretrained/silu_pwl_7.json"
    json_data = load_json(json_filepath)

    # Plot comparison for SiLU function
    plot_activation_vs_pwl(json_data, "silu", output_dir="plots")
