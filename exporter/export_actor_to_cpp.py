#!/usr/bin/env python3
"""
Export trained TD3 actor weights to C++ header format for ESP32 deployment.
Usage: python export_actor_to_cpp.py --model models/td3/your_model/best_model.pt
"""

import argparse
import os
import yaml
import torch
import numpy as np

def load_actor_and_config(model_path):
    """Load the trained actor and its configuration."""
    # Add parent directory to path to import TD3Agent
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from controllers.td3_controller import TD3Agent
    
    # Load config from the same directory as the model
    run_dir = os.path.dirname(model_path)
    config_path = os.path.join(run_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint to get dimensions
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dim = checkpoint.get('state_dim', 3)
    action_dim = checkpoint.get('action_dim', 2)
    max_action = checkpoint.get('max_action', 1.0)
    actor_hidden_sizes = checkpoint.get('actor_hidden_sizes', [64, 64])
    critic_hidden_sizes = checkpoint.get('critic_hidden_sizes', [400, 300])
    
    # Create agent with correct dimensions
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_hidden_sizes=actor_hidden_sizes,
        critic_hidden_sizes=critic_hidden_sizes,
        actor_lr=0.001,  # Not used for export
        critic_lr=0.001   # Not used for export
    )
    
    # Load the trained weights
    agent.load(model_path)
    actor = agent.actor.eval()
    
    return actor, config, max_action, state_dim, action_dim

def extract_layers_from_actor(actor):
    """Extract weight matrices and biases from the actor network."""
    layers = []
    
    # Extract hidden layers (with ReLU activation)
    for i, linear_layer in enumerate(actor.hidden_layers):
        weight = linear_layer.weight.detach().cpu().numpy()  # Shape: [out_features, in_features]
        bias = linear_layer.bias.detach().cpu().numpy()      # Shape: [out_features]
        layers.append({
            'weight': weight,
            'bias': bias,
            'activation': 'relu',
            'name': f'hidden_{i}'
        })
    
    # Extract output layer (with tanh activation, scaled by max_action)
    output_weight = actor.out.weight.detach().cpu().numpy()
    output_bias = actor.out.bias.detach().cpu().numpy()
    layers.append({
        'weight': output_weight,
        'bias': output_bias,
        'activation': 'tanh',
        'name': 'output'
    })
    
    return layers

def format_array_cpp(arr, indent=2):
    """Format a numpy array as C++ array initialization."""
    if arr.ndim == 1:
        # 1D array: { val1, val2, val3 }
        values = ", ".join(f"{val:.8f}f" for val in arr)
        return f"{{ {values} }}"
    elif arr.ndim == 2:
        # 2D array: { {row1}, {row2}, {row3} }
        rows = []
        for row in arr:
            row_str = ", ".join(f"{val:.8f}f" for val in row)
            rows.append(f"  {{ {row_str} }}")
        return "{\n" + ",\n".join(rows) + "\n}"
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")

def generate_cpp_header(layers, config, max_action, state_dim, action_dim):
    """Generate C++ header code with the network weights."""
    
    # Start building the header
    header_lines = [
        "// Auto-generated TD3 Actor weights for ESP32",
        "// Generated from trained PyTorch model",
        "#pragma once",
        "",
        "// Network configuration",
        f"static const int ACTOR_INPUT_DIM = {state_dim};",
        f"static const int ACTOR_OUTPUT_DIM = {action_dim};",
        f"static const int ACTOR_NUM_LAYERS = {len(layers)};",
        f"static const float ACTOR_MAX_ACTION = {max_action:.6f}f;",
        ""
    ]
    
    # Add past states configuration if enabled
    past_states = config.get('past_states', {})
    if past_states.get('enabled', False):
        source = past_states.get('source', 'wheels')
        header_lines.extend([
            "// Past states configuration",
            f"static const bool PAST_STATES_ENABLED = true;",
            f"static const int PAST_STATES_COUNT = {past_states.get('count', 0)};",
            f"static const int PAST_STATES_STRIDE = {past_states.get('stride', 1)};",
            f"static const bool PAST_STATES_USE_SENSORS = {str(source == 'sensors').lower()};",
            f"static const bool PAST_STATES_USE_WHEELS = {str(source == 'wheels').lower()};",
            ""
        ])
    else:
        header_lines.extend([
            "// Past states configuration",
            f"static const bool PAST_STATES_ENABLED = false;",
            f"static const bool PAST_STATES_USE_SENSORS = false;",
            f"static const bool PAST_STATES_USE_WHEELS = false;",
            ""
        ])
    
    # Add layer dimensions
    header_lines.append("// Layer dimensions")
    for i, layer in enumerate(layers):
        out_dim, in_dim = layer['weight'].shape
        header_lines.extend([
            f"static const int L{i}_IN = {in_dim};",
            f"static const int L{i}_OUT = {out_dim};",
            f"static const char L{i}_ACTIVATION[] = \"{layer['activation']}\";"
        ])
    header_lines.append("")
    
    # Add weight matrices and biases
    for i, layer in enumerate(layers):
        header_lines.extend([
            f"// Layer {i} ({layer['name']}) weights and biases",
            f"static const float L{i}_WEIGHTS[L{i}_OUT][L{i}_IN] = ",
            format_array_cpp(layer['weight']) + ";",
            "",
            f"static const float L{i}_BIASES[L{i}_OUT] = ",
            format_array_cpp(layer['bias']) + ";",
            ""
        ])
    
    return "\n".join(header_lines)

def main():
    parser = argparse.ArgumentParser(description="Export TD3 actor to C++ header")
    parser.add_argument("--model", required=True, help="Path to the trained model (.pt file)")
    parser.add_argument("--output", help="Output file path (optional, auto-generated if not specified)")
    
    args = parser.parse_args()
    
    try:
        # Load the model and configuration
        print(f"Loading model: {args.model}")
        actor, config, max_action, state_dim, action_dim = load_actor_and_config(args.model)
        
        # Extract layers
        print(f"Extracting layers from actor...")
        layers = extract_layers_from_actor(actor)
        
        print(f"Model info:")
        print(f"  Input dim: {state_dim}")
        print(f"  Output dim: {action_dim}")
        print(f"  Max action: {max_action}")
        print(f"  Layers: {len(layers)}")
        for i, layer in enumerate(layers):
            out_dim, in_dim = layer['weight'].shape
            print(f"    Layer {i}: {in_dim} -> {out_dim} ({layer['activation']})")
        
        # Generate C++ header
        print(f"Generating C++ header...")
        cpp_header = generate_cpp_header(layers, config, max_action, state_dim, action_dim)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Auto-generate filename based on model path
            model_name = os.path.splitext(os.path.basename(args.model))[0]  # Remove .pt extension
            model_dir = os.path.basename(os.path.dirname(args.model))  # Get parent directory name
            filename = f"td3_actor_{model_dir}_{model_name}.h"
            
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            exports_dir = os.path.join(script_dir, "exports")
            
            # Create exports directory if it doesn't exist
            os.makedirs(exports_dir, exist_ok=True)
            
            output_path = os.path.join(exports_dir, filename)
        
        # Write the header file
        with open(output_path, 'w') as f:
            f.write(cpp_header)
        
        print(f"C++ header written to: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())