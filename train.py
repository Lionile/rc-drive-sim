#!/usr/bin/env python3
"""
Training script for RL controllers.
"""

import argparse
import yaml
import random
import numpy as np
import torch
import os
from datetime import datetime
from pathlib import Path

from simulation.environment import Environment


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_env(map_id, render=False):
    """Build environment for training."""
    map_path = f"maps/map_start{map_id}.png"
    env = Environment(
        map_path=map_path,
        show_collision_box=render,  # Show collision box if rendering
        show_sensors=render,        # Show sensors if rendering
        show_racing_line=False,
        show_track_edges=False,
        headless=not render  # Run headless when not rendering
    )
    return env


def main():
    parser = argparse.ArgumentParser(description="Train RL controllers")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--episodes", type=int, help="Number of episodes to train")
    parser.add_argument("--total-steps", type=int, help="Total steps to train")
    parser.add_argument("--save-dir", default="models", help="Directory to save models")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.episodes:
        config['episodes'] = args.episodes
    if args.total_steps:
        config['total_steps'] = args.total_steps
        config['episodes'] = None  # Use steps instead of episodes
    
    # Set seeds
    set_seeds(config['seed'])
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detect algorithm from config path
    config_name = Path(args.config).stem
    algo = config_name  # e.g., 'td3' from 'td3.yaml'
    
    save_dir = Path(args.save_dir) / algo / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training {algo.upper()} controller")
    print(f"Config: {args.config}")
    print(f"Save directory: {save_dir}")
    print(f"Seed: {config['seed']}")
    if config.get('episodes'):
        print(f"Episodes: {config['episodes']}")
    if config.get('total_steps'):
        print(f"Total steps: {config['total_steps']}")
    if args.render:
        print(f"Rendering: ON")
        print("Training Controls:")
        print("  S - Toggle sensors")
        print("  C - Toggle collision box")
        print("  T - Toggle track edges")
        print("  R - Toggle racing line")
        print("  D - Toggle distance heatmap")
        print("  H - Toggle rendering")
    else:
        print(f"Rendering: OFF (use --render to enable)")
    print("-" * 50)
    
    # Build environment
    env = build_env(config['map'], render=args.render)
    
    # Import and create trainer based on algorithm
    if algo == 'td3':
        from controllers.td3_controller import TD3Trainer
        trainer = TD3Trainer(env, config, save_dir, render=args.render)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Save resolved config
    config_save_path = save_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Start training
    try:
        trainer.train()
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Model saved to: {save_dir}")
    except KeyboardInterrupt:
        print(f"\n⚠ Training interrupted by user")
        print(f"✓ Partial results saved to: {save_dir}")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
