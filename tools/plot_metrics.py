import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(csv_path, output_dir=None):
    """
    Load TD3 training metrics from a CSV and plot Reward and Average Speed over time.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Could not find metrics file at {csv_file}")
        return

    # Load data
    df = pd.read_csv(csv_file)
    if 'episode' not in df.columns:
        print("Error: CSV does not contain an 'episode' column.")
        return

    # Set up the output directory
    if output_dir is None:
        out_path = csv_file.parent
    else:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    # Apply rolling average for smoother visualization
    window_size = min(50, max(1, len(df) // 20))  # Dynamic rolling window based on data size
    df['rolling_reward'] = df['total_reward'].rolling(window=window_size, min_periods=1).mean()
    df['rolling_speed'] = df['avg_speed'].rolling(window=window_size, min_periods=1).mean()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('TD3 Training Metrics', fontsize=16, fontweight='bold')

    # Plot 1: Total Reward
    ax1.plot(df['episode'], df['total_reward'], alpha=0.3, color='blue', label='Raw Reward')
    ax1.plot(df['episode'], df['rolling_reward'], color='darkblue', linewidth=2, label=f'Rolling Avg (n={window_size})')
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_title('Episode Reward over Time')

    # Plot 2: Average Speed
    ax2.plot(df['episode'], df['avg_speed'], alpha=0.3, color='orange', label='Raw Speed')
    ax2.plot(df['episode'], df['rolling_speed'], color='darkorange', linewidth=2, label=f'Rolling Avg (n={window_size})')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Speed (px/s)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_title('Average Speed over Time')

    plt.tight_layout()
    
    # Save output
    output_filename = out_path / f"training_metrics_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved high-resolution training plot to: {output_filename}")
    
    # Optional: Display plot if running interactively
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot TD3 training metrics from a CSV file.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the metrics.csv file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for the plot image (defaults to CSV folder)")
    args = parser.parse_args()

    plot_metrics(args.csv, args.output)
