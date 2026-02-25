import argparse
import sys
from pathlib import Path

# Add project root to path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.map_utils import visualize_track_lines

def main():
    parser = argparse.ArgumentParser(description="Visualize the image processing pipeline that builds the track environment from a raw mask.")
    parser.add_argument("--map", type=str, default="assets/maps/map_start3.png", help="Path to the map start image (PNG)")
    args = parser.parse_args()
    
    map_path = Path(args.map)
    if not map_path.exists():
        print(f"Error: Map file not found at {map_path}")
        return
        
    print(f"Processing and visualizing map: {map_path}")
    print("This will open a matplotlib window showing raw track analysis alongside the stylized renderer.")
    visualize_track_lines(str(map_path))

if __name__ == "__main__":
    main()
