"""Test script for tracer visualization."""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.visualization import TracerVisualizer


def main():
    """Main entry point for the visualization server."""
    parser = argparse.ArgumentParser(description='Run tracer visualization server')
    parser.add_argument(
        '--tracer_json',
        default=os.path.join(root, 'workdir', 'online_trading_agent', 'tracer.json'),
        type=str,
        help='Path to tracer.json file'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8088,
        help='Port number for the web server (default: 8080)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    args = parser.parse_args()
    
    tracer_path = Path(args.tracer_json)
    if not tracer_path.exists():
        print(f"Error: Tracer JSON file not found: {tracer_path}")
        sys.exit(1)
    
    try:
        visualizer = TracerVisualizer(str(tracer_path), port=args.port)
        visualizer.run(debug=args.debug)
    except Exception as e:
        print(f"Error starting visualization server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

