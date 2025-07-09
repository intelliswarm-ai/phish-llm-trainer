#!/usr/bin/env python
"""Wrapper script to run model evaluation from project root"""
import os
import sys
import subprocess

# Disable OpenTelemetry if installed
os.environ['OTEL_SDK_DISABLED'] = 'true'

# Add project root to path to enable proper imports
sys.path.insert(0, os.path.dirname(__file__))

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import transformers
        import sklearn
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import tqdm
        return True
    except ImportError as e:
        print(f"Error: Missing required dependencies - {e}")
        print("\nPlease install the required dependencies by running:")
        print("pip install -r requirements.txt")
        return False

def main():
    """Main function to run evaluation"""
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Import after dependency check
    try:
        from src.evaluate import main as evaluate_main
        evaluate_main()
    except Exception as e:
        print(f"Error running evaluation: {e}")
        print("\nUsage example:")
        print("python evaluate_wrapper.py --checkpoint models/[your-model-dir]/checkpoint.pt")
        print("\nOptional arguments:")
        print("  --data_path: Path to dataset (default: datasets/phishing_email.csv)")
        print("  --batch_size: Batch size for evaluation (default: 32)")
        print("  --output_dir: Directory to save results (default: ./evaluation_results)")
        sys.exit(1)

if __name__ == "__main__":
    main()