#!/usr/bin/env python
"""Wrapper script to run evaluation on all test datasets"""
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
    """Main function to run evaluation on test datasets"""
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Import after dependency check
    try:
        from evaluate_test_datasets import main as evaluate_main
        evaluate_main()
    except Exception as e:
        print(f"Error running evaluation: {e}")
        print("\nUsage:")
        print("python evaluate_test_datasets_wrapper.py")
        print("\nOptional arguments:")
        print("  --checkpoint: Path to model checkpoint")
        print("  --test_dir: Directory containing test datasets (default: test-datasets)")
        print("  --output_dir: Directory to save results (default: ./test_evaluation_results)")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()