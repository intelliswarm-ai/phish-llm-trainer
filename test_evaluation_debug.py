#!/usr/bin/env python
"""Debug script to test evaluation step by step"""
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Disable OpenTelemetry
os.environ['OTEL_SDK_DISABLED'] = 'true'

from src.evaluate import load_model
from src.data_preprocessing import DataPreprocessor

def main():
    checkpoint_path = "models/phishing_detector_quick_20250709_082040/best_model_epoch_1/checkpoint.pt"
    data_path = "datasets/phishing_email.csv"
    
    print("Step 1: Checking files exist...")
    print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
    print(f"Data file exists: {os.path.exists(data_path)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStep 2: Using device: {device}")
    
    print("\nStep 3: Loading model...")
    try:
        model = load_model(checkpoint_path, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\nStep 4: Loading data...")
    preprocessor = DataPreprocessor()
    try:
        train_df, val_df, test_df = preprocessor.load_and_preprocess_data(data_path)
        print(f"Data loaded successfully!")
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Check if test_df has data
        if len(test_df) == 0:
            print("ERROR: Test dataset is empty!")
            return
            
        # Check columns
        print(f"Test df columns: {test_df.columns.tolist()}")
        print(f"First few rows of test_df:")
        print(test_df.head())
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nStep 5: Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = preprocessor.create_data_loaders(
            train_df, val_df, test_df,
            batch_size=32
        )
        print(f"Data loaders created successfully!")
        print(f"Test loader batches: {len(test_loader)}")
        
        # Test getting one batch
        print("\nTesting first batch...")
        for batch in test_loader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            break
            
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nDebug complete! All steps passed.")

if __name__ == "__main__":
    main()