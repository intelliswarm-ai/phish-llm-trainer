#!/usr/bin/env python
"""Evaluate the trained model on multiple test datasets"""
import os
import sys
import json
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Disable OpenTelemetry
os.environ['OTEL_SDK_DISABLED'] = 'true'

from src.evaluate import load_model, Evaluator
from src.data_preprocessing import DataPreprocessor, PhishingEmailDataset
from torch.utils.data import DataLoader


def prepare_test_data(df: pd.DataFrame, preprocessor: DataPreprocessor) -> pd.DataFrame:
    """Prepare test data in the format expected by the model"""
    # Combine subject and body into text_combined
    df['text_combined'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    
    # Clean the text
    df['text_combined'] = df['text_combined'].apply(preprocessor.clean_text)
    
    # Ensure label is integer
    df['label'] = df['label'].astype(int)
    
    # Keep only required columns
    return df[['text_combined', 'label']].rename(columns={'text_combined': 'text'})


def evaluate_single_dataset(
    model_path: str,
    dataset_path: str,
    dataset_name: str,
    preprocessor: DataPreprocessor,
    device: torch.device,
    output_dir: str
) -> Dict:
    """Evaluate model on a single test dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating on: {dataset_name}")
    print(f"{'='*60}")
    
    # Load the dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} samples from {dataset_name}")
        
        # Check label distribution
        label_counts = df['label'].value_counts().sort_index()
        print(f"Label distribution: {dict(label_counts)}")
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None
    
    # Prepare the data
    test_df = prepare_test_data(df, preprocessor)
    
    # Create dataset and dataloader
    test_dataset = PhishingEmailDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        preprocessor.tokenizer,
        max_length=512
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = load_model(model_path, device)
    evaluator = Evaluator(model, device)
    
    # Create output directory for this dataset
    dataset_output_dir = os.path.join(output_dir, dataset_name.replace('.csv', ''))
    
    # Evaluate
    try:
        metrics = evaluator.evaluate(test_loader, save_path=dataset_output_dir)
        
        print(f"\nResults for {dataset_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        
        return {
            'dataset': dataset_name,
            'samples': len(df),
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"Error evaluating {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model on multiple test datasets')
    parser.add_argument('--checkpoint', type=str, 
                       default='models/phishing_detector_quick_20250709_082040/best_model_epoch_1/checkpoint.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='test-datasets',
                       help='Directory containing test datasets')
    parser.add_argument('--output_dir', type=str, default='./test_evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    preprocessor = DataPreprocessor()
    
    # Get all CSV files in test directory
    test_datasets = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.csv')])
    print(f"\nFound {len(test_datasets)} test datasets: {test_datasets}")
    
    # Evaluate on each dataset
    all_results = []
    
    for dataset_file in test_datasets:
        dataset_path = os.path.join(args.test_dir, dataset_file)
        result = evaluate_single_dataset(
            args.checkpoint,
            dataset_path,
            dataset_file,
            preprocessor,
            device,
            args.output_dir
        )
        
        if result:
            all_results.append(result)
    
    # Save summary results
    summary_path = os.path.join(args.output_dir, 'summary_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"\n{result['dataset']}:")
        print(f"  Samples: {result['samples']}")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  F1 Score: {result['metrics']['f1']:.4f}")
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()