import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm

from .model import DistilBertForPhishingDetection
from .data_preprocessing import DataPreprocessor


class Evaluator:
    def __init__(self, model: DistilBertForPhishingDetection, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def predict(self, dataloader) -> Tuple[List[int], List[int], List[float]]:
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs['logits'], dim=-1)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return all_predictions, all_labels, all_probs
    
    def evaluate(self, dataloader, save_path: Optional[str] = None) -> Dict[str, float]:
        if dataloader is None:
            raise ValueError("Dataloader is None. Please check data loading.")
            
        predictions, labels, probs = self.predict(dataloader)
        
        if not predictions or not labels or not probs:
            raise ValueError("No predictions found. Please check if the dataloader contains data.")
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        auc_score = roc_auc_score(labels, probs)
        
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc_score),
            'confusion_matrix': cm.tolist(),
            'support': {
                'legitimate': int(np.sum(np.array(labels) == 0)),
                'phishing': int(np.sum(np.array(labels) == 1))
            }
        }
        
        if save_path:
            self.save_results(metrics, predictions, labels, probs, save_path)
        
        return metrics
    
    def save_results(self, metrics: Dict, predictions: List[int], 
                    labels: List[int], probs: List[float], save_path: str):
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.plot_confusion_matrix(metrics['confusion_matrix'], save_path)
        self.plot_roc_curve(labels, probs, save_path)
        
        report = classification_report(labels, predictions, 
                                     target_names=['Legitimate', 'Phishing'])
        with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to {save_path}")
        print(f"\nClassification Report:")
        print(report)
    
    def plot_confusion_matrix(self, cm: List[List[int]], save_path: str):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()
    
    def plot_roc_curve(self, labels: List[int], probs: List[float], save_path: str):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'roc_curve.png'))
        plt.close()


def load_model(checkpoint_path: str, device: torch.device) -> DistilBertForPhishingDetection:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = DistilBertForPhishingDetection(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        dropout_rate=config['dropout_rate']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate phishing email detection model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, 
                       default='datasets/phishing_email.csv',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        print(f"Loading model from: {args.checkpoint}")
        model = load_model(args.checkpoint, device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    preprocessor = DataPreprocessor()
    
    try:
        print(f"Loading data from: {args.data_path}")
        train_df, val_df, test_df = preprocessor.load_and_preprocess_data(args.data_path)
        print(f"Data loaded: Test samples = {len(test_df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        _, _, test_loader = preprocessor.create_data_loaders(
            train_df, val_df, test_df,
            batch_size=args.batch_size
        )
        print(f"Data loaders created. Test batches = {len(test_loader)}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    evaluator = Evaluator(model, device)
    
    # Create output directory early to ensure it exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created/verified: {args.output_dir}")
    
    print("Evaluating on test set...")
    try:
        metrics = evaluator.evaluate(test_loader, save_path=args.output_dir)
        
        print(f"\nTest Set Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        error_log_path = os.path.join(args.output_dir, 'evaluation_error.txt')
        with open(error_log_path, 'w') as f:
            f.write(f"Error during evaluation: {str(e)}\n\n")
            f.write("Full traceback:\n")
            traceback.print_exc(file=f)
        print(f"Error log saved to: {error_log_path}")


if __name__ == "__main__":
    main()