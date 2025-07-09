# Model Evaluation Guide

## Prerequisites

Before running the evaluation script, ensure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Evaluation

To evaluate a trained model, use the following command format:

```bash
python evaluate_wrapper.py --checkpoint models/[your-model-dir]/checkpoint.pt
```

### Example

For the model trained on 2025-07-09:

```bash
python evaluate_wrapper.py --checkpoint models/phishing_detector_quick_20250709_082040/best_model_epoch_1/checkpoint.pt
```

### Optional Arguments

- `--data_path`: Path to the dataset CSV file (default: `datasets/phishing_email.csv`)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--output_dir`: Directory to save evaluation results (default: `./evaluation_results`)

### Full Example with All Options

```bash
python evaluate_wrapper.py \
    --checkpoint models/phishing_detector_quick_20250709_082040/best_model_epoch_1/checkpoint.pt \
    --data_path datasets/phishing_email.csv \
    --batch_size 32 \
    --output_dir ./evaluation_results
```

## Output

The evaluation script will generate the following metrics and files:

1. **Console Output**:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - AUC-ROC

2. **Saved Files** (in the output directory):
   - `metrics.json`: All metrics in JSON format
   - `classification_report.txt`: Detailed classification report
   - `confusion_matrix.png`: Visualization of the confusion matrix
   - `roc_curve.png`: ROC curve visualization

## Troubleshooting

If you encounter a "ModuleNotFoundError", ensure:
1. You have activated your Python virtual environment (if using one)
2. All dependencies are installed: `pip install -r requirements.txt`
3. You are running the script from the project root directory

If the checkpoint file is not found, verify:
1. The checkpoint path is correct
2. The checkpoint.pt file exists in the specified directory