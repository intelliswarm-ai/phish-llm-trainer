# Test Dataset Evaluation Results

This document presents the evaluation results of our DistilBERT phishing detection model on various external test datasets. The model was tested on five different phishing email datasets to assess its generalization capabilities.

## Test Datasets Overview

| Dataset | Total Samples | Legitimate Emails | Phishing Emails |
|---------|---------------|-------------------|-----------------|
| CEAS_08 | 39,154 | 17,312 | 21,842 |
| Enron | 29,767 | 15,791 | 13,976 |
| Ling | 2,859 | 2,401 | 458 |
| Nazario | 1,565 | 0 | 1,565 |
| Nigerian Fraud | 3,332 | 0 | 3,332 |

## Performance Summary

### CEAS_08 Dataset
- **Accuracy**: 98.78%
- **Precision**: 98.96%
- **Recall**: 98.86%
- **F1 Score**: 98.91%
- **AUC-ROC**: 99.83%

The model achieved exceptional performance on the CEAS_08 dataset with balanced detection rates for both legitimate and phishing emails.

### Enron Dataset
- **Accuracy**: 96.88%
- **Precision**: 99.52%
- **Recall**: 93.80%
- **F1 Score**: 96.57%
- **AUC-ROC**: 99.82%

Strong performance with very high precision, indicating few false positives. The slightly lower recall suggests some phishing emails were missed.

### Ling Dataset
- **Accuracy**: 96.15%
- **Precision**: 100.00%
- **Recall**: 75.98%
- **F1 Score**: 86.35%
- **AUC-ROC**: 99.80%

Perfect precision (no false positives) but lower recall, indicating a conservative approach that may miss some phishing emails.

### Nazario Dataset (Phishing-only)
- **Accuracy**: 87.86%
- **Precision**: 100.00%
- **Recall**: 87.86%
- **F1 Score**: 93.54%
- **AUC-ROC**: N/A (single class)

This dataset contains only phishing emails. The model correctly identified 87.86% of them as phishing.

### Nigerian Fraud Dataset (Phishing-only)
- **Accuracy**: 91.87%
- **Precision**: 100.00%
- **Recall**: 91.87%
- **F1 Score**: 95.76%
- **AUC-ROC**: N/A (single class)

Another phishing-only dataset. The model successfully detected 91.87% of Nigerian fraud emails as phishing.

## Key Findings

1. **Excellent Generalization**: The model demonstrates strong performance across diverse datasets, maintaining high accuracy (87.86% - 98.78%) on all test sets.

2. **High Precision**: The model shows consistently high precision across all datasets, with perfect precision (100%) on three datasets, indicating minimal false positives.

3. **Variable Recall**: Recall varies more significantly (75.98% - 98.86%), suggesting the model may be more conservative on certain types of phishing emails.

4. **Specialized Phishing Detection**: The model performs well on specialized phishing datasets (Nazario, Nigerian Fraud), correctly identifying the majority of sophisticated phishing attempts.

## Confusion Matrices Summary

### CEAS_08
- True Negatives: 17,084 (98.68% of legitimate emails correctly classified)
- False Positives: 228 (1.32% of legitimate emails misclassified)
- False Negatives: 248 (1.14% of phishing emails misclassified)
- True Positives: 21,594 (98.86% of phishing emails correctly classified)

### Enron
- True Negatives: 15,728 (99.60% of legitimate emails correctly classified)
- False Positives: 63 (0.40% of legitimate emails misclassified)
- False Negatives: 867 (6.20% of phishing emails misclassified)
- True Positives: 13,109 (93.80% of phishing emails correctly classified)

### Ling
- True Negatives: 2,401 (100% of legitimate emails correctly classified)
- False Positives: 0 (0% of legitimate emails misclassified)
- False Negatives: 110 (24.02% of phishing emails misclassified)
- True Positives: 348 (75.98% of phishing emails correctly classified)

## Recommendations

1. **Production Deployment**: The model shows strong generalization and is suitable for production deployment with high confidence.

2. **Threshold Tuning**: For applications requiring higher recall (catching more phishing emails), consider adjusting the classification threshold, especially for datasets similar to Ling.

3. **Continuous Monitoring**: While performance is excellent, continue monitoring for new phishing patterns, particularly those that might be missed (as indicated by variable recall rates).

4. **False Positive Management**: The low false positive rates make this model suitable for automated filtering with minimal legitimate email disruption.

## Test Execution

To reproduce these results, use the evaluation script:

```bash
python evaluate_test_datasets.py --checkpoint models/[your-model]/checkpoint.pt
```

This will evaluate the model on all test datasets and generate detailed metrics and visualizations in the `test_evaluation_results/` directory.