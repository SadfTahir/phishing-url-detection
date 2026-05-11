# Phishing URL Detection using Machine Learning

## Information Security Project — Spring 2026

## Overview
This project detects phishing URLs using a Random Forest classifier trained on 11,430 URLs with 23 selected features.

## Results
- **Accuracy: 96.02%**
- Phishing Recall: 97%
- Legitimate Precision: 97%

## Dataset
[Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)
- 11,430 URLs (balanced: 50% phishing, 50% legitimate)
- 87 features → 23 selected

## How to Run
1. Download dataset from Kaggle
2. Open `phishing_detection.py` in Google Colab
3. Upload `dataset_phishing.csv`
4. Run all cells

## Tech Stack
- Python, scikit-learn, pandas, matplotlib, seaborn
- Algorithm: Random Forest (100 trees)

## Files
- `phishing_detection.py` — Main project code
- `confusion_matrix.png` — Results chart
- `feature_importance.png` — Feature analysis chart
