# Fake Job Posting Detection using NLP and Linear SVM

## Project Overview
This project builds a machine learning model to detect fraudulent job postings using Natural Language Processing (NLP). Text data is transformed using TF-IDF and classified using Linear Support Vector Machines (SVM).

## Business Problem
Online job platforms often contain fraudulent postings. Missing these (false negatives) can harm users, while excessive false positives can reduce trust. The goal is to balance recall and precision.

## Dataset
The dataset contains job postings labeled as real or fake, with multiple text-based features such as title, description, and requirements.

## Methodology
- Text preprocessing and cleaning  
- TF-IDF feature extraction  
- Logistic Regression baseline  
- Linear SVM modeling  
- Hyperparameter tuning (GridSearchCV)  
- Threshold analysis  
- Model interpretation  

## Results
- Precision (Fraud): ~0.82  
- Recall (Fraud): ~0.84  
- F1 Score: ~0.83  
- ROC-AUC: ~0.98  

## Final Model
The final model is a tuned Linear SVM with balanced performance between precision and recall.

## Key Insights
- Fraudulent jobs often contain vague, money-oriented language  
- Real jobs contain structured, technical, and role-specific terms  

## How to Run

1. Install dependencies:
    pip install -r requirements.txt

2. Launch Jupyter Notebook:
    jupyter notebook

3. Open the file:
    fake-job-posting-detection-nlp-linear-svm.ipynb

4. Run all cells to reproduce results

## Repository Structure

fake-job-posting-detection-nlp-svm/
│
├── fake-job-posting-detection-nlp-linear-svm.ipynb
├── README.md
├── requirements.txt
└── images/

## Future Improvements
- Incorporate structured metadata features  
- Experiment with advanced models such as XGBoost or transformer-based NLP models  
- Deploy the model using a web application such as Streamlit
