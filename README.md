## Project Overview 
A machine learning project to predict autism spectrum disorder based on input features. This repository contains data, notebooks, trained models, and supporting scripts for building, evaluating, and deploying the classifier.

## Repository Structure
```bash
Autism-Detection/
├── train.csv
├── test.csv
├── AutismDetection.ipynb
├── best_model.pkl
├── requirements.txt
└── README.md



## 🧠 Project Overview

This project aims to develop a machine learning model that can detect autism (or likelihood thereof) given input variables (demographic, clinical, behavioral features). The pipeline involves:

1. **Data loading & exploration**  
2. **Data preprocessing / feature engineering**  
3. **Model training & hyperparameter tuning**  
4. **Evaluation** (accuracy, recall, precision, confusion matrix)  
5. **Serialization** of the best model + encoders  
6. Optionally, **inference / deployment**

## Prerequisites

- Python 3.7+  
- Required libraries (you can install via `requirements.txt`)  
  ```text
  numpy
  pandas
  scikit-learn
  matplotlib / seaborn
  joblib / pickle
  jupyter

## Steps to run
```bash
# Clone the repo
   git clone https://github.com/mehershiri/Autism-Detection.git
   cd Autism-Detection
# Install dependecies if needed
   pip install -r requirements.txt
# Start Jupyter Lab/ Notebook using the following commands: 
   jupyter notebook
   or
   jupyter lab
# Open and run AutismDetection.ipynb step by step.
   -The notebook loads train.csv and test.csv
   -It preprocesses features, encodes categorical variables
   -Trains multiple models, chooses the best one
   -Saves the best model & encoders (best_model.pkl, encoders.pkl)
   -Evaluates performance on the test set
