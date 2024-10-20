
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.sklearn
import os
import re

# Set up MLflow
mlflow.set_experiment("Insurance Model Experiment")

# Ensure models directory exists

# Class for data exploration
class DataExplorer:
    def __init__(self, data):
        self.data = data

    def explore_data(self):
        print(self.data.head().T)
        print(self.data.describe())
        print(self.data.info())
        return self

    def plot_histograms(self):
        self.data.hist(bins=20, figsize=(20, 15))
        plt.suptitle('Histograms of Numerical Features')
        plt.show()
        return self

    def plot_distribution(self, target_column):
        plt.figure(figsize=(8, 4))
        sns.countplot(x=target_column, data=self.data, palette='bright')
        plt.title(f'Distribution of {target_column}')
        plt.show()
        return self

    def plot_correlation_matrix(self, first_n_columns=43):
        subset_corr = self.data.iloc[:, :first_n_columns].join(self.data['CARAVAN']).corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(subset_corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(f'Correlation Matrix for First {first_n_columns} Columns and CARAVAN')
        plt.show()
        return self

# Class for handling data preprocessing, modeling, and evaluation
class InsuranceModel:
    def __init__(self, train_path, eval_path, target_path):
        self.train_path = train_path
        self.eval_path = eval_path
        self.target_path = target_path
        self.train_data = None
        self.eval_data = None
        self.target_data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        # Load training, evaluation, and target data
        self.train_data = pd.read_csv(self.train_path, sep='\t', header=None)
        self.eval_data = pd.read_csv(self.eval_path, sep='\t', header=None)
        self.target_data = pd.read_csv(self.target_path, sep='\t', header=None, names=['Target'])

        # Load and apply dictionary.txt for column names
        with open('../docs/insurance+company+benchmark+coil+2000/dictionary.txt', 'r', encoding='ISO-8859-1') as file:
            file_content = file.read()

        # Extract the Data Dictionary table using updated regular expressions
        pattern = re.compile(r"(\d+)\s+([A-Z]+[A-Z0-9]*)\s+(.+)")
        matches = pattern.findall(file_content)

        # Create a DataFrame from the matches
        df = pd.DataFrame(matches, columns=['Nr', 'Name', 'Description'])

        # Use the column names as they are for train_data
        column_names = df['Name'].tolist()
        self.train_data.columns = column_names

        # Assign columns for eval_data (without 'CARAVAN')
        self.eval_data.columns = column_names[:-1]  # Exclude CARAVAN for eval_data

        return self

    def preprocess_data(self):
        # Assuming 'CARAVAN' is the target column
        X = self.train_data.drop(columns=['CARAVAN'], errors='ignore')  # Adjust to avoid KeyError
        y = self.train_data['CARAVAN']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return self

    def handle_imbalance(self):
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
        return self

    def train_model(self):
        # Start MLflow logging
        with mlflow.start_run():
            model = XGBClassifier()
            model.fit(self.X_train, self.y_train)
            self.model = model

            # Log parameters
            mlflow.log_param("model_type", "XGBoost")
            
            # Evaluate model
            predictions = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            recall = recall_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Save the model with MLflow
            mlflow.sklearn.log_model(self.model, "model")

            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1 Score: {f1_score(self.y_test, predictions)}')

        return self

    def save_model(self):
        # Also save the model to the 'models' directory for DVC version control
        model_path = '../models/insurance_model.pkl'
        joblib.dump(self.model, model_path)
        print(f'Model saved successfully in {model_path}')
        return self

def main():
    train_path = '../data/raw/train/ticdata2000.txt'
    eval_path = '../data/raw/eval/ticeval2000.txt'
    target_path = '../data/raw/eval/tictgts2000.txt'

    model = InsuranceModel(train_path, eval_path, target_path)
    
    # Chain the methods for loading data, preprocessing, training, and evaluation
    (model.load_data()
         .preprocess_data()
         .handle_imbalance()
         .train_model()
         .save_model())

    # Perform EDA with a separate dataset copy
    eda = DataExplorer(model.train_data)
    eda.explore_data().plot_histograms().plot_distribution('CARAVAN').plot_correlation_matrix()

if __name__ == "__main__":
    main()
