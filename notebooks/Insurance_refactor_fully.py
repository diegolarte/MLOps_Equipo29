from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn
import os
import re
from mlflow.models.signature import infer_signature

# Set up MLflow
mlflow.set_experiment("Insurance Model Experiment")

# Ensure models directory exists
if not os.path.exists('../models'):
    os.makedirs('../models')

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
        self.xgb_model = None
        self.dt_model = None
        self.lr_model = None

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

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return self

    def handle_imbalance(self):
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
        return self

    def train_model(self):
        # Start MLflow logging
        with mlflow.start_run():
            # Modified XGBoost model with hyperparameters
            self.xgb_model = XGBClassifier(
                n_estimators=150,          # number of trees
                learning_rate=0.7,         # step size shrinkage
                max_depth=4,               # maximum depth of a tree
                subsample=0.9,             # percentage of samples used per tree
                colsample_bytree=0.4,      # percentage of features used per tree
                random_state=42
            )
            self.xgb_model.fit(self.X_train, self.y_train)
            # Decision Tree model with hyperparameters
            self.dt_model = DecisionTreeClassifier(
                max_depth=5, 
                min_samples_split=7, 
                random_state=42
            )
            self.dt_model.fit(self.X_train, self.y_train)

            # Logistic Regression model with hyperparameters
            self.lr_model = LogisticRegression(
                max_iter=3444, 
                C=0.7, 
                solver='newton-cg', 
                random_state=42
            )
            self.lr_model.fit(self.X_train, self.y_train)

            # Log parameters
            mlflow.log_param("xgboost_model_type", "XGBoost")
            mlflow.log_param("decision_tree_model_type", "Decision Tree")
            mlflow.log_param("logistic_regression_model_type", "Logistic Regression")

            # Evaluate XGBoost
            xgb_predictions = self.xgb_model.predict(self.X_test)
            xgb_accuracy = accuracy_score(self.y_test, xgb_predictions)
            xgb_precision = precision_score(self.y_test, xgb_predictions)
            xgb_recall = recall_score(self.y_test, xgb_predictions)
            xgb_f1 = f1_score(self.y_test, xgb_predictions)

            # Evaluate Decision Tree
            dt_predictions = self.dt_model.predict(self.X_test)
            dt_accuracy = accuracy_score(self.y_test, dt_predictions)
            dt_precision = precision_score(self.y_test, dt_predictions)
            dt_recall = recall_score(self.y_test, dt_predictions)
            dt_f1 = f1_score(self.y_test, dt_predictions)

            # Evaluate Logistic Regression
            lr_predictions = self.lr_model.predict(self.X_test)
            lr_accuracy = accuracy_score(self.y_test, lr_predictions)
            lr_precision = precision_score(self.y_test, lr_predictions)
            lr_recall = recall_score(self.y_test, lr_predictions)
            lr_f1 = f1_score(self.y_test, lr_predictions)

            # Print the evaluation results
            print(f"XGBoost - Accuracy: {xgb_accuracy}, Precision: {xgb_precision}, Recall: {xgb_recall}, F1 Score: {xgb_f1}")
            print(f"Decision Tree - Accuracy: {dt_accuracy}, Precision: {dt_precision}, Recall: {dt_recall}, F1 Score: {dt_f1}")
            print(f"Logistic Regression - Accuracy: {lr_accuracy}, Precision: {lr_precision}, Recall: {lr_recall}, F1 Score: {lr_f1}")

            # Log metrics for XGBoost
            mlflow.log_metric("xgb_accuracy", xgb_accuracy)
            mlflow.log_metric("xgb_precision", xgb_precision)
            mlflow.log_metric("xgb_recall", xgb_recall)
            mlflow.log_metric("xgb_f1", xgb_f1)

            # Log metrics for Decision Tree
            mlflow.log_metric("dt_accuracy", dt_accuracy)
            mlflow.log_metric("dt_precision", dt_precision)
            mlflow.log_metric("dt_recall", dt_recall)
            mlflow.log_metric("dt_f1", dt_f1)

            # Log metrics for Logistic Regression
            mlflow.log_metric("lr_accuracy", lr_accuracy)
            mlflow.log_metric("lr_precision", lr_precision)
            mlflow.log_metric("lr_recall", lr_recall)
            mlflow.log_metric("lr_f1", lr_f1)

            # Generate an input example from the test data
            input_example = self.X_test[:5]

            # Infer signatures for the models
            xgb_signature = infer_signature(self.X_test, self.xgb_model.predict(self.X_test))
            dt_signature = infer_signature(self.X_test, self.dt_model.predict(self.X_test))
            lr_signature = infer_signature(self.X_test, self.lr_model.predict(self.X_test))


            # Log models with signature, input example, and explicit pip requirements
            mlflow.sklearn.log_model(sk_model=self.xgb_model,artifact_path= "xgboost_model",  signature=xgb_signature, registered_model_name="xgboost_model")
            mlflow.sklearn.log_model(sk_model=self.dt_model, artifact_path="decision_tree_model", signature=dt_signature, registered_model_name="decision_tree_model")
            mlflow.sklearn.log_model(sk_model=self.lr_model, artifact_path="logistic_regression_model",  signature=lr_signature, registered_model_name="logistic_regression_model")

        return self

    def save_model(self):
        # Save each model separately
        xgb_model_path = '../models/xgb_model.pkl'
        dt_model_path = '../models/dt_model.pkl'
        lr_model_path = '../models/lr_model.pkl'

        joblib.dump(self.xgb_model, xgb_model_path)
        joblib.dump(self.dt_model, dt_model_path)
        joblib.dump(self.lr_model, lr_model_path)

        print(f'XGBoost model saved successfully in {xgb_model_path}')
        print(f'Decision Tree model saved successfully in {dt_model_path}')
        print(f'Logistic Regression model saved successfully in {lr_model_path}')
        
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
