import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

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
        self.best_model = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path, sep='\t', header=None)
        self.eval_data = pd.read_csv(self.eval_path, sep='\t', header=None)
        self.target_data = pd.read_csv(self.target_path, sep='\t', header=None, names=['Target'])
        return self

    def preprocess_data(self):
        # Preprocessing: Define features and target
        X = self.train_data[['MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD', 'MGODRK', 'MGODPR',
                             'MGODOV', 'MGODGE', 'MRELGE', 'MRELSA', 'MRELOV', 'MFALLEEN',
                             'MFGEKIND', 'MFWEKIND', 'MOPLHOOG', 'MOPLMIDD', 'MOPLLAAG',
                             'MBERHOOG', 'MBERZELF', 'MBERMIDD', 'holds_policy']]
        y = self.train_data['CARAVAN']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        return self

    def handle_imbalance(self):
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        return self

    def train_model(self):
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__gamma': [0, 0.1, 0.2],
            'model__scale_pos_weight': [1, 3, 5]
        }
        
        pipeline = Pipeline(steps=[('smote', SMOTE(random_state=42)), ('model', XGBClassifier(random_state=42))])
        
        random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=50,
                                           scoring='roc_auc', cv=5, verbose=1, random_state=42, n_jobs=-1)
        random_search.fit(self.X_train, self.y_train)
        self.best_model = random_search.best_estimator_
        print("Best parameters found: ", random_search.best_params_)
        return self

    def evaluate_model(self):
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        return self

    def save_model(self, filename='xgboost_model.pkl'):
        joblib.dump(self.best_model, filename)
        print(f"Model saved as {filename}")
        return self

# Main execution function
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
         .evaluate_model()
         .save_model())

    # Perform EDA with a separate dataset copy
    eda = DataExplorer(model.train_data)
    eda.explore_data().plot_histograms().plot_distribution('CARAVAN').plot_correlation_matrix()

if __name__ == "__main__":
    main()
