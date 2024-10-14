import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

# Class for EDA tasks
class DataExplorer:
    def __init__(self, data):
        self.data = data
    
    def explore_data(self):
        print(self.data.head().T)
        print(self.data.describe())
        print(self.data.info())
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

    def plot_histograms(self):
        self.data.hist(bins=20, figsize=(20, 15))
        plt.suptitle('Histograms of Numerical Features')
        plt.show()
        return self

# Class for handling and modeling insurance data
class InsuranceModel:
    def __init__(self, train_path, eval_path, target_path):
        self.train_path = train_path
        self.eval_path = eval_path
        self.target_path = target_path
        self.model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('reduce_dim', TruncatedSVD(n_components=50)),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.train_data = None
        self.eda_data = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path, sep='\t', header=None)
        self.eval_data = pd.read_csv(self.eval_path, sep='\t', header=None)
        self.target_data = pd.read_csv(self.target_path, sep='\t', header=None, names=['Target'])
        # Copy for EDA
        self.eda_data = self.train_data.copy()
        return self

    def rename_columns(self, new_column_names):
        self.train_data.columns = new_column_names
        self.eda_data.columns = new_column_names  # Sync columns for EDA
        return self

    def preprocess_data(self):
        X = self.train_data.drop(columns=[85])  # Assuming CARAVAN is in column 85
        y = self.train_data[85]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return self
    
    def train_model(self):
        self.model_pipeline.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self):
        y_pred = self.model_pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()

        print(classification_report(self.y_test, y_pred))
        return self
    
    def cross_validate_model(self):
        scores = cross_val_score(self.model_pipeline, self.X_train, self.y_train, cv=5)
        print("Average Cross-Validation Accuracy:", np.mean(scores))
        return self

# Main execution function
def main():
    train_path = '../data/raw/train/ticdata2000.txt'
    eval_path = '../data/raw/eval/ticeval2000.txt'
    target_path = '../data/raw/eval/tictgts2000.txt'
    
    # Initialize the model class
    model = InsuranceModel(train_path, eval_path, target_path)
    
    # Chain the methods for loading data, preprocessing, training, and evaluation
    (model.load_data()
         .preprocess_data()
         .train_model()
         .evaluate_model()
         .cross_validate_model())
    
    # Perform EDA with a separate dataset copy
    eda = DataExplorer(model.eda_data)
    eda.explore_data().plot_histograms().plot_distribution('CARAVAN').plot_correlation_matrix()

if __name__ == "__main__":
    main()
