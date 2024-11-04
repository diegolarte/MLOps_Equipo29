
# Insurance Model Refactor

This project provides a comprehensive machine learning pipeline for analyzing and predicting insurance policy ownership.
The main script, `Insurance_refactor_fully.py`, includes data exploration, model training, evaluation, and logging capabilities using MLflow.

## Project Structure

- **data/**: Folder for storing datasets.
- **docs/**: Contains documentation and references related to the insurance company benchmark data.
- **models/**: Stores the trained models in `.pkl` format.
- **notebooks/**: Jupyter notebooks for experimentation and additional analysis.
- **reports/**: Directory for generating and storing figures, charts, and textual summaries.
- **Insurance_refactor_fully.py**: Main Python script for running the model pipeline.
- **README.md**: Project documentation.

## Requirements

- Python 3.8+
- Install the necessary packages by running:
  ```bash
  pip install -r requirements.txt
  ```

## Features

### 1. Data Exploration

The `DataExplorer` class provides an interface to explore and summarize the dataset, offering:
- Head of the dataset
- Descriptive statistics
- Data type and null value info

### 2. Model Training

The script uses various models including:
- **XGBoost**: For boosted tree-based predictions.
- **Decision Tree**: Simple tree classifier.
- **Logistic Regression**: For binary classification.

### 3. Oversampling

Utilizes SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

### 4. MLflow Tracking

The project uses MLflow to track experiments, log models, and metrics. Each model’s performance is logged under the "Insurance Model Experiment".

### 5. Model Persistence

Trained models are saved in the `models/` directory using `joblib`.

## Usage

To run the script, execute:

```bash
python notebooks/Insurance_refactor_fully.py
```

### Running MLflow

Make sure to have MLflow running to log and track the experiments:

```bash
mlflow ui
```

You can access the MLflow dashboard at `http://127.0.0.1:5000`.

## License

This project is licensed under the MIT License.

## Author

Martín Jaramillo
