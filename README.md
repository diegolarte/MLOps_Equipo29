
# Insurance Model Refactoring Project

This project focuses on the analysis and prediction of insurance policies using machine learning. The project structure includes model training, evaluation, and documentation of insurance data processing and model management. Key components include data preprocessing, model development, MLflow for model tracking, and various utilities for maintaining a robust and efficient pipeline.

## Table of Contents
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Models](#models)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Tracking Experiments](#tracking-experiments)
- [Contributing](#contributing)

---

## Project Structure

```
project-directory/
├── data/                       # Directory for raw and processed data files
├── docs/                       # Documentation for data and project
│   └── insurance+company+benchmark+coil+2000/
│       ├── dictionary.txt      # Dictionary of terms and data definitions
│       ├── tic.data.html       # HTML file for data visualization
│       ├── tic.html            # HTML file for data overview
│       ├── tic.task.html       # Task descriptions
│       └── TicDataDescr.txt    # Data description file
├── mlops_equipo29/             # Python package directory
│   └── __init__.py             # Package initialization file
├── models/                     # Directory for trained model files
│   ├── dt_model.pkl            # Decision Tree model
│   ├── insurance_model.pkl     # General insurance model
│   ├── lr_model.pkl            # Logistic Regression model
│   └── xgb_model.pkl           # XGBoost model
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── Insurance_refactor_fully.py # Core Python script for model refactoring
│   └── Insurance.ipynb         # Jupyter notebook for exploration and analysis
├── mlruns/                     # MLflow tracking directory
├── references/                 # Directory for project references
├── reports/                    # Generated analysis as reports
│   └── figures/                # Figures and visuals for reports
├── pyproject.toml              # Python project configuration file
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies and requirements
└── setup.cfg                   # Configuration file
```

## Data Sources

The data for this project is located in the `data/` directory. Documentation on the dataset can be found in the `docs/insurance+company+benchmark+coil+2000` directory, which includes:
- `dictionary.txt`: A text file explaining the terms and fields used in the dataset.
- `tic.data.html`, `tic.html`, `tic.task.html`: HTML files providing an overview of the data and tasks.
- `TicDataDescr.txt`: A detailed description of the dataset.

## Models

The project employs various machine learning models to analyze and predict insurance-related outcomes:
- **Decision Tree (`dt_model.pkl`)**: A basic tree-based model.
- **Logistic Regression (`lr_model.pkl`)**: A linear model for binary classification.
- **XGBoost (`xgb_model.pkl`)**: An optimized gradient boosting model for higher performance.
- **General Insurance Model (`insurance_model.pkl`)**: An additional model trained specifically on insurance data.

These models are saved as `.pkl` files in the `models/` directory.

## Setup and Installation

To set up and run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/insurance-model-refactoring.git
   cd insurance-model-refactoring
   ```

2. **Install Dependencies**:
   Use the following command to install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Place your dataset files in the `data/` directory. Ensure all data is formatted correctly as per the requirements defined in the `docs/`.

4. **Set up MLflow**:
   This project uses MLflow for experiment tracking. Make sure MLflow is installed and set up on your system.

## Usage

1. **Model Training and Refactoring**:
   The `Insurance_refactor_fully.py` script is the main Python file for model training, evaluation, and logging with MLflow. It includes the following features:
   - Data loading and preprocessing
   - Model training and evaluation
   - Experiment tracking with MLflow

   Run the script using:
   ```bash
   python notebooks/Insurance_refactor_fully.py
   ```

2. **Jupyter Notebooks**:
   The `notebooks/Insurance.ipynb` notebook is available for exploratory data analysis and model testing. It allows you to visualize data, try out models, and refine them.

3. **Makefile**:
   This file contains common commands and shortcuts for running the project. To see available commands, use:
   ```bash
   make help
   ```

## Tracking Experiments

This project uses **MLflow** for tracking model training experiments. Each run logs model parameters, metrics, and artifacts for easy comparison and analysis.

### Start MLflow Tracking Server

To start the MLflow tracking server, run:
```bash
mlflow ui
```
Open a web browser and navigate to `http://localhost:5000` to access the MLflow UI.

### Log Experiments

Experiments are automatically logged in the `mlruns/` directory. Each run is recorded with relevant parameters, metrics, and artifacts for easy monitoring and comparison.

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and create a pull request with your changes. Ensure your code is well-documented and tested.

---

### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
