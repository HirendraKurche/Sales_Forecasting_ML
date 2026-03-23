# Sales Forecasting with Machine Learning

Comprehensive sales forecasting project using time-based feature engineering and a Random Forest regressor.

## 1. Project Overview

This repository demonstrates an end-to-end machine learning workflow for forecasting daily product sales from historical records.

The current implementation:
- Loads historical train and test datasets.
- Converts date columns into datetime format.
- Performs quick exploratory visualization.
- Creates lag-based time-series features.
- Trains a Random Forest model.
- Evaluates performance using Mean Absolute Percentage Error (MAPE).
- Visualizes feature importance.

## 2. Problem Statement

Accurate sales forecasting helps with:
- Inventory planning.
- Workforce and logistics allocation.
- Revenue planning and risk reduction.

Goal: predict the number of units sold (number_sold) for future dates based on past observations.

## 3. Repository Structure

- README.md: Project documentation.
- Sales_Forecasting_with_Machine_Learning.ipynb: Notebook workflow.
- sales_forecasting_with_machine_learning.py: Script version of the pipeline.
- train.csv: Historical labeled data.
- test.csv: Holdout data used for inference-style workflows.

## 4. Dataset Description

The project expects CSV files with columns similar to:
- Date: Calendar date of the sales observation.
- store: Store identifier.
- product: Product identifier.
- number_sold: Units sold (target in training data).

Notes:
- The script currently filters to one store-product pair for lag-feature modeling:
	- store == 0
	- product == 0
- This makes the current model a focused baseline, not yet a global model across all stores/products.

## 5. Technical Stack

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

## 6. Methodology

### 6.1 Data Loading

Both train and test CSV files are read with pandas.

### 6.2 Date Processing

The Date column is converted to datetime to support temporal plotting and time-aware handling.

### 6.3 Exploratory Data Analysis (EDA)

A line plot is generated for a sample series (store 0, product 0) to inspect trend and temporal behavior.

### 6.4 Feature Engineering

The helper function add_lagged_variables creates lag features:
- lag_1, lag_2, lag_3

These represent previous sales values and allow the model to learn short-term temporal dependencies.

### 6.5 Model Training

- Model: RandomForestRegressor
- n_estimators: 100
- random_state: 42
- Validation strategy: train_test_split with shuffle=False to preserve time ordering.

### 6.6 Evaluation

- Metric: Mean Absolute Percentage Error (MAPE)
- Lower MAPE indicates better relative forecasting performance.

### 6.7 Model Interpretation

Feature importances are extracted and plotted to show which lag features contribute most.

## 7. How to Run

### Option A: Python Script

1. Install dependencies:
	 pip install pandas matplotlib seaborn scikit-learn

2. Ensure train.csv and test.csv are in the project root.

3. Run:
	 python sales_forecasting_with_machine_learning.py

### Option B: Jupyter Notebook

1. Open Sales_Forecasting_with_Machine_Learning.ipynb.
2. Run cells in order.

## 8. Expected Outputs

When you run the script/notebook, you should see:
- EDA line chart for selected store-product data.
- Printed validation MAPE.
- Printed feature importance values.
- Feature importance bar chart.

## 9. Current Scope and Limitations

- Single-series modeling: current workflow trains on one store-product pair.
- Lag-only features: no calendar or holiday variables are included.
- Simplified test evaluation: script demonstrates prediction flow but uses validation predictions as a stand-in for test scoring.
- No model persistence: trained model is not saved to disk.

## 10. Suggested Improvements

High-impact next enhancements:
- Train a global model across all store-product combinations.
- Add calendar features (day_of_week, month, quarter, weekend, holiday flags).
- Use rolling window statistics (moving averages, rolling std).
- Compare additional models (XGBoost, LightGBM, CatBoost, linear baselines).
- Use proper time-series cross-validation.
- Save and reload model artifacts for reproducible inference.
- Add a formal inference script for test.csv prediction output.

## 11. Reproducibility Guidelines

- Keep random_state fixed for deterministic model behavior.
- Pin package versions in a requirements.txt file.
- Log configuration values (lag count, split ratio, model hyperparameters).

## 12. Troubleshooting

- FileNotFoundError for CSV files:
	- Verify train.csv and test.csv exist in the run directory.

- Poor forecasting metrics:
	- Increase feature richness (calendar + rolling features).
	- Revisit split strategy and forecast horizon.
	- Tune hyperparameters.

- Plot does not display:
	- Confirm matplotlib backend support in your environment.

## 13. Future Work Roadmap

- Build modular pipeline functions (load, feature_engineer, train, evaluate, predict).
- Add unit tests for feature engineering logic.
- Add experiment tracking (for example with MLflow).
- Deploy a simple API endpoint for inference.

## 14. Acknowledgments

- Dataset source: Kaggle (as referenced by the project).
- Libraries: pandas, seaborn, matplotlib, scikit-learn.

---

If you want, this README can be further extended with:
- Architecture diagram.
- Model comparison table.
- Example forecast output format.





gt