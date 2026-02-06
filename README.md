# CJR-SelfProject
Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

Project Overview
This project implements and evaluates a state-of-the-art deep learning model with attention mechanisms for multivariate time series forecasting. The goal is to move beyond traditional models by explicitly modeling temporal dependencies using attention, improving both accuracy and interpretability.

Project Objectives
- Generate complex, non-stationary multivariate time series data
- Implement an LSTM with attention mechanism
- Compare against baseline models
- Use rolling window cross-validation
- Interpret learned attention weights

Models Implemented
Baseline Models:
1. Historical Average
2. ARIMA
3. Standard LSTM

Proposed Model:
4. LSTM with Bahdanau Attention

Dataset Description
- 5 correlated input features
- 1 target variable
- Includes trend, seasonality, and noise

Validation Strategy
Rolling-origin (walk-forward) cross-validation to prevent data leakage.

Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Attention Analysis
The attention mechanism highlights the importance of recent time steps and seasonal patterns, providing interpretability to model decisions.

Technologies Used
Python, NumPy, Pandas, PyTorch, Scikit-learn, Statsmodels, Matplotlib

Execution
Install dependencies and run:
python attention_time_series_forecasting.py

Author
Dilipkumar Nehru

License
Educational use only.
