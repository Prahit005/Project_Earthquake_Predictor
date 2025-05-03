# Earthquake Predictor using ML and Time-Series Forecasting

This project predicts seismic activity using hybrid machine learning models and time-series forecasting. It combines classification (Random Forest & Gradient Boosting) with forecasting (ARIMA & LSTM) and includes anomaly detection and feature importance visualization.

## 🧠 Features
- Earthquake classification using Random Forest and Gradient Boosting
- Anomaly detection using Local Outlier Factor (LOF)
- Forecasting with ARIMA and LSTM
- Feature importance analysis with SHAP and bar plots
- Save and reuse the best-performing model (Pickle)
- Batch prediction and individual forecasting supported

## 📁 Files
- `eqpredicton.py`: Main script for training, testing, and forecasting
- `dataset.csv`: Input data (numerical features)
- `best_model.pkl`: Saved classifier (generated after training)

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/earthquake-predictor.git
   cd earthquake-predictor
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the script:
   ```bash
   python eqpredicton.py

## 📊 Example Output
    Random Forest Accuracy: 0.85
    Gradient Boosting Accuracy: 0.88
    Best model saved successfully!
    Future Earthquake Predictions (ARIMA): [...]
    Future Earthquake Predictions (LSTM): [...]

## 🧪 Tech Stack
- Python, NumPy, Pandas, Scikit-learn, TensorFlow
- ARIMA (statsmodels), SHAP, Seaborn, Matplotlib

## 📌 Notes
- The dataset must be structured with numerical features and a label in the last column.
- LSTM and ARIMA use the last column as time-series data.
