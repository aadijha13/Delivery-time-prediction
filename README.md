# 🚴 Food Delivery Time Prediction

An end-to-end machine learning project that predicts food delivery time (in minutes) from real-world delivery conditions, deployed as an interactive Streamlit dashboard.

## 🔍 Overview

This project trains a **Random Forest Regressor** to estimate delivery time based on distance, weather, traffic, time of day, vehicle type, food preparation time, and courier experience. The trained model is served through a Streamlit web app with single and batch prediction, live model performance metrics, and dataset insights.

## 📊 Model Performance

Evaluated on a held-out 20% test split (random_state=42):

| Metric | Score |
|---|---|
| R² Score | 0.79 |
| RMSE | 9.62 min |
| MAE | 6.88 min |
| 5-Fold CV R² | 0.74 ± 0.04 |

## 🧠 Features Used

| Feature | Type | Description |
|---|---|---|
| Distance_km | Numerical | Delivery distance in kilometers |
| Weather | Categorical | Weather condition during delivery |
| Traffic_Level | Categorical | Traffic intensity |
| Time_of_Day | Categorical | Morning / Afternoon / Evening / Night |
| Vehicle_Type | Categorical | Bike / Scooter / Car |
| Preparation_Time_min | Numerical | Restaurant food preparation duration |
| Courier_Experience_yrs | Numerical | Courier experience in years |

## ⚙️ Machine Learning Workflow

1. Data cleaning (mode imputation for categoricals, median for courier experience)
2. Exploratory data analysis
3. Label encoding of categorical features
4. Baseline model: Decision Tree Regressor
5. Random Forest Regressor with hyperparameter tuning via `RandomizedSearchCV` (5-fold CV, 100 iterations)
6. Model evaluation (MAE, RMSE, R²)
7. Model serialization with `pickle`
8. Deployment as a Streamlit dashboard

**Best hyperparameters found:** `n_estimators=781, max_depth=11, max_features=3, min_samples_split=3, min_samples_leaf=3`

## 🖥️ Dashboard Features

- **Single Prediction** — enter delivery details and get an instant estimate
- **Batch Prediction** — upload a CSV of multiple deliveries and download scored results
- **Model Insights** — live R²/RMSE/MAE, feature importance, dataset overview charts
- **Prediction History** — track and export predictions made during the session

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Deployment:** Streamlit

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

\`\`\`bash
git clone https://github.com/aadijha13/Delivery-time-prediction.git
cd Delivery-time-prediction
pip install -r requirements.txt
\`\`\`

### Run the app

\`\`\`bash
streamlit run main.py
\`\`\`

The app will open at `http://localhost:8501`.

## 📁 Project Structure

\`\`\`
├── main.py                     # Streamlit dashboard
├── model.ipynb                 # Model training & EDA notebook
├── optimized_rf_model.pkl      # Trained Random Forest model
├── label_encoders.pkl          # Fitted LabelEncoders for categorical features
├── Food_Delivery_Times.csv     # Training dataset
├── requirements.txt            # Python dependencies
└── README.md
\`\`\`


<img width="1440" height="775" alt="Screenshot 2026-08-20 at 12 08 08 AM" src="https://github.com/user-attachments/assets/ef42957e-77c8-420e-b8dd-7b083cffc9bf" />
