# PROJECT – BIKE SHARING (Hour-level Demand Forecasting)

> **Objective:** Predict the number of bikes rented (`cnt`) at an hourly level to optimize inventory planning and bike distribution.

---

## Table of Contents

* [Dataset](#dataset)
* [Setup and Execution](#setup-and-execution)
* [Project Structure](#project-structure)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Feature Engineering](#feature-engineering)
* [Preprocessing and Transformations](#preprocessing-and-transformations)
* [Modeling](#modeling)
* [Evaluation Results](#evaluation-results)
* [Outputs and Kaggle Submission](#outputs-and-kaggle-submission)
* [Notes and Improvement Ideas](#notes-and-improvement-ideas)
* [License](#license)
* [Discussion: Approach to Feature Engineering and Model Selection](#discussion-approach-to-feature-engineering-and-model-selection)

---

## Dataset

* **Source:** [Kaggle – Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset/data)
* **File:** `hour.csv`
* **Target Variable:** `cnt` (total = `casual` + `registered`)
* **Key Columns:**

  * **Time:** `dteday`, `yr` (0=2011, 1=2012), `mnth` (1–12), `hr` (0–23), `weekday`, `workingday`, `holiday`
  * **Weather:** `weathersit` (1–4), `temp`, `atemp`, `hum`, `windspeed`
  * **Users:** `casual`, `registered`

> **Note:** Columns such as `temp`, `atemp`, `hum`, and `windspeed` are normalized.

---

## Setup and Execution

### 1) Environment

Dependencies:

* `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
* `scikit-learn`, `lightgbm`, `xgboost`, `catboost`
* `statsmodels` (for VIF analysis)

```bash
conda create -n bike python=3.10 -y
conda activate bike
pip install numpy pandas matplotlib seaborn scipy scikit-learn lightgbm xgboost catboost statsmodels
```

### 2) Run

Place the dataset under `datasets/hour.csv` and execute the script.

```bash
python bike_sharing_train.py
```

---

## Project Structure

```
.
├─ datasets/
│  └─ hour.csv
├─ bike_sharing_train.py
├─ submission_bike_sharing_model.csv
├─ before_transform.png
└─ README.md
```

---

## Exploratory Data Analysis (EDA)

* Distribution, correlations, and target analysis by category.
* **Findings:**

  * Clear demand peaks at rush hours (7–9 AM, 5–7 PM).
  * Strong seasonal and weather effects (summer > winter).
  * Bad weather correlates with lower rentals.
  * Target variable `cnt` is right-skewed → log transformation applied (`cnt_log = log1p(cnt)`).

---

## Feature Engineering

* **Time Segmentation:** `NEW_time_of_day` (morning/afternoon/evening/night)
* **Rush Hours:** `NEW_rush_hour` (morning_rush/evening_rush/no_rush)
* **Weather Impact:** `NEW_weather_impact` (high/medium/low/very_low)
* **Temperature, Humidity, Wind Categories:** Categorical bins for interpretability.
* **User Categories:** Quantile bins for registered/non-registered counts.
* **Cyclic Encoding:** `hr`, `mnth`, and `weekday` → `sin` and `cos` encoding.

> **Data Leakage Avoidance:** `casual` and `registered` dropped before training.

---

## Preprocessing and Transformations

* Dropped `instant`, `dteday`, `atemp`, `casual`, and `registered`.
* Applied **MinMaxScaler** to all numeric columns (except target).
* Applied **One-Hot Encoding** to categorical columns.
* Conducted **VIF Analysis** → Removed `hum` due to high multicollinearity (VIF ≈ 14.2).

---

## Modeling

* **Models Tested:**

  * `LinearRegression`, `KNN`, `DecisionTreeRegressor`, `RandomForest`, `SVR`, `GradientBoosting`, `XGBoost`, `LightGBM`
* **Target:** Log-transformed `cnt_log`
* **Evaluation Metric:** 10-fold CV RMSE
* **Best Model:** `LightGBM`
* **Hyperparameter Tuning:** GridSearchCV (`learning_rate`, `n_estimators`, `colsample_bytree`) → Best params: `{learning_rate=0.1, n_estimators=500, colsample_bytree=1}`

---

## Evaluation Results

* **LightGBM (CV-10) RMSE ≈ 0.270**
* **Post-Tuning (CV-5) RMSE ≈ 0.274**
* **Train/Test Split RMSE:**

  * Train ≈ **0.1703**
  * Test ≈ **0.1684** → No overfitting detected.
* **Average Prediction Accuracy (custom metric):** ~**95.9%**

> Recommendation: Replace the ad-hoc metric with RMSLE or MAPE for real-world reporting.

**Top Features (by importance):** Time encodings, `temp`, `windspeed`, working day status, and weather impact.

---

## Outputs and Kaggle Submission

Predictions were back-transformed to the original scale (`expm1(cnt_log)`).

Output CSV:

```
submission_bike_sharing_model.csv
└─ Predicted   # Original-scale count predictions
```

---

## Notes and Improvement Ideas

1. Use **TimeSeriesSplit** instead of random CV for temporal consistency.
2. Report **RMSLE/SMAPE** metrics instead of accuracy percentages.
3. Automate hyperparameter tuning with **Optuna** or **Bayesian Optimization**.
4. Compare with **CatBoost/XGBoost** for cross-validation robustness.
5. Add **external weather and holiday APIs** for richer context.
6. Add **lag/rolling windows** to capture temporal dependencies.
7. Address imbalance (e.g., rare `weathersit=4` cases).
8. Integrate **MLflow** for experiment tracking.
9. Build **scikit-learn pipelines** for reproducibility.
10. Deploy with **FastAPI** or **Streamlit** for real-time inference.

---

## License

For educational and research purposes only. Follow the dataset’s license terms.

---

## Discussion: Approach to Feature Engineering and Model Selection

When working with a new dataset, the feature engineering and model selection process typically involves:

1. **Understanding the Data Domain:** Analyze variable meanings, relationships, and potential data leakage risks.
2. **Exploratory Analysis:** Identify trends, seasonality, and outliers using visual and statistical methods.
3. **Feature Engineering:** Derive time-based, categorical, or aggregated features that capture domain behavior.
4. **Encoding and Scaling:** Ensure consistent preprocessing across numerical and categorical data.
5. **Model Benchmarking:** Start with simple interpretable models (Linear/Ridge/Lasso) before moving to complex ones (Boosting/Ensembles).
6. **Validation Design:** Use cross-validation suitable for the data type (e.g., `TimeSeriesSplit` for sequential data).
7. **Hyperparameter Tuning:** Use grid or Bayesian optimization to improve generalization.
8. **Interpretation:** Assess feature importance, SHAP values, and diagnostics to refine engineered features.
9. **Iterative Refinement:** Continuously refine features and models based on evaluation metrics and error analysis.

> The goal is to balance accuracy, interpretability, and generalization, ensuring that the model captures real-world dynamics without overfitting.
