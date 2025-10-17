# PROJECT – BIKE SHARING (Hour-level Demand Forecasting)

> **Objective:** Predict the number of bikes rented (`cnt`) at an hourly level to optimize inventory planning and bike distribution.

---

## Table of Contents

* [Dataset](#dataset) – explains the data source and key columns.
* [Setup and Execution](#setup-and-execution) – shows how to install libraries and run the project.
* [Project Structure](#project-structure) – shows folder and file organization.
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda) – explores patterns and trends in the data.
* [Feature Engineering](#feature-engineering) – explains how new features were created to improve the model.
* [Preprocessing and Transformations](#preprocessing-and-transformations) – covers cleaning, encoding, and scaling steps.
* [Modeling](#modeling) – describes model training, tuning, and selection.
* [Evaluation Results](#evaluation-results) – summarizes model performance and findings.
* [Outputs and Kaggle Submission](#outputs-and-kaggle-submission) – details how results were saved and submitted.
* [Notes and Improvement Ideas](#notes-and-improvement-ideas) – provides ideas for future improvements.
* [License](#license) – states usage rights and conditions.
* [Discussion: Approach to Feature Engineering and Model Selection](#discussion-approach-to-feature-engineering-and-model-selection) – gives simple steps for working with new datasets.

---

## Dataset

* **Source:** [Kaggle – Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset/data)
* **File:** `hour.csv`
* **Target Variable:** `cnt` (total = `casual` + `registered`)
* **Key Columns:**

  * **Time:** `dteday`, `yr` (0=2011, 1=2012), `mnth` (1–12), `hr` (0–23), `weekday`, `workingday`, `holiday`
    *→ Time-based attributes strongly affect user behavior patterns.*
  * **Weather:** `weathersit` (1–4), `temp`, `atemp`, `hum`, `windspeed`
    *→ Weather influences outdoor activities and rental rates.*
  * **Users:** `casual`, `registered`
    *→ Segments user types for demand interpretation.*

> **Note:** Columns such as `temp`, `atemp`, `hum`, and `windspeed` are normalized.

---

## Setup and Execution

### 1) Environment

Dependencies:

* `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
* `scikit-learn`, `lightgbm`, `xgboost`, `catboost`
* `statsmodels` (for VIF analysis)
  *→ These libraries were chosen for statistical analysis, visualization, and advanced ensemble modeling.*

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

*→ Command-line execution ensures reproducibility across environments.*

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

*→ The structure follows a clean modular approach: raw data, training script, results, and documentation.*

---

## Exploratory Data Analysis (EDA) - explores patterns and trends in the data.

* Distribution, correlations, and target analysis by category.
* **Findings:**

  * Clear demand peaks at rush hours (7–9 AM, 5–7 PM).
    *→ Identifies user commute behavior.*
  * Strong seasonal and weather effects (summer > winter).
    *→ Highlights demand seasonality.*
  * Bad weather correlates with lower rentals.
    *→ Confirms weather sensitivity.*
  * Target variable `cnt` is right-skewed → log transformation applied (`cnt_log = log1p(cnt)`).
    *→ Reduces variance and stabilizes model predictions.*

---

## Feature Engineering- new features were created to improve the model.

* **Time Segmentation:** `NEW_time_of_day` (morning/afternoon/evening/night)
  *→ Captures behavioral differences across day parts.*
* **Rush Hours:** `NEW_rush_hour` (morning_rush/evening_rush/no_rush)
  *→ Reflects high-demand commuting patterns.*
* **Weather Impact:** `NEW_weather_impact` (high/medium/low/very_low)
  *→ Simplifies weather categories for interpretability.*
* **Temperature, Humidity, Wind Categories:**
  *→ Converts continuous variables into meaningful bins.*
* **User Categories:** Quantile bins for registered/non-registered counts
  *→ Normalizes user variation and reduces outlier influence.*
* **Cyclic Encoding:** `hr`, `mnth`, and `weekday` → `sin` and `cos` encoding
  *→ Maintains temporal continuity between cyclical values (e.g., hour 23 → 0).*

> **Data Leakage Avoidance:** `casual` and `registered` dropped before training to ensure model generalization.

---

## Preprocessing and Transformations

* Dropped `instant`, `dteday`, `atemp`, `casual`, and `registered`.
  *→ Removed redundant or leakage-prone columns.*
* Applied **MinMaxScaler** to all numeric columns (except target).
  *→ Normalized feature scales for gradient-based models.*
* Applied **One-Hot Encoding** to categorical columns.
  *→ Converted categorical features to numeric format.*
* Conducted **VIF Analysis** → Removed `hum` (VIF ≈ 14.2).
  *→ Addressed multicollinearity for stable model coefficients.*

---

## Modeling

* **Models Tested:**
  *→ Evaluated diverse algorithms to compare bias-variance performance.*

  * `LinearRegression`, `KNN`, `DecisionTreeRegressor`, `RandomForest`, `SVR`, `GradientBoosting`, `XGBoost`, `LightGBM`
* **Target:** Log-transformed `cnt_log`
  *→ Ensures more Gaussian-like target distribution.*
* **Evaluation Metric:** 10-fold CV RMSE
  *→ Chosen for its interpretability and sensitivity to large errors.*
* **Best Model:** `LightGBM`
  *→ Selected for its efficiency, interpretability, and high performance.*
* **Hyperparameter Tuning:** GridSearchCV (`learning_rate`, `n_estimators`, `colsample_bytree`) → Best: `{learning_rate=0.1, n_estimators=500, colsample_bytree=1}`
  *→ Systematic optimization for balanced bias-variance trade-off.*

---RMSE (Root Mean Squared Error)-> o evaluate the average magnitude of the prediction errors. 
---MAE (Mean Absolute Error):->to understand the average absolute deviation between predicted and actual values.

## Evaluation Results

* **LightGBM (CV-10) RMSE ≈ 0.270**
* **Post-Tuning (CV-5) RMSE ≈ 0.274**
* **Train/Test Split RMSE:**

  * Train ≈ **0.1703**
  * Test ≈ **0.1684** → No overfitting detected.
    *→ Indicates strong generalization capability.*
* **Average Prediction Accuracy (custom metric):** ~**95.9%**
  *→ Demonstrates close alignment between predictions and actuals.*

> **Recommendation:** Replace the custom accuracy with RMSLE or MAPE for production-level interpretability.

**Top Features (by importance):** Time encodings, `temp`, `windspeed`, working day status, and weather impact.
*→ Emphasizes the importance of temporal and environmental factors.*

---

## Outputs and Kaggle Submission

Predictions were back-transformed to the original scale (`expm1(cnt_log)`).
*→ Enables direct comparison with actual bike counts.*

Output CSV:

```
submission_bike_sharing_model.csv
└─ Predicted   # Original-scale count predictions
```

---

## Notes and Improvement Ideas

1. Use **TimeSeriesSplit** instead of random CV for temporal consistency.
   *→ Ensures proper validation for sequential data.*
2. Report **RMSLE/SMAPE** metrics instead of accuracy percentages.
   *→ Provides more robust error interpretation.*
3. Automate hyperparameter tuning with **Optuna** or **Bayesian Optimization**.
   *→ Improves efficiency of model exploration.*
4. Compare with **CatBoost/XGBoost** for robustness.
   *→ Tests cross-model performance consistency.*
5. Add **external weather/holiday APIs** for richer context.
   *→ Expands dataset informativeness.*
6. Add **lag/rolling features** to capture temporal dependencies.
   *→ Helps the model understand past influence on demand.*
7. Address imbalance (e.g., rare `weathersit=4` cases).
   *→ Prevents bias in minority weather conditions.*
8. Integrate **MLflow** for experiment tracking.
   *→ Facilitates reproducibility and team collaboration.*
9. Build **scikit-learn pipelines** for modularity.
   *→ Streamlines training and deployment.*
10. Deploy with **FastAPI** or **Streamlit** for real-time inference.
    *→ Converts research into production-ready applications.*

---

## License

For educational and research purposes only. Follow the dataset’s license terms.
*→ Always verify usage rights before commercial deployment.*
Jupyter Notebook for end-to-end experiment monitoring and rapid iteration.
---

## Feature Engineering and Model Selection

When starting with a new dataset, the process of building features and choosing models usually includes these steps:

1. **Understand the Data:** Learn what each column means and check for mistakes or information leaks.
   *→ Helps you decide which features are useful.*
2. **Explore the Data:** Look at patterns, trends, and unusual values using charts and simple statistics.
   *→ Gives ideas about what affects the target.*
3. **Create New Features:** Make new columns that show time effects, groups, or summaries of the data.
   *→ Adds more useful information for the model.*
4. **Encode and Scale:** Convert text data to numbers and keep all numeric values on a similar scale.
   *→ Helps models learn more evenly.*
5. **Try Different Models:** Start with simple ones (Linear Regression) before trying more complex models (like Gradient Boosting).
   *→ Finds what type of model fits best.*
6. **Use Correct Validation:** For time-based data, use methods like `TimeSeriesSplit` to test properly.
   *→ Checks model performance fairly.*
7. **Tune Parameters:** Use tools like grid search or Optuna to test different settings.
   *→ Improves accuracy without overfitting.*
8. **Interpret the Model:** Look at feature importance or SHAP values to see what matters most.
   *→ Makes the model easier to explain.*
9. **Improve Step by Step:** Change features and models based on results and keep testing.
   *→ Builds a stronger and more stable model over time.*

> **Goal:** Keep a good balance between accuracy, simplicity, and generalization so the model works well on real data.
