# House Prices – Advanced Regression Techniques

Predicting house prices using advanced regression models with feature engineering, cross-validation, and XGBoost. This project follows an end-to-end machine learning workflow, from data exploration to final submission.

---

## 🗂 Project Structure

HousePrices-Kaggle/
│
├── data/
│ ├── raw/ # Original train.csv and test.csv (not included in repo)
├── notebooks/
│ └── 01_exploration_and_baseline.ipynb # Full notebook with EDA, preprocessing, modeling
├── src/
│ └── utils.py # Helper functions (seed, metric, save/load)
├── submissions/
├── requirements.txt
├── .gitignore
└── README.md



---

## 🧠 Features & Techniques

- **Missing values handling:** SimpleImputer for numeric/categorical, special-case imputation for 'None' values  
- **Categorical encoding:** One-Hot and Ordinal encoding  
- **Target transformation:** `log1p` for SalePrice to stabilize variance and normalize distribution  
- **Modeling:** XGBoost as primary model, RandomForestRegressor as baseline  
- **Cross-validation:** K-Fold CV for safe and reproducible evaluation  
- **Hyperparameter tuning:** GridSearch for XGBRegressor (n_estimators, max_depth, learning_rate, subsample, colsample_bytree)  
- **Pipeline:** ColumnTransformer + Pipeline ensures preprocessing and modeling are reproducible and safe from leakage  

---

## 📊 Exploratory Data Analysis (EDA)

- Summary statistics for numeric features  
- Distribution plots (histograms, boxplots)  
- Correlation heatmaps  
- Missing value overview and special-case handling  

---

## ⚙️ How to Run

1. Download `train.csv` and `test.csv` from Kaggle:
   [House Prices – Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
2. Place the files in `data/raw/`  
3. Install required Python packages:

```bash
pip install -r requirements.txt



📈 Model Performance

XGBoost (final model)

CV RMSE: 0.12748

Kaggle Rank: 961 / ~5000 teams

RandomForestRegressor (baseline)

Slightly higher CV RMSE (~0.145), stable but not best-performing

The CV RMSE is calculated in log-space to match Kaggle's evaluation metric (Root-Mean-Squared-Error of log-transformed SalePrice).


💾 Submission

Final predictions generated using the trained XGBoost pipeline:

y_pred = model_pipe.predict(X_test)
y_pred_final = np.expm1(y_pred)  # convert back from log1p
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": y_pred_final
})
submission.to_csv("submission.csv", index=False)


🔗 References

Kaggle Competition: House Prices – Advanced Regression Techniques

Scikit-learn Pipeline & ColumnTransformer docs: https://scikit-learn.org

XGBoost Python docs: https://xgboost.readthedocs.io


