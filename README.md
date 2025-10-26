# 🍷 Red Wine Quality Prediction — CRISP-DM Full Report

> Using Scikit-learn to analyze and model wine quality based on physicochemical attributes.

Dataset: [Red Wine Quality - UCI/Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`
Framework: **CRISP-DM (Cross-Industry Standard Process for Data Mining)**

---

## Step 1: Business Understanding

The main goal is to **predict the quality of red wine** based on its chemical properties such as acidity, sugar content, pH, and alcohol percentage.

Wine quality, rated on a scale of 0–10, is influenced by complex chemical interactions.
Predicting quality automatically enables:

* **Quality control** in production.
* **Optimization of fermentation** and ingredient adjustments.
* **Efficient product grading** for wineries.

**Objective:** Build a regression model that can accurately predict the numeric wine quality score using scikit-learn.

---

## Step 2: Data Understanding

### 2.1 Data Loading and Overview

The dataset contains **1,599 samples** of red wine with **12 variables**:

* 11 physicochemical attributes (independent variables)
* 1 sensory quality score (dependent variable)

```python
import pandas as pd
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.info()
df.describe()
```

### 2.2 Missing Values

No missing values were found in this dataset.

### 2.3 Exploratory Data Analysis (EDA)

We visualized the distribution of features and correlations to understand relationships with wine quality.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='quality', data=df)
plt.title('Wine Quality Distribution')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```

Key Observations:

* **Alcohol** and **volatile acidity** show strong correlation with wine quality.
* Most quality scores are between **5 and 7**, indicating moderate imbalance.

---

## Step 3: Data Preparation

### 3.1 Feature and Target Split

```python
X = df.drop('quality', axis=1)
y = df['quality']
```

### 3.2 Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3.3 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## Step 4: Modeling

We trained and compared three regression algorithms:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained.")
```

---

## Step 4.5: Feature Selection

To improve performance and interpretability, we selected the **Top 10 features** using `SelectKBest(f_regression)`.

```python
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=10)
X_new = selector.fit_transform(X_train, y_train)

selected_features = X.columns[selector.get_support()]
print(selected_features)
```

### Selected Top Features

| Feature              | Importance (F-score) |
| -------------------- | -------------------- |
| alcohol              | ↑ very important     |
| volatile acidity     | ↑                    |
| sulphates            | ↑                    |
| citric acid          | ↑                    |
| total sulfur dioxide | moderate             |
| density              | moderate             |
| chlorides            | moderate             |
| fixed acidity        | moderate             |
| pH                   | low                  |
| residual sugar       | low                  |

### Visualization

```python
plt.figure(figsize=(10,5))
plt.barh(selected_features, selector.scores_[selector.get_support()], color='teal')
plt.title('Top Feature Scores (SelectKBest)')
plt.xlabel('F-score')
plt.ylabel('Feature')
plt.show()
```

---

## Step 5: Evaluation

### 5.1 Model Evaluation Metrics

We used standard regression metrics:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² (Coefficient of Determination)**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

results = {name: evaluate(model, X_test, y_test) for name, model in models.items()}
pd.DataFrame(results).T
```

| Model             | MAE   | RMSE  | R²    |
| ----------------- | ----- | ----- | ----- |
| Linear Regression | ~0.56 | ~0.74 | ~0.36 |
| Random Forest     | ~0.42 | ~0.64 | ~0.55 |
| Gradient Boosting | ~0.43 | ~0.65 | ~0.53 |

**Random Forest Regressor achieved the best R² ≈ 0.55.**

---

## Step 5.2 Visualization

### Residual Plot

```python
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot - Random Forest")
plt.xlabel("Predicted Quality")
plt.ylabel("Residuals")
plt.show()
```

### Prediction Plot with 95% Confidence Interval

```python
sigma = np.std(residuals)
ci = 1.96 * sigma
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_best, alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
plt.fill_between(
    np.linspace(y_test.min(), y_test.max(), 100),
    np.linspace(y_test.min(), y_test.max(), 100) - ci,
    np.linspace(y_test.min(), y_test.max(), 100) + ci,
    color="lightblue",
    alpha=0.3,
    label="95% Confidence Interval"
)
plt.title("Prediction Plot with Confidence Interval (Random Forest)")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.legend()
plt.show()
```

Interpretation:

* Points near the red line represent accurate predictions.
* Most predictions fall within the 95% confidence band.
* Random Forest captures nonlinear effects between chemical features and wine quality.

---

## Step 6: Deployment (Optional)

The trained best model can be exported using `joblib`.

```python
import joblib
joblib.dump(best_model, 'best_wine_quality_model.pkl')
# Load with:
# model = joblib.load('best_wine_quality_model.pkl')
```

Example usage:

```python
sample = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
predicted_quality = best_model.predict(sample)
print(predicted_quality)
```

---

## 🏁 Step 7: Conclusion

* **Best Model:** Random Forest Regressor
* **Performance:** R² ≈ 0.55, showing moderate predictive capability.
* **Most Important Features:** Alcohol, Volatile Acidity, Sulphates
* **Improvement Ideas:**

  * Apply hyperparameter tuning (GridSearchCV)
  * Test ensemble blending (e.g., stacking)
  * Try nonlinear models (XGBoost, CatBoost)
  * Collect more samples or sensory data for higher accuracy

---

✅ **Final Deliverable:** `Red_Wine_Quality_CRISPDM_Enhanced.ipynb`
📊 **Methodology:** CRISP-DM with Regression Modeling
📈 **Result:** Reliable prediction of wine quality using physicochemical data.
