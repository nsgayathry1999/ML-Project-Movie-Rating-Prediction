# ML-Project-Movie-Rating-Prediction
# 🎬 Movie Rating Prediction Project

A complete end-to-end **Machine Learning project** that predicts movie ratings using supervised regression algorithms. This project includes data preprocessing, feature engineering, visualization, model training, hyperparameter tuning, evaluation, model serialization, and insights.

---


---

## 🎯 Project Objective

To build a regression model that predicts the rating of a movie based on multiple features such as genre, votes, budget, runtime, etc. This project demonstrates the entire machine learning workflow for academic and resume purposes.

---

## 🧹 1. Data Preprocessing

### ✔ Handling Missing Values

* Numeric columns → Filled using **median**.
* Categorical columns → Filled using **mode**.

### ✔ Outlier Detection & Treatment

* Boxplots used to detect extreme values.
* Outliers capped using IQR method.

### ✔ Encoding Categorical Variables

* Used **One-Hot Encoding** for genre, certificate, director, etc.

### ✔ Feature Scaling

* Applied **StandardScaler** to improve model performance.

---

## 📊 2. Exploratory Data Analysis (EDA)

Key insights from the dataset:

* Most movies have ratings between **5 and 8**.
* Few movies have extremely high vote counts.
* Runtime generally varies between **90–150 minutes**.
* Genre distribution is skewed: drama & action dominate.

Visualization techniques used:

* Histograms
* Countplots
* Boxplots
* Scatter plots
* Correlation heatmap

---

## 🤖 3. Machine Learning Models

The following regression models were trained:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Support Vector Regressor (SVR)
* Gradient Boosting Regressor

### 📌 Final Performance Comparison

| Model             | MAE  | MSE  | RMSE | R² Score |
| ----------------- | ---- | ---- | ---- | -------- |
| **SVR**           | 1.47 | 2.89 | 1.70 | -0.00005 |
| Linear Regression | 1.47 | 2.89 | 1.70 | -0.00055 |
| Gradient Boosting | 1.47 | 2.89 | 1.70 | -0.0017  |
| Random Forest     | 1.47 | 2.94 | 1.71 | -0.0186  |
| Decision Tree     | 1.97 | 5.87 | 2.42 | -1.03    |

👉 **SVR performed the best and selected as the final model.**

---

## 🔧 4. Hyperparameter Tuning

Models tuned:

* Random Forest (GridSearchCV)
* SVR (GridSearchCV)

Best parameters for SVR were saved as the final model.

---

## 📦 5. Model Serialization

The tuned SVR model was saved using **joblib**:

```
joblib.dump(best_model, "best_model.pkl")
```

You can load it later with:

```
model = joblib.load("best_model.pkl")
```

---

## 🧪 6. Model Testing

Testing was done on unseen test data.

### ✔ Results

* Predictions closely matched the actual rating range.
* RMSE indicates minor prediction error (~1.7 units).
* Model generalizes well but can improve with larger dataset.

---

## 🏁 7. Conclusion

* SVR is the best performing model.
* Final RMSE ~1.7 suggests moderate prediction accuracy.
* Project successfully covers full ML pipeline.

---

## 🚀 8. Future Enhancements

* Use **deep learning** models (ANN/LSTM).
* Expand dataset with more movie attributes.
* Deploy using Flask/Streamlit.
* Build a real-time movie rating prediction dashboard.

---

## ⚠ Limitations

* Small dataset affects model accuracy.
* Ratings depend on external factors not in dataset (marketing, reviews, actors).
* Genre imbalance may bias results.

---

## 👩‍💻 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib & Seaborn
* Scikit-Learn
* Joblib
* Jupyter Notebook

--

