# 🏦 Customer Churn Prediction

A comparative machine learning study predicting whether a bank customer will close their account. Three classification models — **Logistic Regression**, **Random Forest**, and **XGBoost** — are trained on the same pipeline and evaluated side by side to identify the strongest performer and understand the trade-offs between them.

---

## 📊 Dataset

**Source:** [Churn Modelling Dataset — Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

- 10,000 customer records from a fictional European bank
- **Target variable:** `Exited` — whether a customer closed their account (binary)
- Class split: ~80% retained, ~20% churned

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Class balance check to identify dataset imbalance
- Churn rate by geography — which countries show the highest exit rates
- Churn rate by gender and number of products held
- Correlation matrix across all numeric features

### 2. Preprocessing
- Dropped non-predictive identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- **OneHotEncoding** on categorical features: `Geography`, `Gender` (with `drop='first'` to avoid multicollinearity)
- 80/20 stratified train/test split (`random_state=42`)
- **StandardScaler** applied after splitting to prevent data leakage

### 3. Models

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline — interpretable lower-bound benchmark |
| Random Forest | 100-tree ensemble — handles non-linear relationships and feature interactions |
| XGBoost | Gradient boosting with `scale_pos_weight` to address class imbalance directly |

### 4. Evaluation
- 5-fold cross-validation on the training set across accuracy, precision, recall, and F1-score
- Test set classification reports for all three models
- Confusion matrices displayed side by side
- Primary focus on **recall** — minimising missed churners (false negatives) is the core business objective

---

## 📈 Results

	            Accuracy	Precision	Recall	F1-Score
Model				
Logistic Regression 	0.7076	0.3797	0.6865	0.4890
Random Forest	        0.8610	0.7822	0.4423	0.5644
XGBoost	                0.8266	0.5686	0.6190	0.5927


**XGBoost is the strongest overall performer**, achieving the best balance of recall and F1-score on the hold-out test set. Random Forest was competitive. Logistic Regression served as a reliable interpretable baseline.

Accuracy alone is a limited metric here — a naive model predicting "retained" for every customer would still score ~80%. The confusion matrices reveal the real trade-off: Logistic Regression catches more churners but at the cost of more false alarms, while XGBoost strikes the better practical balance between recall and precision.

---

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

Python 3.10+ recommended.

---

## 🚀 Usage

1. Clone the repository
2. Place `Churn_Modelling.csv` in the same directory as the notebook
3. Open `Customer_churn_Prediction.ipynb` in Jupyter or VS Code
4. Run all cells from top to bottom

---

## 🔭 Future Improvements

**Modelling**
- Tune XGBoost hyperparameters (learning rate, max depth, number of estimators) using `Optuna` or `GridSearchCV` — current configuration uses defaults
- Adjust the classification threshold below 0.5 to improve recall on the churned class at an acceptable precision cost
- Apply SMOTE oversampling as an alternative to `scale_pos_weight` for handling class imbalance

**Features**
- Engineer interaction features — e.g. balance-to-salary ratio, products-per-year-of-tenure
- Include recency and engagement signals if available — login frequency, support contacts, transaction volume

**Evaluation**
- Add PR-AUC (Precision-Recall AUC) alongside standard metrics — more informative on imbalanced datasets
- Report cross-validation standard deviations alongside means to capture model stability across folds

**Explainability & Deployment**
- Add SHAP value analysis to identify which features drive individual predictions — adds business interpretability and portfolio depth
- Wrap the XGBoost model in a Streamlit app for interactive churn probability scoring
- Serialise the trained model with `joblib` and expose it via a FastAPI endpoint

---

## 📄 Data Source

[Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

---

## 📝 License

This project is for educational and portfolio purposes.
