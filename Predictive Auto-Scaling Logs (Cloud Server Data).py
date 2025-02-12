# Install required libraries
!pip install shap optuna lightgbm

# Import necessary libraries
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Generate Moving Average
df['moving_avg'] = df['cpu_usage'].rolling(window=4).mean().fillna(0)

# Compute Correlation (Optional: Uncomment if needed)
# correlation = df.corr()

# Create "is_hint" column efficiently
df['is_hint'] = ((df['cpu_usage'] > 78) | (df['memory_usage'] > 79)).astype(int)

# Plot Histogram & Boxplot
df[['cpu_usage', 'memory_usage']].plot(kind='hist', bins=30, alpha=0.5)
df.plot(figsize=(18, 6), kind='box')

# Drop timestamp column safely
df_temp = df.reset_index(drop=True).drop(columns=df.filter(like='timestamp').columns, errors='ignore')

# Check Data Information
df.info()

# Prepare Data for Model
X = df_temp.drop(columns=['auto_scale'])
y = df_temp['auto_scale']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Classification Report
y_pred_rf = rf.predict(X_test)
print(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")

# Cross-validation with TimeSeriesSplit or StratifiedKFold
ts = TimeSeriesSplit(n_splits=5) if 'timestamp' in df.columns else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(rf, X_train, y_train, scoring='roc_auc', cv=ts, verbose=2)
print(f"Mean CV ROC AUC Score: {cv_scores.mean()}")

# SHAP Feature Importance
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values[1], np.array(X_test)[1])

# Compute SHAP Importance Correctly
shap_importance = np.abs(shap_values[1]).mean(axis=1)

# Store Important Features
important_features = pd.DataFrame({
    'column': X_train.columns,
    'importance': shap_importance
}).sort_values(by='importance', ascending=False).head(5)

# Extract Data with Important Features
model_important_data = df.loc[:, important_features['column'].values]

# LightGBM Hyperparameter Optimization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.6, log=True),
        "max_depth": trial.suggest_int('max_depth', 3, 26),
        'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 2e-8, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 2e-8, 10),
        'n_estimators': 1000
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtest],
        callbacks=[
            lgb.early_stopping(50),
            optuna.integration.LightGBMPruningCallback(trial, 'auc')
        ]
    )

    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_pred)

# Run Hyperparameter Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Train Best LightGBM Model
best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# Predict and Evaluate
y_pred_lgb = best_model.predict_proba(X_test)[:, 1]
print(f"Final ROC AUC Score: {roc_auc_score(y_test, y_pred_lgb)}")
