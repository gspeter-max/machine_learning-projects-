import numpy as np
import pandas as pd
import shap
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# 🚀 1. Generate Synthetic Data
np.random.seed(42)
num_samples = 5000

df = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-01-01', periods=num_samples, freq='H'),
    'cpu_usage': np.random.normal(50, 15, num_samples).clip(0, 100),
    'memory_usage': np.random.normal(50, 15, num_samples).clip(0, 100),
    'network_usage': np.random.normal(10, 5, num_samples).clip(0, 50),
    'disk_io': np.random.normal(20, 10, num_samples).clip(0, 100),
})

# 🚀 2. Inject Anomalies
df.loc[np.random.choice(df.index, size=300, replace=False), ['cpu_usage', 'memory_usage']] += 30
df['label'] = ((df['cpu_usage'] > 80) | (df['memory_usage'] > 85)).astype(int)

# 🚀 3. Feature Engineering
df['rolling_mean'] = df['cpu_usage'].rolling(window=24, min_periods=1).mean()
df['rolling_std'] = df['cpu_usage'].rolling(window=24, min_periods=1).std()

# Drop timestamp
df.drop(columns=['timestamp'], inplace=True)

# 🚀 4. Train-Test Split
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 5. Train Baseline XGBoost
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss', use_label_encoding=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 🚀 6. Evaluate Model
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")

# 🚀 7. SHAP Feature Importance
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 🚀 8. Hyperparameter Tuning with Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float("learning_rate", 1e-2, 0.8),
        'max_depth': trial.suggest_int("max_depth", 2, 10)
    }
    model = XGBClassifier(**params, use_label_encoding=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return roc_auc_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 🚀 9. Train Best Model
best_params = study.best_params
best_model = XGBClassifier(**best_params, use_label_encoding=False, eval_metric='logloss')
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict_proba(X_test)[:, 1]

print(f"🔥 FINAL ROC AUC Score: {roc_auc_score(y_test, y_pred_best)}")
