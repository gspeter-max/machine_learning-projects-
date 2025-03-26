from sklearn.preprocessing import LabelEncoder
from typing import Union
import numpy as np
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class explaining:
    def __init__(self, model, x_train, model_type, y_train, top_k=10, categorical_features: Union[list, np.ndarray, str] = None):
        self.model = model
        self.x_train = x_train.copy()  # Avoid modifying original data
        self.model_type = model_type
        self.top_k = top_k
        self.categorical_features = categorical_features
        self.y_train = y_train

        # Encode categorical features safely
        if self.categorical_features is not None:
            if isinstance(self.categorical_features, str):
                self.categorical_features = [self.categorical_features]
            elif isinstance(self.categorical_features, np.ndarray):
                self.categorical_features = self.categorical_features.tolist()

            self.encoders = {}  # Store encoders for inverse transformation
            for col in self.categorical_features:
                le = LabelEncoder()
                self.x_train[col] = le.fit_transform(self.x_train[col])
                self.encoders[col] = le  # Save encoder for reference

    def shap_explain(self):
        # Fit model
        self.model.fit(self.x_train, self.y_train)

        # Select appropriate SHAP explainer
        if self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, self.x_train)
        elif self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise ValueError("Unsupported model type. Use 'linear' or 'tree'.")

        # Compute SHAP values
        shap_values = self.explainer.shap_values(self.x_train)

        # Handle multi-class output
        if isinstance(shap_values, list):  # Multi-class case
            shap_values1 = np.mean(np.abs(shap_values), axis=(0, 1))  # Aggregate across classes
        else:
            shap_values1 = np.mean(np.abs(shap_values), axis=0)

        # Identify top features
        top_features = np.argsort(shap_values1)[::-1][:self.top_k]

        print(f"Total features: {list(self.x_train.columns)}")
        print(f"Top {self.top_k} features: {self.x_train.columns[top_features]}")

        # Convert SHAP values to Explanation object
        single_instance = self.x_train.iloc[0:1]  # Take first observation
        shap_values_single = self.explainer.shap_values(single_instance)

        if isinstance(shap_values_single, list):  # Multi-class
            explanation = shap.Explanation(values=shap_values_single[0], 
                                           base_values=self.explainer.expected_value[0], 
                                           data=single_instance)
        else:  # Binary classification or regression
            explanation = shap.Explanation(values=shap_values_single, 
                                           base_values=self.explainer.expected_value, 
                                           data=single_instance)

        # Summary and waterfall plots
        shap.summary_plot(shap_values, self.x_train)
        shap.plots.waterfall(explanation)

# Load Data
data = pd.read_csv('/content/synthetic_tree_model_data.csv')
x = data.drop('target', axis=1)
y = data['target']

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train and Explain
model = RandomForestClassifier()
categorical = ['job_type', 'education_level', 'marital_status', 'loan_purpose']
explain = explaining(model, x_train, 'tree', y_train, categorical_features=categorical)
explain.shap_explain()
