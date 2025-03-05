import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Assuming df is already defined
x = df.drop(columns=['timestamp', 'target'])
y = df['target']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Train logistic regression model
logist = LogisticRegression()
logist.fit(x_train, y_train)

# Define ECDD Drift Detection
class ECDD:
    def __init__(self, lambdaa=0.90, threshold=0.70):
        self.lambdaa = lambdaa
        self.threshold = threshold
        self.ecdd_loss = None
        self.prev_loss = None  # Correctly initializing previous loss
    
    def update_ecdd_loss(self, loss):
        if self.ecdd_loss is None:
            self.ecdd_loss = loss
            self.prev_loss = loss
            return 'not drift', self.ecdd_loss
        
        self.prev_loss = self.ecdd_loss  # Keep previous loss unchanged before updating
        self.ecdd_loss = self.prev_loss * self.lambdaa + (1 - self.lambdaa) * loss
        
        if abs(self.ecdd_loss - self.prev_loss) > self.threshold:
            return 'drift detect', self.ecdd_loss
        
        return 'not drift', self.ecdd_loss

# Fixing loss computation to avoid log(0) issue
def compute_loss(y_true, y_predict):
    y_predict = np.clip(y_predict, 1e-9, 1 - 1e-9)  # Avoid log(0) issue
    loss = -((y_true * np.log(y_predict)) + ((1 - y_true) * np.log(1 - y_predict)))
    return loss

# Predict probabilities and compute loss
y_predict = logist.predict_proba(x_test)[:, 1]
loss_values = compute_loss(y_test.values, y_predict)

# Apply ECDD
detector = ECDD()
is_drift = []
ecdd_loss_values = []

for loss in loss_values:
    status, ecdd_loss = detector.update_ecdd_loss(loss)
    ecdd_loss_values.append(ecdd_loss)
    is_drift.append(status == 'drift detect')

# Convert to NumPy array for correct indexing
is_drift = np.array(is_drift)

# Scatter plot visualization
plt.figure(figsize=(10, 6))

# Normal data points
plt.scatter(
    x_test.iloc[:, 3], y_test,
    color='blue', label='Normal Data Points', alpha=0.6
)

# Anomalous (drift-detected) points
plt.scatter(
    x_test.iloc[is_drift, 3], y_test[is_drift],
    color='red', label='Drift Detected (Anomalous Points)', alpha=0.9
)
plt.xlabel("Feature 4 (x_test Column 3)")
plt.ylabel("Target (y_test)")
plt.title("ECDD Drift Detection Scatter Plot")
plt.legend()
plt.show()
