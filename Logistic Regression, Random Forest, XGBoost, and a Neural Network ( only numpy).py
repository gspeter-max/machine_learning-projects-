import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix, classification_report


df = pd.read_csv('/content/drive/MyDrive/creditcard.csv')


X = df.drop(columns=['Class'])
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_minority = X_train[y_train == 1].values

class SMOTE:
    def __init__(self, k=5, n_samples=100):
        self.k = k  
        self.n_samples = n_samples  

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def k_neighbors(self, sample, X_minority):
        distances = np.array([self.euclidean_distance(sample, x) for x in X_minority])
        sorted_indices = np.argsort(distances)
        return X_minority[sorted_indices[1:self.k + 1]]  

    def generate_synthetic_samples(self, X_minority):
        synthetic_samples = []

        for _ in range(self.n_samples):
            idx = np.random.randint(0, len(X_minority))  
            sample = X_minority[idx]
            
            neighbors = self.k_neighbors(sample, X_minority)  
            neighbor = neighbors[np.random.randint(0, len(neighbors))]  
            
            diff = neighbor - sample  
            gap = np.random.rand()  
            
            synthetic_sample = sample + gap * diff  
            synthetic_samples.append(synthetic_sample)

        return np.array(synthetic_samples)


smote = SMOTE(k=5, n_samples=500) 
synthetic_samples = smote.generate_synthetic_samples(X_minority)


X_train_balanced = np.vstack((X_train.values, synthetic_samples))
y_train_balanced = np.hstack((y_train.values, np.ones(len(synthetic_samples))))  


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])


logistic_model = LogisticRegression()
logistic_model.fit(X_train_balanced, y_train_balanced)


y_pred = logistic_model.predict(X_test.values)


print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

X = df.drop(columns=['Class'])  
y = df['Class']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight = 'balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Classification Report:\n", classification_report(y_test, y_pred_rf))

precision, recall, _ = precision_recall_curve(y_test, y_pred_rf)
pr_auc = auc(recall, precision)
print(f"Random Precision-Recall AUC : {pr_auc:.4f}")

import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',scaled_weight = 'balanced')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("XGBoost - Classification Report:\n", classification_report(y_test, y_pred_xgb))

precision, recall, _ = precision_recall_curve(y_test, y_pred_xgb)
pr_auc = auc(recall, precision)
print(f"Random Precision-Recall AUC : {pr_auc:.4f}")


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    url = "/content/drive/MyDrive/creditcard.csv"
    df = pd.read_csv(url)  
    
    X = df.drop(columns=['Class']).values  
    y = df['Class'].values.reshape(-1, 1)  
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X) 
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01):
        # Initialize weights & biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)

    def binary_cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)) / m

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y_true):
        m = y_true.shape[0]

        # Compute gradients
        dZ2 = self.A2 - y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU derivative
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X_train, y_train, epochs=200):
        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            loss = self.binary_cross_entropy(y_train, y_pred)
            self.backward(X_train, y_train)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)

input_dim = X_train.shape[1]
hidden_dim = 16

model = NeuralNetwork(input_dim, hidden_dim, learning_rate=0.01)
model.train(X_train, y_train, epochs=200)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"Test Accuracy: {accuracy:.4f}")

output : -- 
'''
<ipython-input-4-c553be2fb11e>:66: RuntimeWarning: overflow encountered in exp
  return 1 / (1 + np.exp(-z))
Confusion Matrix:
[[56863     1]
 [   98     0]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.00      0.00      0.00        98

    accuracy                           1.00     56962
   macro avg       0.50      0.50      0.50     56962
weighted avg       1.00      1.00      1.00     56962

Precision-Recall AUC: 0.0009
Random Forest Confusion Matrix:
 [[56863     1]
 [   24    74]]
Random Forest Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.99      0.76      0.86        98

    accuracy                           1.00     56962
   macro avg       0.99      0.88      0.93     56962
weighted avg       1.00      1.00      1.00     56962

Random Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.99      0.76      0.86        98

    accuracy                           1.00     56962
   macro avg       0.99      0.88      0.93     56962
weighted avg       1.00      1.00      1.00     56962

Random Precision-Recall AUC : 0.8711
/usr/local/lib/python3.11/dist-packages/xgboost/core.py:158: UserWarning: [14:17:19] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "scaled_weight", "use_label_encoder" } are not used.

  warnings.warn(smsg, UserWarning)
XGBoost - Confusion Matrix:
 [[56861     3]
 [   22    76]]
XGBoost - Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.96      0.78      0.86        98

    accuracy                           1.00     56962
   macro avg       0.98      0.89      0.93     56962
weighted avg       1.00      1.00      1.00     56962

Random Precision-Recall AUC : 0.8690
Epoch 0: Loss = 0.6933
Epoch 20: Loss = 0.6457
Epoch 40: Loss = 0.6027
Epoch 60: Loss = 0.5637
Epoch 80: Loss = 0.5284
Epoch 100: Loss = 0.4964
Epoch 120: Loss = 0.4672
Epoch 140: Loss = 0.4407
Epoch 160: Loss = 0.4166
Epoch 180: Loss = 0.3945
Test Accuracy: 0.9983
''' 
