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
