import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.tree = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def entropy(self, y):
        values, count = np.unique(y, return_counts=True)
        probability = count / count.sum()
        return -np.sum(probability * np.log2(probability))

    def best_split(self, X, y):
        best_threshold = None
        best_feature = None
        previous_entropy = self.entropy(y)
        best_gain = 0

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                    continue

                left_entropy = self.entropy(y[left_idx])
                right_entropy = self.entropy(y[right_idx])

                weighted_entropy = (np.sum(left_idx) / len(y)) * left_entropy + (np.sum(right_idx) / len(y)) * right_entropy
                gain = previous_entropy - weighted_entropy

                if gain > best_gain:
                    best_threshold = threshold
                    best_feature = feature
                    best_gain = gain
        
        return best_feature, best_threshold

    def fit(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return np.bincount(y).argmax()

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.fit(X[left_idx], y[left_idx], depth + 1),
            'right': self.fit(X[right_idx], y[right_idx], depth + 1)
        }

    def sample_predict(self, tree, sample):
        if not isinstance(tree, dict):
            return tree
        
        feature, threshold = tree['feature'], tree['threshold']

        if sample[feature] <= threshold:
            return self.sample_predict(tree['left'], sample)
        else:
            return self.sample_predict(tree['right'], sample)

    def predict(self, X):
        return np.array([self.sample_predict(self.tree, sample) for sample in X])

# Load Dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Our Decision Tree
my_tree = DecisionTree(max_depth=5, min_samples_split=5)
my_tree.tree = my_tree.fit(X_train, y_train)

# Predict
y_pred = my_tree.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Our Decision Tree:", accuracy)

# Compare with Sklearn DecisionTreeClassifier
sklearn_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn = sklearn_tree.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("Accuracy of Sklearn Decision Tree:", accuracy_sklearn)


'''
🔥 Accuracy of Our Decision Tree: 1.0
✅ Accuracy of Sklearn Decision Tree: 1.0
''' 
import numpy as np

X_test = np.array([
    [5.1, 3.5, 1.4, 0.2],  
    [6.2, 2.9, 4.3, 1.3],  
    [7.7, 3.8, 6.7, 2.2],  
    [5.9, 3.0, 5.1, 1.8],  
    [4.8, 3.4, 1.6, 0.2]   
])
y_test = np.array([0, 1, 2, 2, 0])  

resutl_predict = my_tree.predict(X_test_gpt) 
print(resutl_predict)

''' 
[0 1 2 2 0]
''' 
