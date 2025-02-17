import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

class AdaptiveGradientBoosting:
    def __init__(self, n_estimators, learning_rate, max_depth):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        total = len(y)
        class_len = np.sum(y == 1)

        initial_prediction = np.mean(y)
        final_prediction = np.full_like(y, initial_prediction, dtype=np.float32)

        class_0_weight = total / (total - class_len)
        class_1_weight = total / class_len

        gradient = final_prediction - y
        hessian = final_prediction * (1 - final_prediction)

        gradient *= class_1_weight * y + class_0_weight * (1 - y)
        hessian *= class_1_weight * y + class_0_weight * (1 - y)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, gradient)  
            self.trees.append(tree)  

            final_prediction += self.learning_rate * tree.predict(X)

            gradient = final_prediction - y
            hessian = final_prediction * (1 - final_prediction)

        self.final_prediction = 1 / (1 + np.exp(-final_prediction))

    def predict(self, X):
        pred = np.full(X.shape[0], np.mean(self.final_prediction), dtype=np.float32)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return 1 / (1 + np.exp(-pred))  

    def evaluate(self, y_true, y_predict):
        y_pred_binary = (y_predict > 0.5).astype(int) 
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        roc_auc = roc_auc_score(y_true, y_predict)  
        return precision, recall, f1, roc_auc

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        best_feature, best_threshold, best_gain = None, None, -np.inf
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                
                mse_left = self._mse(y[left_idx])
                mse_right = self._mse(y[right_idx])
                
                mse_total = (np.sum(left_idx) * mse_left + np.sum(right_idx) * mse_right) / n_samples
                
                gain = self._mse(y) - mse_total
                
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain
        
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.mean(y)
        
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {"feature": feature, "threshold": threshold, "left": left_subtree, "right": right_subtree}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        if isinstance(node, dict):
            if x[node["feature"]] <= node["threshold"]:
                return self._predict_sample(x, node["left"])
            else:
                return self._predict_sample(x, node["right"])
        else:
            return node
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

# Test Code
def test_model():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (np.sum(X, axis=1) > 2.5).astype(int)  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = AdaptiveGradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precision, recall, f1, roc_auc = model.evaluate(y_test, y_pred)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

test_model()
