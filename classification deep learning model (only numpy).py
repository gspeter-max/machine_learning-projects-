import numpy as np
from keras.datasets import mnist

class neural_network:
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        z = np.exp(x - np.max(x, axis=0, keepdims=True))
        return z / np.sum(z, axis=0)

    def initial_weights(self):
        weight = {
            'w1': np.random.randn(self.hidden_dim_1, self.input_dim) * 0.01,
            'b1': np.zeros((self.hidden_dim_1, 1)),
            'w2': np.random.randn(self.hidden_dim_2, self.hidden_dim_1) * 0.01,
            'b2': np.zeros((self.hidden_dim_2, 1)),
            'w3': np.random.randn(self.output_dim, self.hidden_dim_2) * 0.01,
            'b3': np.zeros((self.output_dim, 1))
        }
        return weight

    def forward_propagation(self, x, weight):
        z1 = np.dot(weight['w1'], x) + weight['b1']
        a1 = self.relu(z1)

        z2 = np.dot(weight['w2'], a1) + weight['b2']
        a2 = self.relu(z2)

        z3 = np.dot(weight['w3'], a2) + weight['b3']
        a3 = self.softmax(z3)

        cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3}
        return a3, cache

    def derivative_relu(self, x):
        return (x > 0).astype(float)

    def binary_cross_entropy(self, a3, y):
        m = y.shape[1]
        return -np.sum(y * np.log(a3 + 1e-6)) / m

    def backward_propagation(self, x, y, weight, cache):
        m = x.shape[1]

        dz3 = cache['a3'] - y
        dw3 = (1 / m) * np.dot(dz3, cache['a2'].T)
        db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)

        da2 = np.dot(weight['w3'].T, dz3)
        dz2 = da2 * self.derivative_relu(cache['z2'])
        dw2 = (1 / m) * np.dot(dz2, cache['a1'].T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        da1 = np.dot(weight['w2'].T, dz2)
        dz1 = da1 * self.derivative_relu(cache['z1'])
        dw1 = (1 / m) * np.dot(dz1, x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        gradients = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2, 'dw3': dw3, 'db3': db3}
        return gradients

    def update_weights(self, weight, gradients, learning_rate=0.2):
        for key in weight:
            weight[key] -= learning_rate * gradients['d' + key]
        return weight

    def train_model(self, x_train, y_train, epochs=20, batch_size=64, learning_rate=0.2):
        weight = self.initial_weights()
        m = x_train.shape[1]

        for epoch in range(epochs):
            shuffle_indices = np.random.permutation(m)
            x_train = x_train[:, shuffle_indices]
            y_train = y_train[:, shuffle_indices]

            for i in range(0, m, batch_size):
                x_batch = x_train[:, i:i+batch_size]
                y_batch = y_train[:, i:i+batch_size]

                a3, cache = self.forward_propagation(x_batch, weight)
                loss = self.binary_cross_entropy(a3, y_batch)
                gradients = self.backward_propagation(x_batch, y_batch, weight, cache)
                weight = self.update_weights(weight, gradients, learning_rate)

            print(f'Epoch {epoch + 1}: Loss = {loss:.4f}')
        return weight

    def predict(self, x, weight):
        a3, _ = self.forward_propagation(x, weight)
        return np.argmax(a3, axis=0)

    def accuracy(self, x, y, weight):
        y_pred = self.predict(x, weight)
        y_label = np.argmax(y, axis=0)
        return np.mean(y_pred == y_label) * 100

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0


y_train = np.eye(10)[y_train].T
y_test = np.eye(10)[y_test].T
input_size = 784
hidden_size1 = 128
hidden_size2 = 64
output_size = 10
learning_rate = 0.01
epochs = 20
batch_size = 64

model = neural_network(input_size, hidden_size1, hidden_size2, output_size)
weights = model.train_model(X_train, y_train, epochs, batch_size, learning_rate)
accuracy = model.accuracy(X_test, y_test, weights)
print(f'Final Accuracy: {accuracy:.2f}%')

''' 
Epoch 1: Loss = 2.2993
Epoch 2: Loss = 2.2995
Epoch 3: Loss = 2.2418
Epoch 4: Loss = 1.2989
Epoch 5: Loss = 0.5786
Epoch 6: Loss = 0.5084
Epoch 7: Loss = 0.7931
Epoch 8: Loss = 0.3766
Epoch 9: Loss = 0.4583
Epoch 10: Loss = 0.3634
Epoch 11: Loss = 0.2921
Epoch 12: Loss = 0.3622
Epoch 13: Loss = 0.1813
Epoch 14: Loss = 0.1744
Epoch 15: Loss = 0.0692
Epoch 16: Loss = 0.0845
Epoch 17: Loss = 0.5326
Epoch 18: Loss = 0.0951
Epoch 19: Loss = 0.4689
Epoch 20: Loss = 0.3504
Final Accuracy: 93.70%
''' 
