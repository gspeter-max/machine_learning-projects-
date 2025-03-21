import tensorflow as tf
import numpy as np
import requests
import tarfile
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import relu, softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def download_data():
    url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    response = requests.get(url)
    
    with open('cora.tgz', 'wb') as f:
        f.write(response.content)
    
    with tarfile.open('cora.tgz', 'r:gz') as tar:
        tar.extractall()

def load_cora():
    if not os.path.exists('cora/cora.content') or not os.path.exists('cora/cora.cites'):
        download_data()

    data_path = 'cora/cora.content'
    edge_path = 'cora/cora.cites'

    # Load node features and labels
    data = np.genfromtxt(data_path, dtype=str) 
    features = np.array(data[:, 1:-1], dtype=np.float32)
    labels = LabelEncoder().fit_transform(data[:, -1])  

    # Build adjacency matrix
    edges = np.genfromtxt(edge_path, dtype=int)
    index_map = {j: i for i, j in enumerate(data[:, 0].astype(int))}
    edge_index = np.array([(index_map[i], index_map[j]) for i, j in edges])

    num_nodes = features.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for i, j in edge_index:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  

    # Normalize adjacency matrix (Symmetric Normalization)
    adj_matrix += np.eye(num_nodes)  
    D_inv_sqrt = np.diag(1.0 / np.sqrt(adj_matrix.sum(axis=1)))
    adj_matrix = D_inv_sqrt @ adj_matrix @ D_inv_sqrt  

    return features, adj_matrix.astype(np.float32), labels

X, A, y = load_cora()
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


A_train = A[train_idx][:, train_idx]  # Only keep rows & cols of training nodes
A_test = A[test_idx][:, test_idx]  # Only keep rows & cols of testing nodes


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None):
        super().__init__()
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[1], self.output_dim), initializer="glorot_uniform", trainable=True)

    def call(self, inputs, adj):
        Z = tf.matmul(inputs, self.W)  
        Z = tf.matmul(adj, Z)
        if self.activation:
            Z = self.activation(Z)
        return Z


class GCNModel(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.gcn1 = GCNLayer(hidden_dim, activation=relu)
        self.gcn2 = GCNLayer(output_dim, activation=None)  # No activation in last layer
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, adj, training=False):
        x = self.gcn1(inputs, adj)
        x = self.dropout(x, training=training)
        x = self.gcn2(x, adj)
        return x  # Softmax applied in loss function


def train_model():
    num_classes = len(set(y))
    model = GCNModel(hidden_dim=16, output_dim=num_classes)
    optimizer = Adam(learning_rate=0.01)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)  
    
    @tf.function
    def train_step(x, adj, y_true):
        with tf.GradientTape() as tape:
            logits = model(x, adj, training=True)
            loss = loss_fn(y_true, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(200):
        loss = train_step(X_train, A_train, y_train)  
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.numpy():.4f}")

    y_pred = tf.argmax(model(X_test, A_test, training=False), axis=1).numpy() 
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

train_model()
