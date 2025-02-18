import numpy as np
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.keras.layers import Layer 
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam

# Load node features and labels
node_data = {}
with open("cora/cora.content", "r") as f:
    for line in f:
        elements = line.strip().split("\t")
        node_id = int(elements[0])  # First column is node ID
        features = np.array(list(map(int, elements[1:-1])))  # Features (binary)
        label = elements[-1]  # Last column is the label
        node_data[node_id] = (features, label)

# Convert to matrix format
node_ids = list(node_data.keys())
node_features = np.array([node_data[n][0] for n in node_ids])

# Create an index mapping for nodes
node_idx_map = {node: idx for idx, node in enumerate(node_ids)}

# Load graph edges (adjacency list)
edges = []
with open("cora/cora.cites", "r") as f:
    for line in f:
        src, dst = map(int, line.strip().split("\t"))
        if src in node_idx_map and dst in node_idx_map:
            edges.append((node_idx_map[src], node_idx_map[dst]))

# Create adjacency matrix
num_nodes = len(node_ids)
adj_matrix = np.zeros((num_nodes, num_nodes))

for src, dst in edges:
    adj_matrix[src, dst] = 1
    adj_matrix[dst, src] = 1  # Since it's an undirected graph

# Normalize adjacency matrix
def normalize_adj(adj):
    d = np.diag(np.sum(adj, axis=1))
    d_inv_sqrt = np.linalg.inv(np.sqrt(d))
    return d_inv_sqrt @ adj @ d_inv_sqrt

adj_matrix = normalize_adj(adj_matrix)

# Define GNCLayer
class GNCLayer(Layer): 
    def __init__(self, output_dim, activation_function=None, **kwargs): 
        super(GNCLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation_function = activation_function 
        
    def build(self, input_shape): 
        input_dim = input_shape[1]
        self.w = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, adj_matrix):
        output = tf.matmul(adj_matrix, tf.matmul(inputs, self.w))
        if self.activation_function:
            output = self.activation_function(output)
        return output 

# Define GCNModel
class GCNModel(tf.keras.Model): 
    def __init__(self, num_classes): 
        super(GCNModel, self).__init__()
        self.gcn1 = GNCLayer(output_dim=16, activation_function=tf.nn.relu)
        self.gcn2 = GNCLayer(output_dim=num_classes, activation_function=tf.nn.sigmoid)

    def call(self, inputs, adj_matrix): 
        output = self.gcn1(inputs, adj_matrix) 
        output = self.gcn2(output, adj_matrix) 
        return output 

# Get number of classes
labels = [node_data[node_id][1] for node_id in node_ids]
num_classes = len(set(labels))

# Convert labels to one-hot encoding
lb = LabelBinarizer() 
labels_onehot = lb.fit_transform(labels)

# Train-test split
train_idx = np.arange(0, int(0.8 * node_features.shape[0]))
test_idx = np.arange(int(0.8 * node_features.shape[0]), node_features.shape[0])

train_idx = tf.convert_to_tensor(train_idx,dtype = tf.int32)
test_idx = tf.convert_to_tensor(test_idx,dtype = tf.int32)

                            
# Initialize model
model = GCNModel(num_classes)

# Convert inputs to tensors
features = tf.convert_to_tensor(node_features, dtype=tf.float32)
norm_adj = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

# Training setup
optimizer = Adam(learning_rate=0.12)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
epochs = 200

# Training loop
for epoch in range(epochs): 
    with tf.GradientTape() as tape: 
        logits = model(features, norm_adj) 
        loss = loss_fn(tf.gather(labels_onehot,train_idx), tf.gather(logits, train_idx)) 

    gradients = tape.gradient(loss, model.trainable_variables) 
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 

    if epoch % 10 == 0: 
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

# Model evaluation
predictions = model(features, norm_adj)
pred_classes = np.argmax(predictions.numpy(), axis=1)

accuracy = np.mean(pred_classes[test_idx] == np.argmax(labels_onehot[test_idx], axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")

''' 

Epoch 0, Loss: 1.9452
Epoch 10, Loss: 0.2294
Epoch 20, Loss: 0.1002
Epoch 30, Loss: 0.0545
Epoch 40, Loss: 0.0369
Epoch 50, Loss: 0.0276
Epoch 60, Loss: 0.0219
Epoch 70, Loss: 0.0184
Epoch 80, Loss: 0.0162
Epoch 90, Loss: 0.0146
Epoch 100, Loss: 0.0135
Epoch 110, Loss: 0.0127
Epoch 120, Loss: 0.0120
Epoch 130, Loss: 0.0115
Epoch 140, Loss: 0.0111
Epoch 150, Loss: 0.0107
Epoch 160, Loss: 0.0104
Epoch 170, Loss: 0.0102
Epoch 180, Loss: 0.0100
Epoch 190, Loss: 0.0100
Test Accuracy: 77.31%
''' 
