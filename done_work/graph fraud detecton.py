import networkx as nx
import numpy as np

num_nodes = 10000
num_edges = 100000
G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)

for (u, v) in G.edges():
    G[u][v]['weight'] = np.random.uniform(100, 10000)

print("Graph generated with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges.")

pagerank = nx.pagerank(G, weight='weight')
betweenness = nx.betweenness_centrality(G, weight='weight')

from community import community_louvain

undirected_graph = G.to_undirected()
partition = community_louvain.best_partition(undirected_graph)

features = {}
for node in G.nodes():
    features[node] = {
        'pagerank': pagerank[node],
        'betweenness_centrality': betweenness[node],
        'community': partition[node],
        'transaction_count': G.degree(node),
        'average_transaction': np.mean([G[u][v]['weight'] for u, v in G.edges(node)]) if G.degree(node) > 0 else 0
    }

print("Feature extraction is complete.")
print(features)

import tensorflow as tf

node_features = np.array([[ 
    features[node]['pagerank'],
    features[node]['betweenness_centrality'],
    features[node]['community'],
    features[node]['transaction_count'],
    features[node]['average_transaction']
] for node in G.nodes()])

print(node_features)
labels = np.random.randint(0, 2, size=(num_nodes,))

class FraudGNN(tf.keras.Model):
    def __init__(self, input_dim):
        super(FraudGNN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.output_layer(x)

print(node_features.shape)
input_dim = node_features.shape[1]

model = FraudGNN(input_dim)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(node_features, labels, epochs=100, batch_size=124, validation_split=0.2)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    node_features,
    labels,
    epochs=100,
    batch_size=125,
    validation_split=0.2
)
