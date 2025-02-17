import numpy as np
import tensorflow as tf

class SelfAttention:
    def __init__(self, q, k, v): 
        self.q = q 
        self.k = k 
        self.v = v 

    def softmax(self, x): 
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Stable softmax
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def transform(self): 
        d_k = self.q.shape[-1]  # Correcting key dimension

        score = np.matmul(self.q, self.k.transpose(0, 2, 1))  
        scale_score = score / np.sqrt(d_k)  # Scale by sqrt(d_k) to stabilize softmax
        att_weight = self.softmax(scale_score)
        output = np.matmul(att_weight, self.v)

        return att_weight, output

class MultiHeadAttention:
    def __init__(self, num_heads, d_model): 
        assert d_model % num_heads == 0, "num_heads must divide d_model"

        self.num_heads = num_heads
        self.d_model = d_model 
        self.d_head = d_model // num_heads  

        
        self.wq = np.random.randn(d_model, d_model) 
        self.wk = np.random.randn(d_model, d_model) 
        self.wv = np.random.randn(d_model, d_model)
        self.wo = np.random.randn(d_model, d_model)  

    def split_heads(self, x): 
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_head)  
        return x.transpose(0, 2, 1, 3)  # [batch, heads, seq, d_head]

    def attention(self, q, k, v): 
        d_k = q.shape[-1]  
        score = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)  
        softmax_score = np.exp(score - np.max(score, axis=-1, keepdims=True))  
        softmax_score /= np.sum(softmax_score, axis=-1, keepdims=True)  
        return np.matmul(softmax_score, v)

    def forward(self, q, k, v): 
        q = np.matmul(q, self.wq) 
        k = np.matmul(k, self.wk) 
        v = np.matmul(v, self.wv) 

        q = self.split_heads(q)
        k = self.split_heads(k) 
        v = self.split_heads(v)

        attention_output = self.attention(q, k, v)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(q.shape[0], -1, self.d_model)
        return np.matmul(attention_output, self.wo)

class FeedForwardNetwork(tf.keras.layers.Layer):  
    def __init__(self, d_model, hidden_dim):  
        super(FeedForwardNetwork, self).__init__()

        self.d_model = d_model  
        self.hidden_dim = hidden_dim

        self.ffn1 = tf.keras.layers.Dense(hidden_dim, activation='relu')  
        self.ffn2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        return self.ffn2(self.ffn1(x))  


class EncoderLayer(tf.keras.layers.Layer):  
    def __init__(self, d_model, num_heads, hidden_dim):  
        super(EncoderLayer, self).__init__()  

        self.mha = MultiHeadAttention(num_heads, d_model)  
        self.ffn = FeedForwardNetwork(d_model, hidden_dim)  

        
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  

    def call(self, x, mask=None):  
        attention_output = self.mha.forward(x, x, x)  
        x = self.ln1(x + attention_output)

        ffn_output = self.ffn.call(x)  
        return self.ln2(x + ffn_output)  
