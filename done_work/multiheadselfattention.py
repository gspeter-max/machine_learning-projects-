
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Layer

class MultiHeadSelfAttention(Layer): 

    def __init__(self, embiding_size, num_head): 
        super(MultiHeadSelfAttention, self).__init__()

        self.total_embiding_size = embiding_size
        self.num_head = num_head 
        self.head_embiding_size = embiding_size // num_head 

        self.Dense_query = Dense(embiding_size)
        self.Dense_key = Dense(embiding_size)
        self.Dense_value = Dense(embiding_size)

        self.combine_ = Dense(embiding_size)


    def self_attention(self,q, k, v): 
        score = tf.matmul(q, k, transpose_b = True)
        d_k = tf.cast(k.shape[-1],dtype = tf.float32)
        weight = tf.nn.softmax(score/tf.sqrt(d_k)) 
        output = tf.matmul(weight , v)
        return output , weight 


    def head_split(self, x, batch_size):
        x = tf.reshape(x, (batch_size,-1, self.num_head, self.head_embiding_size))
        return tf.transpose(x,(0,2,1,3))
    
    def call(self, input): 
        batch_size = input.shape[0]

        q = self.Dense_query(input)
        k = self.Dense_key(input)
        v = self.Dense_value(input)

        q_head = self.head_split(q, batch_size)
        k_head = self.head_split(k , batch_size) 
        v_head = self.head_split(v , batch_size)

        atttention_score , _ = self.self_attention(q_head, k_head, v_head) 
        
        attention_score = tf.transpose(atttention_score, (0,2,1,3))
        combine_heads = tf.reshape(attention_score , (batch_size, -1, self.total_embiding_size))

        output = self.combine_(combine_heads)

        return output

''' test code '''         
import tensorflow as tf

# Define input parameters
batch_size = 2
sequence_length = 5
embedding_size = 16
num_heads = 4

# Create random input data
test_input = tf.random.uniform((batch_size, sequence_length, embedding_size))

# Initialize your MultiHeadSelfAttention layer
attention_layer = MultiHeadSelfAttention(embedding_size, num_heads)

# Pass the input through the layer
output = attention_layer(test_input)

# Print output shape and output tensor
print("Input Shape:", test_input.shape)
print("Output Shape:", output.shape)
print("Output Tensor:\n", output)
