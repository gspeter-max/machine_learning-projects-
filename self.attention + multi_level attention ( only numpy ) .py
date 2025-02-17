import numpy as np

class self_attention: 
    def __init__(self , q, k, v): 
        self.q = q 
        self.k = k 
        self.v = v 

    def softmax(self,x): 
        z = np.exp(x - np.max(x, axis = -1, keepdims = True)) 
        return z / np.sum(x, axis = -1 , keepdims = True)
    
    def transform(self): 
        d_k = self.q.shape[-1]
        
        
        score = np.matmul(self.q, self.k.transpose(0, 2, 1))  
        scale_score = score / np.sqrt(d_k)
        att_weight = self.softmax(scale_score)
        output = np.matmul(att_weight, self.v)
        
        return att_weight, output

    
class multi_level_attention:
    
    def __init__(self, num_head, d_model): 
        assert d_model % num_head == 0, 'num must be a division of d_model'
        
        self.num_head = num_head
        self.d_model = d_model 
        self.d_head = d_model // num_head 

        self.wq = np.random.randn(d_model, d_model) 
        self.wk = np.random.randn(d_model, d_model) 
        self.wv = np.random.randn(d_model, d_model)
        self.wo = np.random.randn(d_model, d_model) 

    def split_data(self,x): 
        batch_size , seq_len, d_model = x.shape
        data = x.reshape(batch_size, seq_len, self.num_head, self.d_head) 
        return data.transpose(0, 2, 1, 3) 

    def attentions(self, q, k, v): 
        d_k = q.shape[1] 
        score = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k) 
        exp_score = np.exp(score - np.max(score, axis=-1, keepdims=True)) 
        soft_score = exp_score / np.sum(exp_score, axis=-1, keepdims=True) 
        result = np.matmul(soft_score, v)
        return result 

    def forward(self, q, k, v): 
        q = np.matmul(q, self.wq) 
        k = np.matmul(k, self.wk) 
        v = np.matmul(v, self.wv) 

        q = self.split_data(q)
        k = self.split_data(k) 
        v = self.split_data(v)

        attention_output = self.attentions(q, k, v).transpose(0, 2, 1, 3).reshape(q.shape[0], -1, self.d_model)
        output = np.matmul(attention_output, self.wo)
        return output


def test_attention_classes():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_head = 2
    
    q = np.random.randn(batch_size, seq_len, d_model)
    k = np.random.randn(batch_size, seq_len, d_model)
    v = np.random.randn(batch_size, seq_len, d_model)
    
    # Testing self-attention
    self_attention_instance = self_attention(q, k, v)
    att_weights, output = self_attention_instance.transform()
    print("Self-Attention Weights:\n", att_weights)
    print("Self-Attention Output:\n", output)

    # Testing multi-level attention
    multi_level_attention_instance = multi_level_attention(num_head, d_model)
    multi_level_output = multi_level_attention_instance.forward(q, k, v)
    print("\nMulti-Level Attention Output:\n", multi_level_output)

test_attention_classes()

