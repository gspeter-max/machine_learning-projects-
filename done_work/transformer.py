import numpy as np

class my_transformers:

    def __init__(self, user_querys, d_model):
        self.query_len = len(user_querys)
        self.user_query = user_querys
        self.word_to_index = { word : index for word, index in enumerate(user_querys) }
        self.index_to_word = {index : word for  word , index in enumerate(user_querys) }

        self.d_model  = d_model



    def word_encoding(self):

        self.word_embeding = np.random.randn(self.query_len , self.d_model)
        print(self.word_embeding)
        ''' 
        you have option you create a inital weight of that and after that during training that is update the importance
        or value of the word in duing (self attnetion) 

        self.word_embedding = np.random.rand(len(word_sentance), d_model ) 
        d_model = that is the meaning of that how many demension to discribe you text like 512, 774 


        '''

    def potential_encoding(self):

        self.potential_embedding  = np.zeros((self.query_len, self.d_model))
        print(self.potential_embedding)
        pos = np.arange(self.query_len)[ :, np.newaxis ]
        i = np.arange(self.d_model)[np.newaxis : ]
        angles = pos / np.power(10000 , ((2 *( i // 2)) / np.sqrt(self.d_model)))
        print(angles)

        self.potential_embedding[ :, ::2] = np.sin(angles[ : , ::2] )
        self.potential_embedding[: , 1::2] = np.cos( angles[ :  , 1::2])

        print(self.potential_embedding)

    def po_wr_adding(self):
            self.x = np.add( self.word_embeding , self.potential_embedding )
    
    
    def softmax(self,x):

        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex)

    def self_attention( self, q,k,v):

        score = (np.dot(q, k.T)/ self.d_model)
        soft_score = self.softmax(score)
        return np.dot(soft_score , v )

    def multi_head_attention(self,num_head, x):

        w_q = np.random.randn(self.d_model, self.d_model)
        w_k = np.random.randn(self.d_model, self.d_model)
        w_v = np.random.randn(self.d_model, self.d_model)
        w_0 = np.random.randn(self.d_model, self.d_model)

        q  = np.dot( x, w_q)
        k = np.dot( x, w_k)
        v = np.dot( x, w_v)

        assert self.d_model % num_head == 0

        sub_len = int(self.d_model / num_head)

        head_q = np.reshape( q, (self.query_len,num_head, sub_len))
        head_k = np.reshape( k, ( -1,num_head, sub_len))
        head_v = np.reshape( v, ( -1, num_head, sub_len))

        concat_ = np.zeros((self.query_len , num_head, sub_len))
        for i in range(num_head):
            concat_[:,i,:] = self.self_attention(head_q[:,i,:], head_k[:,i,:], head_v[:,i,:])

        combine_concat_ = np.reshape(concat_, (concat_.shape[0], concat_.shape[1] * concat_.shape[2] ))
        
        print(combine_concat_.shape)
        return np.dot(combine_concat_ , w_0)
        
    def layernormalization(self,x):

        lambdas = np.ones(( self.query_len,self.d_model))
        beta = np.zeros((self.query_len, self.d_model))

        return  ((x - np.mean(x , axis = 1 , keepdims = True))/ np.std(x , axis = 1, keepdims = True))* lambdas  + beta

    def add_norm(self, x,num_head):

        mha_output = self.multi_head_attention(num_head, x)
        add_thing = mha_output + x
        print(add_thing)

        return self.layernormalization(add_thing)

    class feedforward_network:

        def __init__(self,input_shape, laten_shape):
            self.input_shape = input_shpape
            self.laten_shape = laten_shape

        def Leakyrelu(self,x, negative_slop = 0.001 ):
            return np.where(x <= 0, self.negative_slop * x , x)


        def derivative(self,x,negative_slop):
            return np.where( x<= 0 , self.negative_slop , 1)


        def make_weights(self):

            self.w1 =  np.random.randn(self.input_shape,self.laten_shape )* 0.01,
            self.b1 =  np.zeros((self.layer_1_dim,1 )),

            self.w2 =  np.random.randn(self.laten_shape , self.input_shape)* 0.01,

                                                                                                                                                                                            
            weights  = {
                                'w1' : self.w1 ,
                                'b1' : self.b1,
                                'w2' : self.w2,
                                'b2' : self.b2
                            }
            return weights
            
        def forward_propagation(self, x):

            weights = self.make_weights()
            z1 =  np.dot(x, weights['w1'] ) + weights['b1']
            a1 =  self.Leakyrelu(z1)

            z2 = np.dot(a1, weights['w2']) + weights['b2']
            a2 = self.Leakyrelu(z2)

            forward_propagation = {
                    'z1' : z1 ,
                    'a1' : a1 ,
                    'z2' : z2 ,
                    'a2' : a2
                }
            return a2

    
