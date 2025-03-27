import tensorflow as tf
import numpy as np


class BayesianLinearRegression: 
    
    def __init__(self): 
        self.model = None 
    
    def encoder(self, x): 
        input = tf.keras.Input(shape=(x.shape[1],)) 
        x = tf.layers.dense(x, 100, activation=tf.nn.relu)
        x = tf.layers.dense(x, 200, activation=tf.nn.relu)
        x = tf.layers.dense(x, 200, activation=tf.nn.relu) 
        x = tf.layers.dense(x, 100, activation=tf.nn.relu)
        
        model = tf.keras.Model(input, x) 
        return model 

    