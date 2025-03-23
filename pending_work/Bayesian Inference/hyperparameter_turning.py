import numpy as np 


import tensorflow as tf 
from tensorflow.keras.optimizers import AdamW
from variational_inference import variational_inference 

def make_turining(model, x_test, y_test, training=True):0
    
    object = variational_inference(21)

    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) 
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    optimizer = AdamW(learning_rate = 0.02, weight_decay = 0.001) 
    with tf.GradientTape() as tape:
        '''
        before doing that model(x_test, training = True) 
        1. compute the loss 
        2. compute the gradients of the loss with respect to the model's trainable variables 
        '''
        predict = model(x_test, training=True)
        loss = object.myloss(y_test, predict)
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
    
    return {'loss': loss.numpy(), 'gradients': gradients} 



object = variational_inference(21)
model = object.combine_model()
x_test = np.random.rand(10,21)
y_test = np.random.rand(10,21)
reuslt = make_turining(model, x_test, y_test, training=True) 
print(reuslt)
