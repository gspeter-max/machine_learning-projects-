
import numpy as np
import tensorflow as tf


class variational_inference:

    def __init__(self,laten_dim):
        self.laten_dim = laten_dim

        self.mean = None
        self.std = None
        self.log_std = None
        self.z = None
        self.model = self.combine_model()

    def make_stats(self):

        self.mean = tf.Variable(tf.random.normal([self.laten_dim]) )
        self.log_std = tf.Variable(tf.random.normal([self.laten_dim]))

        self.std = tf.exp(0.5 * self.log_std)
        self.z = self.mean + self.std * tf.random.normal([self.laten_dim])

        ''' z = mean + std * Normal~(0,1) '''


    def myloss(self,input, generated,type = 'mse'):

        self.make_stats() 
        if type == "binary":
            loss_function = tf.keras.losses.BinaryCrossentropy()
        else:
            loss_function = tf.keras.losses.MeanSquaredError()

        loss = loss_function(input, generated)

        kl = 0.5 * tf.reduce_mean(self.mean**2 + self.std**2 -1 - self.log_std)
        return kl + loss


    def combine_model(self):

        inputs = tf.keras.layers.Input(shape = (self.laten_dim ,))
        x = tf.keras.layers.Dense(128, activation = 'relu')(inputs)
        x = tf.keras.layers.Dense(self.laten_dim)(x)
        x = tf.keras.layers.Dense(128, activation = 'relu')(x)
        x = tf.keras.layers.Dense(self.laten_dim,activation = 'softmax')(x)

        model = tf.keras.Model(inputs, x)
        return model

    def compileing(self):
        self.model.compile(
        optimizer = tf.keras.optimizers.AdamW(learning_rate= 0.002, weight_decay= 0.001),
        loss = self.myloss ,
        metrics = ['accuracy']

        )

    def fit(self,x_train, y_train, epochs = 30):

        self.model.fit(x_train, y_train, epochs = 30, batch_size = 80) 


    def prediciton(self,x_test):

        output  = self.model.predict(x_test)
        return output

''' 
if __name__ == "__main__": 
    
    vae = variational_inference(21)
    vae.make_stats()
    vae.compileing()

    import numpy as np
    import tensorflow as tf

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data (10000 samples, 21 features)
    x_train = np.random.rand(10000, 21)  # Train data (values between 0 and 1)
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    y_train = x_train.copy()  # VAE tries to reconstruct input

    x_test = np.random.rand(2000, 21)  # Test data
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    y_test = x_test.copy()

    # Convert to TensorFlow tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


    vae.fit(x_train, y_train)

    output = vae.prediciton(x_test)
    print("Reconstructed Output:\n", output)
    print("Original Input:\n", y_test)
    
    import matplotlib.pyplot as plt 
    import seaborn as sns 

    plt.figure(figsize = (10,4)) 
    sns.distplot(output, label = 'Reconstructed Output') 
    sns.distplot(y_test, label = "Original Input") 
    plt.legend() 
    plt.show() 
'''