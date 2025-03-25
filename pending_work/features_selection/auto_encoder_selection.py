'''
           Input X  →  Encoder (L1 + L2 Regularized)  →  Latent Space (z)  
                                      ↓
                      Self-Attention on z  
                                      ↓
             Select Top Features Based on Attention  
                                      ↓
                     Apply SHAP for Explainability  
                                      ↓
               Train Final Model for Prediction


'''


import tensorflow as tf 

class feature_encoder: 

    def __init__(self,lambda_1 , lambda_2): 
        self.lambda1 = lambda_1 
        self.lambda2 = lambda_2 
        self.model = None

        

    def compute_loss(self,y_true , y_pred, lambda_1 = None, lambda_2 = None): 
        
        if lambda_1 is None: 
            lambda_1 = self.lambda1
        if lambda_2 is None :
            lambda_2 = self.lambda2

        lasso = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.model.trainable_weights])
        ridge = tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights]) 

        loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred) 

        return  loss1 + lambda_1 * lasso + lambda_2 * ridge 

    def loss_att(self,y_true , y_pred, lambda_1 = None, lambda_2 = None): 
        
        if lambda_1 is None: 
            lambda_1 = self.lambda1
        if lambda_2 is None :
            lambda_2 = self.lambda2

        lasso = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.model.trainable_weights])
        ridge = tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights]) 

        loss1 = tf.reduce_mean(tf.square(tf.sqrt(y_true - y_pred)))  

        return  loss1 + lambda_1 * lasso + lambda_2 * ridge 

    def encoder(self, input, laten_dim=68): 

        inputs = tf.keras.layers.Input(shape=(input.shape[1],))  
        layer1 = tf.keras.layers.Dense(128, activation='relu')(inputs) 
        layer2 = tf.keras.layers.Dense(laten_dim, activation='relu')(layer1) 

        layer2 = tf.keras.layers.Reshape((1,laten_dim))(layer2) 

        attention_layer = tf.keras.layers.Attention(use_scale=True)
        attention_output = attention_layer([layer2, layer2, layer2]) 
        
        attention_output = tf.keras.layers.Reshape((laten_dim,))(attention_output) 
        

        model = tf.keras.Model(inputs=inputs, outputs=attention_output)
        model.compile(optimizer='adam', loss=self.loss_att)

        return model 




import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Generate synthetic high-dimensional data (1000 rows, 100 features)
x, y = make_classification(n_samples=1000, n_features=100, 
                           n_informative=10, n_redundant=10, 
                           n_classes=2, random_state=42)


feature = feature_encoder(0.01,0.01) 
model = feature.encoder(x) 
y_true = model.predict(x)

from sklearn.decomposition import PCA 

pca = PCA(n_components= 60) 
x = pca.fit_transform(y_true)

from sklearn.model_selection import train_test_split     
from sklearn.ensemble import RandomForestClassifier

 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

clf = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
clf.fit(x_train, y_train) 
import shap 

explainer = shap.TreeExplainer(clf, x_train) 
shap_values = explainer.shap_values(x_train) 

temp_shap = shap_values.mean(axis=0)[:,1]
top_50 = np.argsort(np.abs(temp_shap)[::-1])[:50]

x_train = x_train[:,top_50] 
x_test = x_test[:,top_50] 

clf_1 = RandomForestClassifier( 
    n_estimators=100, random_state=42, n_jobs=-1
) 
clf_1.fit(x_train, y_train) 
y_pred = clf_1.predict(x_test) 

from sklearn.metrics import accuracy_score 

print(f'Accuracy: {accuracy_score(y_test, y_pred)}') 
'''accuracy: 0.79''' 


