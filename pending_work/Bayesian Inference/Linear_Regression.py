from sklearn.metrics import mean_squared_error 

''' 
1, comppute the inital prior information ( parameters) ( but that get overfit , normal distribution that take )
2. likehood = ( features , house_prices , location)
3. p(d / p) * p( prior ) ==>  that to update you belives 

'''

import numpy as np 

class bayesian_regression: 

    '''def __init__(self,input_shape,uncentinity_parameter):
        self.lambdas = uncentinity_parameter
        self.input_shape = input_shape 
        

    def make_param(self):

        you have a alternative of that for  using mutlinomial or higher normal_distirbution 
        but that is use when you have correlated wights 

        std = np.sqrt(self.lambdas) 
        inital_weight = np.random.normal(0, std , (self.input_shape,1) ) 

    def make_likehood(self,std,x):

        std = np.sqrt(std)
        dist_mean = x * self.make_param()
        likehood = np.random.normal(dist_mean,std)
        return likehood

    '''
    import pandas as pd 
    import numpy as np
    
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    true_slope = 2.5
    true_intercept = -1.0

    # Add Gaussian noise
    noise = np.random.normal(0, 2.0, size=x.shape)
    y = true_slope * x + true_intercept + noise

    x = x.reshape(-1,1) 
    y = y.reshape(-1,1)

    def __init__(self,uncentinity_parameter = 2,input_shape = x.shape):
        
        self.lambdas = uncentinity_parameter
        self.input_shape = input_shape

    # make sure that is in 2d that all 

    def fit(self, x = x, y = y): 
        if isinstance(x,np.ndarray):
            
            Posterior_spradness = np.linalg.inv(np.dot(self.lambdas, np.eye(self.input_shape[1])) +np.dot( x.T , x))
            Posterior_mean =  np.dot(Posterior_spradness , np.dot(x.T , y)).flatten()  
            

            self.Posterior_dist = np.random.multivariate_normal(Posterior_mean, Posterior_spradness)

        
        else: 
            raise ValueError('make sure that x and y in array') 

    def predict(self,x_test): 
        
        if self.Posterior_dist is None: 
            raise ValueError('Posterior dist is None')
        else: 
            x_test = x_test.reshape(-1,1)
            print(self.Posterior_dist.shape)
            y = np.dot(x_test, self.Posterior_dist)
            return y.reshape(-1,1)


model = bayesian_regression()
model.fit() 
    
x_test = np.linspace(0, 10, 100)
y_test = true_slope * x_test + true_intercept + noise
predictions = model.predict(x_test)
print(y_test.shape)
print(predictions.shape)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


        