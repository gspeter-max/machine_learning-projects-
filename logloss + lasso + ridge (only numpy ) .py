import numpy as np

class regularization_techniques: 

    def __init__(self,m,y_true, y_pred,reg_params : float,cofficient,elasticNetParam : float ): 
        self.m = m 
        self.y_true = y_true
        self.y_pred = y_pred 
        self.reg_params = reg_params 
        self.cofficient = cofficient 
        self.elasticNet = elasticNetParam 


    def logloss(self): 
        upper_division = (self.y_true * np.log(self.y_pred) ) + ( (1 - self.y_true) * np.log(1 - self.y_pred))
        result = (-1 / self.m) * np.sum(upper_division) 
        return np.array(result) 

    def elastic_net(self): 
        first_term = self.elasticNet * np.sum(np.abs(self.cofficient))
        second_term = (1 - self.elasticNet) * np.sum(np.square(self.cofficient)) 
        elastic = first_term  + second_term  
        return np.array(elastic)

    def call(self) :
        first  = self.reg_params * self.elastic_net()
        result = self.logloss() + np.float64(first) 
        return result 
        
m = 100  # Number of samples
y_trues = np.random.randint(0, 2, size=m)  # Random binary labels (0 or 1)
y_pred = np.random.rand(m)  # Predicted probabilities (between 0 and 1)
coefficients = np.random.randn(10)  # Random coefficients (theta values)
  # Single bias term (theta_0)
reg_param = 0.1  
elastic_net_param = 0.5 

losses = regularization_techniques(m, y_trues,y_pred, reg_param,coefficients, elastic_net_param)
loss = losses.call()
print(loss)
