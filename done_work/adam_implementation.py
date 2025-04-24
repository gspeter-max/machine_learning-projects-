import matplotlib.pyplot as plt 
import torch 

class Adam(torch.nn.Module): 

    def __init__(self,beta1,beta2): 
        super().__init__()
        self.beta1 = torch.tensor(beta1)  
        self.beta2 = torch.tensor(beta2)  
        self.g = [] 
        self.w = [] 
    
    def forward(self,turns = 50004,lr = 0.0001): 
        
        w_  = torch.tensor(0.0) 
        v_ = torch.tensor(0.0) 
        s_ = torch.tensor(0.0)
        
        self.w.append(w_.item())
        self.g.append(v_.item())
        
        for i in range(turns): 
            func_derivative = 2 * ( w_ - 3 ) 
            s_ = self.beta2 * s_ + ( 1 - self.beta2 ) * (func_derivative**2) 
            v_ = (self.beta1 * v_) + ( 1 - self.beta1 ) * func_derivative
            v_hat = v_ / (1 - self.beta1**(i + 1 ))
            s_hat = s_ / (1 - self.beta2**(i + 1)) 
            self.g.append(v_.item())
            w_ = w_ - ((lr * v_hat) / (torch.sqrt(s_hat) + 1e-7) ) 
            self.w.append(w_.item())
        
        return (self.g, self.w)

mumentum = Adam(0.999,0.889)
gradient, weights = mumentum.forward()
plt.figure(figsize = (10, 6))
plt.plot(gradient, label = 'gradient_values')
plt.plot(weights, label = 'weight_values')
plt.legend() 
plt.show() 
