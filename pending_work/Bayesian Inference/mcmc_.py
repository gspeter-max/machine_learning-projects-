import numpy as np 
import matplotlib.pyplot as plt 


def p(x): 
    return np.exp( -x**2 / 2) 

def sampling(n_samping = 1000, perposal_std = 1.0): 

    sampling = [] 
    x = 0 

    for _ in range(n_samping): 

        x_new = x + np.random.normal(0, perposal_std) 
        accept = np.min( 0, p(x_new) / p(x))

        if np.random.rand() < accept: 
            x = x_new
        sampling.append(x) 
    
    return np.array(sampling) 

sampling = sampling()

plt.figure(figsize = (10,4))
plt.hist(sampling, bins = 100, density = True, alpha = 0.5, color = 'blue') 
plt.show() 

