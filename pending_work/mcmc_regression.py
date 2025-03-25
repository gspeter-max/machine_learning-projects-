''' Markov Chain '''

import numpy as np 

P = np.array([[0.8, 0.2], [0.6, 0.4]])
states = ["Sunny", "Rainy"]

def made_prediction(steps = 12, state = 0): 
    
    current_state = state 
    for _ in range(steps): 
        print(states[current_state], end = '-->') 
        current_state = np.random.choice([0,1], p =P[current_state]) 
        print(states[current_state], end = '-->') 


''' mcmc regression '''

def make_distribution(n_samples = 100): 

    def pdf(x): 
        return np.exp(-x**2/ 2) 
    

    x = 0 
    samples = [] 

    for _ in range(n_samples): 
        x_new = x + np.random.normal()

        acceptance_ratio = max(0,pdf(x_new) / pdf(x) ) 

        if np.random.normal() < acceptance_ratio: 
            x = x_new
        samples.append(x) 

    return samples 

y = make_distribution()
print(y)

import matplotlib.pyplot as plt  
import seaborn as sns

plt.figure(figsize= (10,4)) 
sns.distplot(y, bins = 20, kde = True) 
plt.show() 



def regression_function(x) : 
    ''' proper regressions ''' 

    import pymc3 as pm
    import matplotlib.pyplot as plt 

    np.random.seed(42)
    X = np.linspace(0, 1, 50)
    true_beta = [2, 3]  # True parameters
    y = true_beta[0] + true_beta[1] 


    with pm.Model() as model: 

        ''' 

        y = mx + c 
        m = cofficients 
        c = intercepts

        '''

        intercepts  = pm.Normal('intercepts', mu = 0, sigma = 10) 
        coffcient = pm.Normal('coffcient', mu = 0, sigma = 10) 
        # here you need opsitive stddev s2 approches you have first is compute exp funciton ====>  np.exp(0.5 * lgo_std) 
        # another is using that pymc3 you comptue halfdistribution  of a particuler std 

        sigma = pm.halfNormal('std', sigma = 1) 

        y = intercepts + x * coffcient 

        y_prad = pm.Normal('y_obs', mu = y, sigma = sigma ,observed = y) 

        trace = pm.sample(100,  return_inferencedata = False) 
        