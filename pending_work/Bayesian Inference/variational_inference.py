
'''
1. first take random (mean, std)
2. compute z= u + std * N~(0 ,1)
3. compute the loss using elbow 
4. do optimization ( sgb)

'''

def compute_stats(): 

    mean = np.random.normal(0,1)
    std = np.random.normal(0,1) 

    std = np.exp( 0.5 * std) 

    z = mean + std * np.random.normal(0,1)

    return  z 

