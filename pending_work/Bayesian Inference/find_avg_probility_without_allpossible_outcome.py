import pandas as pd 
import numpy as np


def find_prob(df): 

    time_spending = df['time_spending'] 
    probility = df['probability'] 

    probility.loc[probility < 0 ] = np.nan 
    missing_values = probility.isnull().sum()  
    
    
    if missing_values > 0: 
        remaning_probability  = 1 -  probility.loc[probility.notna()].sum()
    
        if remaning_probability < 1: 
            raise ValueError ('that input is not compatible') 
    
        probility = probility.fillna((remaning_probability / missing_values))

    probility /= np.sum(probility)

    return np.dot(probility , time_spending) 

df = pd.DataFrame({
    'time_spending': np.random.randint(1, 10, size=10**6),  # 1 million random values
    'probability': np.random.rand(10**6)  # 1 million random probabilities
})

df['probability'][np.random.choice(df.index, size=100_000, replace=False)] = np.nan  # 10% missing values
df['probability'][np.random.choice(df.index, size=50_000, replace=False)] = -0.1  # Some negative values

# Run optimized function
print("Final Expected Time:", find_prob(df))