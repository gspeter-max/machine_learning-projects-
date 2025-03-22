import pandas as pd 
import numpy as np


def find_prob(df): 

    time_spending = df['time_spending'] 
    probility = df['probability'] 

    probility.iloc[probility < 0 ] = np.nan 
    missing_values = probility.isnull().sum()  
    
    
    if missing_values > 0: 
        remaning_probability  = 1 -  probility.loc[probility != np.nan].sum()
    
        if remaning_probability > 1: 
            raise ValueError ('that input is not compatible') 
    
        probility = probility.fillna((remaning_probability / missing_values))

    probility /= np.sum(probility)

    return np.dot(probility , time_spending) 

data = {
    'time_spending' : [1,2,3,4] , 
    'probability' : [0.15, -0.2, 0.40,0.20] 
}

df = pd.DataFrame(data)
avg_prob  = find_prob(df) 
print(avg_prob) 
