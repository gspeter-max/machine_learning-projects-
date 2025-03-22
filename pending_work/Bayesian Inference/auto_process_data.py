
class datas: 

    def load_data(self): 

        import pandas as pd 
        import numpy as np


            # Generate synthetic data
            np.random.seed(42)
            X = np.linspace(0, 10, 100)
            true_slope = 2.5
            true_intercept = -1.0

            # Add Gaussian noise
            noise = np.random.normal(0, 2.0, size=X.shape)
            y = true_slope * X + true_intercept + noise

            return x,y 
        