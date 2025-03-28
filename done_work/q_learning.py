

import numpy as np
import pandas as pd


class QLearning:

    def __init__(self,learning_rate, reward, decision_factor,data,random_probability,epochs = 100):
        self.learning_rate = learning_rate
        self.reward = reward
        self.decision_factor = decision_factor
        self.rp = random_probability
        self.df = data

        self.visit = {(state, action): 0 for state in self.df.index for action in self.df.columns }
        self.index_to_state = {index : state for index, state in enumerate(self.df.index)}
        self.epochs = epochs 


    def random_actions(self,state) -> str:
        if np.random.uniform(0,1) < self.rp:
            return str(np.random.choice(self.df.columns))
        return str(self.df.loc[state].idxmax())

    def update_qvalues(self,state : str,action : str,next_state : str,next_action: str):
        if state not in self.df.index and action not in self.df.columns: 
            return 
            
        self.visit[(state,action)] += 1

        alpha = 1 / ( 1 + self.visit[(state,action)])

        updated =  self.get_qvalues(state,action) + (alpha * ( self.reward + (self.decision_factor * self.get_qvalues(next_state,next_action)) - self.get_qvalues(state,action)))
        self.df.loc[state,action] = updated
        return updated

    def get_qvalues(self,state : str,action : str) -> float:
        return  float(self.df.loc[state,action])

    def fit(self):
        for _ in range(self.epochs): 
            for state_index in range(len(self.df.index)):
                state = self.index_to_state[state_index]
                action = self.random_actions(state)
                next_state = str(np.random.choice(self.df.index))
                next_action = str(np.random.choice(self.df.columns)) 
                self.update_qvalues(state,action,next_state,next_action)

# Example Usage

states = ["red", "green", "blue"]
actions = ["go", "stop", "wait"]

np.random.seed(42)
data = {
    "State": np.random.choice(states, 500),
    "Action": np.random.choice(actions, 500),
    "Q-Value": np.random.uniform(-5, 5, 500)
}
df = pd.DataFrame(data)
df = df.pivot_table(values="Q-Value", columns="Action", index="State", aggfunc="mean")

# Initialize & Train Q-Learning
ql = QLearning(learning_rate=0.1, reward=-100, decision_factor=0.9, data=df, random_probability=0.45)
print("Initial Q-table:\n", ql.df)
ql.fit()
print("Updated Q-table after training:\n", ql.df)
