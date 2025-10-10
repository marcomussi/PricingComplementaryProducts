import numpy as np



class BernoulliEnvironmentOneDim: 

    
    def __init__(self, mean0, seed=0):
        self.mean0 = mean0
        self.reset(seed)  

    
    def step(self, action):
        return np.random.binomial(1, self.mean0(action))

    
    def reset(self, seed=0):
        np.random.seed(seed)



class BernoulliEnvironmentTwoDims: 

    
    def __init__(self, mean0, mean1, seed=0):
        self.mean0 = mean0
        self.mean1 = mean1
        self.reset(seed)  

    
    def step(self, action):
        return np.random.binomial(1, self.mean0(action[0]) * self.mean1(action[1]))

    
    def reset(self, seed=0):
        np.random.seed(seed)
