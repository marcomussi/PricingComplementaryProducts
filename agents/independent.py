import numpy as np
from regressors.gaussianprocess import HeteroscedasticGaussianProcessRegressor



class OptimisticIndependentPricingAgent:


    def __init__(self, actions, L_kernel, beta):
        self.actions = np.array(actions).ravel()
        self.n_actions = actions.shape[0]
        self.beta = beta 
        self.n_samples = np.zeros(self.n_actions)
        self.n_sales = np.zeros(self.n_actions)
        self.base_sigma_sq = 1/4
        self.regressor = HeteroscedasticGaussianProcessRegressor(L_kernel)
        self.last_action = None
    

    def update(self, new_sales, new_impressions):
        self.n_sales[self.last_action] = self.n_sales[self.last_action] + new_sales
        self.n_samples[self.last_action] = self.n_samples[self.last_action] + new_impressions
    

    def pull(self):
        if np.sum(self.n_samples) == 0:
            self.last_action = np.random.randint(0, self.n_actions)
        else:
            mask = self.n_samples > 0
            self.regressor.load_data(self.actions[mask], 
                                     self.n_sales[mask] / self.n_samples[mask], 
                                     self.base_sigma_sq / self.n_samples[mask])
            mu, sigma = self.regressor.compute(self.actions.reshape(-1, 1))
            ucb = mu + self.beta * np.sqrt(sigma)
            self.last_action = np.argmax(self.actions.ravel() * ucb.ravel())
        return self.last_action