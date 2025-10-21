import numpy as np
from regressors.gaussianprocess import HeteroscedasticGaussianProcessRegressorRBF, GaussianProcessRegressorRBF



class IGPUCB:

    
    def __init__(self, actions, kernel_L, sigma_sq, B, delta, het=True):
        self.actions = actions # (action_number, action_dimension)
        self.action_dim = self.actions.shape[1]
        self.n_actions = self.actions.shape[0]
        self.kernel_L = kernel_L
        self.sigma_sq = sigma_sq
        self.B = B
        self.delta = delta
        self.action_enum = np.linspace(0, self.n_actions-1, self.n_actions, dtype=int)
        self.het = het
        self.reset()
        

    def pull(self):
        if self.no_samples:
            self.last_action = self.actions[np.random.choice(self.action_enum), :]
        else:
            mu, sigmasq = self.regressor.compute(self.actions)
            beta = self.B + self.sigma_sq * np.sqrt(2 * (self.regressor.get_info_gain() + 1 + np.log(1/self.delta)))
            argmax_vals = np.argmax(mu + beta * np.sqrt(sigmasq))
            choice = argmax_vals[0] if isinstance(argmax_vals, np.ndarray) else argmax_vals
            self.last_action = self.actions[choice, :]
        return self.last_action # action, not its id
    
    
    def get_optimistic_estimates(self, return_raw_info=False):
        if self.no_samples:
            if return_raw_info:
                return np.ones((self.n_actions)), np.zeros((self.n_actions)), np.ones((self.n_actions))
            else:
                return np.ones((self.n_actions))
        else:
            mu, sigmasq = self.regressor.compute(self.actions)
            beta = self.B + self.sigma_sq * np.sqrt(2 * (self.regressor.get_info_gain() + 1 + np.log(1/self.delta)))
            ucbs = mu + beta * np.sqrt(sigmasq)
            if return_raw_info:
                return ucbs.ravel(), mu.ravel(), sigmasq.ravel()
            else:
                return ucbs.ravel()
    
    
    def update(self, reward, sample_weight=1):
        if self.last_action is None:
            raise ValueError("No action has been selected yet.")
        if sample_weight != 1 and not self.het:
            raise NotImplementedError("Batch updates not implemented for standard Gaussian Processes.")
        if sample_weight == 0:
            raise ValueError("sample_weight cannot be 0")
        self.regressor.add_sample(self.last_action.reshape(1, self.action_dim), np.array(reward).reshape(1, 1), sample_weight=sample_weight)
        self.no_samples = False
    

    def update_complete(self, action, reward, sample_weight=1):
        if sample_weight != 1 and not self.het:
            raise NotImplementedError("Batch updates not implemented for standard Gaussian Processes.")
        self.regressor.add_sample(action.reshape(1, self.action_dim), np.array(reward).reshape(1, 1), sample_weight=sample_weight)
        self.no_samples = False


    def reset(self):
        self.no_samples = True
        if self.het:
            self.regressor = HeteroscedasticGaussianProcessRegressorRBF(self.kernel_L, self.sigma_sq, 
                                                                        input_dim=self.action_dim, one_sample_mod=True)
        else:
            self.regressor = GaussianProcessRegressorRBF(self.kernel_L, self.sigma_sq, input_dim=self.action_dim, 
                                                         keep_info_gain_estimate=True)
        self.last_action = None
