import numpy as np



class NoiseEnv: 

    
    def __init__(self, actions, values, noise_sigmasq, bernoulli=False):
        assert isinstance(actions, np.ndarray), "NoiseEnv: error action must be a vector"
        assert isinstance(values, np.ndarray), "NoiseEnv: error values must be a vector"
        assert values.ndim == 1 and actions.shape[0] == values.shape[0], "NoiseEnv: error in dimensions"
        self.actions = actions
        self.values = values
        self.bernoulli = bernoulli
        if self.bernoulli:
            self.noise_sigmasq = 1/4
        else: 
            self.noise_sigmasq = noise_sigmasq
        # Fix: convert actions to tuple for dict key
        self.action_to_id = {tuple(self.actions[i]) : i for i in range(self.actions.shape[0])}
        np.random.seed(0)

    
    def step(self, action):
        if self.bernoulli:
            return np.random.binomial(1, self.values[self.action_to_id[tuple(action)]])
        else:
            return np.random.normal(self.values[self.action_to_id[tuple(action)]], np.sqrt(self.noise_sigmasq))

    
    def reset(self, seed):
        np.random.seed(seed)



class DeterministicToyEnv:

    
    def __init__(self, n_products, n_actions, actions, demands, user_range, seed=0):
        self.n_products = n_products
        self.n_actions = n_actions
        self.demands = demands
        self.actions = actions
        self.user_range = user_range

    
    def step(self, action):
        assert action.ndim == 1 and action.shape[0] == self.n_products, "error in action"
        sales = np.zeros((self.n_products, 2), dtype=int)
        sales[:, 1] = 1000
        sales[:, 0] = np.round(sales[:, 1] * np.array([self.demands[i, action[i]] for i in range(self.n_products)]))
        return sales

    
    def compute_optimal_actions(self):
        return np.argmax(np.multiply(self.demands, self.actions), axis=1), np.max(np.multiply(self.demands, self.actions), axis=1)
