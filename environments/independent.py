import numpy as np



class IndependentPricingEnv:
    """
    This is the simlator to emulate the behavior of a shop which sells product.
    The simulator consider fixed length time periods (that can be assumed as, e.g., one week).
    Assumption: the sales are independent.
    We maximize the net worth.
    """

    
    def __init__(self, n_products, n_actions, actions, demands, user_range, seed=0):
        assert demands.shape == (n_products, n_actions), "Shape of the demand not coherent"
        assert actions.shape == (n_products, n_actions), "Shape of the actions not coherent"
        assert (demands <= 1).all() and (demands >= 0).all(), "Error in demand values"
        assert user_range.shape == (n_products, 2), "Shape of user_range not coherent"
        assert (user_range > 0).all(), "user_range must be strictly greater than zero"
        assert (user_range[:, 0] <= user_range[:, 1]).all(), "user_range not coherent"
        self.n_products = n_products
        self.n_actions = n_actions
        self.demands = demands
        self.actions = actions
        self.user_range = user_range
        self.reset(seed)

    
    def reset(self, seed=0):
        np.random.seed(seed)

    
    def step(self, action):
        assert action.ndim == 1, "The action must be 1-dimensional"
        assert action.shape[0] == self.n_products, "The action must be of dimension n_products"
        sales = np.zeros((self.n_products, 2), dtype=int)
        sales[:, 1] = (np.random.uniform(0, 1, self.n_products) * (
            self.user_range[:, 1] - self.user_range[:, 0]) + self.user_range[:, 0]).astype(np.int64)
        sales[:, 0] = np.random.binomial(sales[:, 1], np.array([self.demands[i, action[i]] for i in range(self.n_products)]))
        return sales

    
    def compute_optimal_actions(self):
        return np.argmax(np.multiply(self.demands, self.actions), axis=1), np.max(np.multiply(self.demands, self.actions), axis=1)



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
