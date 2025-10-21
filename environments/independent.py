import numpy as np



class IndependentPricingEnvironment:

    
    def __init__(self, n_products, n_actions, margins, demands, user_ranges, alpha, seed=0):
        assert demands.shape == (n_products, n_actions), "Shape of the demand not coherent"
        assert margins.shape == (n_products, n_actions), "Shape of the margins not coherent"
        assert (demands <= 1).all() and (demands >= 0).all(), "Error in demand values"
        assert user_ranges.shape == (n_products, 2), "Shape of user_range not coherent"
        assert (user_ranges > 0).all(), "user_range must be strictly greater than zero"
        assert (user_ranges[:, 0] <= user_ranges[:, 1]).all(), "user_range not coherent"
        assert alpha >= 0 and alpha <= 1, "alpha must be in [0, 1]"
        self.alpha = alpha # 1 revenue, 0 profit
        self.n_products = n_products
        self.n_actions = n_actions
        self.demands = demands
        self.margins = margins
        self.user_ranges = user_ranges
        self.margins_to_idx_lst = []
        for prod in range(n_products):
            self.margins_to_idx_lst.append({margins[prod, act]: act for act in range(n_actions)})
        self.reset(seed)

    
    def step(self, margins):
        assert margins.ndim == 1, "The action must be 1-dimensional"
        assert margins.shape[0] == self.n_products, "The action must be of dimension n_products"
        sales = np.zeros((self.n_products, 2), dtype=int)
        sales[:, 1] = (np.random.uniform(0, 1, self.n_products) * (
            self.user_ranges[:, 1] - self.user_ranges[:, 0]) + self.user_ranges[:, 0]).astype(np.int64)
        vals = -1 * np.ones(self.n_products)
        for i in range(self.n_products):
            vals[i] = self.demands[i, self.margins_to_idx_lst[i][margins[i]]]
        sales[:, 0] = np.random.binomial(sales[:, 1], vals)
        return sales

    
    def compute_optimal_actions_and_values(self):
        vals = np.multiply(self.demands, (self.alpha + self.margins))
        return np.argmax(vals, axis=1), np.max(vals, axis=1), vals
    

    def reset(self, seed=0):
        np.random.seed(seed)
