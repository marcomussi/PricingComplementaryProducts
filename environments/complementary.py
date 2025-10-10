import numpy as np



class ComplementaryPricingEnv:
    """
    This is the simlator to emulate the behavior of a shop which sells product.
    The simulator consider fixed length time periods (that can be assumed as, e.g., one week).
    We assume there exists complementary products. We maximize the net worth.
    """

    
    def __init__(self, n_products, n_actions, actions, demands, costs, user_range, 
                 graph_dict, compl_factor_add, compl_factor_molt, mc_ep, seed):
        assert demands.shape == (n_products, n_actions), "Shape of the demand not coherent"
        assert actions.shape == (n_actions, ), "Shape of the actions not coherent"
        assert (demands <= 1).all() and (demands >= 0).all(), "Error in demand values"
        assert user_range.shape == (n_products, 2), "Shape of user_range not coherent"
        assert (user_range > 0).all(), "user_range must be strictly greater than zero"
        assert (user_range[:, 0] <= user_range[:, 1]).all(), "user_range not coherent"
        assert compl_factor_molt >= 1.0, "compl_factor_molt not coherent"
        assert compl_factor_add >= 0.0, "compl_factor_add not coherent"
        self.n_products = n_products
        self.n_actions = n_actions
        self.demands = demands
        self.actions = actions
        self.costs = costs
        self.user_range = user_range
        self.graph_dict = graph_dict
        self.compl_factor_add = compl_factor_add
        self.compl_factor_molt = compl_factor_molt
        self.mc_ep = mc_ep
        self.max_users = np.max(user_range)
        actions0, actions1 = np.meshgrid(self.actions, self.actions)
        self.byvariate_actions = np.hstack((actions0.ravel().reshape(-1, 1), actions1.ravel().reshape(-1, 1)))
        actions0, actions1 = np.meshgrid(np.linspace(0, self.n_actions-1, self.n_actions, dtype=int), 
                                         np.linspace(0, self.n_actions-1, self.n_actions, dtype=int))
        self.byvariate_actions_map = np.hstack((actions0.ravel().reshape(-1, 1), actions1.ravel().reshape(-1, 1)))
        np.random.seed(seed)

    
    def reset(self, seed=0):
        np.random.seed(seed)

    
    def step(self, action):
        assert action.ndim == 1, "The action must be 1-dimensional"
        assert action.shape[0] == self.n_products, "The action must be of dimension n_products"
        demand_acts = np.array([self.demands[i, action[i]] for i in range(self.n_products)])
        cumul_mx = np.zeros((self.n_products, 2), dtype=int)
        cumul_mx[:, 1] = (np.random.uniform(0, 1, self.n_products) * (self.user_range[:, 1] - self.user_range[:, 0]
                                                                  ) + self.user_range[:, 0]).astype(np.int64)
        cumul_mx[:, 0] = np.random.binomial(cumul_mx[:, 1], demand_acts)
        impressions_mx = np.zeros((self.n_products, self.max_users))
        sales_mx = np.zeros((self.n_products, self.max_users))
        for key in list(self.graph_dict.keys()):
            idx_impr = np.random.choice(self.max_users, cumul_mx[key, 1], replace=False)
            impressions_mx[key, idx_impr] = 1
            idx_sales = np.random.choice(idx_impr, cumul_mx[key, 0], replace=False)
            sales_mx[key, idx_sales] = 1
            for elem in self.graph_dict[key]:
                idx_impr = np.random.choice(self.max_users, cumul_mx[elem, 1], replace=False)
                impressions_mx[elem, idx_impr] = 1
                for idx in list(idx_impr):
                    dem = demand_acts[elem]
                    if impressions_mx[key, idx] == 1:
                        dem = dem * self.compl_factor_molt + self.compl_factor_add
                    sales_mx[elem, idx] = np.random.binomial(1, dem)
                cumul_mx[elem, 0] = np.sum(sales_mx[elem, :])
        return cumul_mx, impressions_mx, sales_mx
    

    def compute_optimum(self):
        optimal_actions = -1 * np.ones(self.n_products, dtype=int)
        optimal_value = 0
        for key in list(self.graph_dict.keys()):
            map_idx, value = self._montecarlo_simulation(key, self.graph_dict[key])
            optimal_value += value
            optimal_actions[key] = self.byvariate_actions_map[map_idx, 0]
            for fol in self.graph_dict[key]:
                optimal_actions[fol] = self.byvariate_actions_map[map_idx, 1]
        return optimal_actions, optimal_value

    
    def compute_optimum_indep(self):
        optimal_actions = -1 * np.ones(self.n_products, dtype=int)
        optimal_value = 0
        for key in list(np.linspace(0, self.n_products - 1, self.n_products, dtype=int)):
            map_idx, value = self._montecarlo_simulation(key, [])
            optimal_value += value
            optimal_actions[key] = self.byvariate_actions_map[map_idx, 0]
        return optimal_actions, optimal_value

    
    def _montecarlo_simulation(self, leader, follower_lst):
        obj = np.zeros(self.byvariate_actions.shape[0])
        for i in range(self.byvariate_actions.shape[0]):
            sales_tot = np.zeros(self.n_products)
            action_vect = np.zeros(self.n_products, dtype=int)
            action_vect[leader] = self.byvariate_actions_map[i, 0]
            for j in follower_lst:
                action_vect[j] = self.byvariate_actions_map[i, 1]
            for sim in range(self.mc_ep):
                sales, _, _ = self.step(action_vect)
                sales_tot = sales_tot + sales[:, 0]
            obj[i] = obj[i] + self.costs[leader] * sales_tot[leader] * self.byvariate_actions[i, 0]
            for fol in follower_lst: 
                obj[i] = obj[i] + self.costs[fol] * sales_tot[fol] * self.byvariate_actions[i, 1]
        return np.argmax(obj), np.max(obj) / self.mc_ep
