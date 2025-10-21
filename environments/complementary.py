import numpy as np



class ComplementaryPricingEnvironment:
    

    def __init__(self, n_products, n_actions, margins, demands, n_baskets, products_probs, alpha, graph_dict, mc_ep=1e3, seed=0):
        assert demands.shape == (n_products, n_actions, 2), "Shape of the demand not coherent"
        assert margins.shape == (n_products, n_actions), "Shape of the margins not coherent"
        assert products_probs.shape == (n_products, ), "Shape of the products_probs not coherent"
        assert (demands <= 1).all() and (demands >= 0).all(), "Error in demand values"
        assert (products_probs <= 1).all() and (products_probs >= 0).all(), "Error in demand values"
        assert alpha >= 0 and alpha <= 1, "alpha must be in [0, 1]"        
        self.n_products = n_products
        self.n_actions = n_actions
        self.n_baskets = n_baskets
        self.demands = demands
        self.margins = margins
        self.alpha = alpha # 1 revenue, 0 profit
        self.products_probs = products_probs
        self.graph_dict = graph_dict # elements for 0 to num_products - 1
        self.mc_ep = mc_ep
        self.margins_to_idx_lst = []
        for prod in range(n_products):
            self.margins_to_idx_lst.append({margins[prod, act]: act for act in range(n_actions)})
        self.leaders_lst = list(self.graph_dict.keys())
        self.followers_lst = list(self.graph_dict.values())
        aux = self.followers_lst.copy()
        self.followers_lst = np.array([x for sublist in self.followers_lst for x in sublist])
        aux.append(self.leaders_lst)
        aux = np.array([x for sublist in aux for x in sublist])
        assert np.issubdtype(aux.dtype, np.integer) and np.all(aux >= 0) and np.all(
            aux <= self.n_products - 1) and len(aux) == len(list(set(aux))), "Error in graph_dict"
        self.follower_to_leader_dict = {value: key for key, values in self.graph_dict.items() for value in values}
        for leader in self.leaders_lst:
                followers = self.graph_dict[leader]
                if isinstance(followers, list):
                    for fl_i in range(1, len(followers)):
                        assert (self.margins[followers[0], :] == self.margins[followers[fl_i], :]).all(), \
                            "Followers of the same leader must have the same margins"
        self.action_values = self.compute_values()
        self.reset(seed)


    def step(self, margins, override_n_baskets=None):
        assert margins.ndim == 1, "The action must be 1-dimensional"
        assert margins.shape[0] == self.n_products, "The action must be of dimension n_products"
        if override_n_baskets is not None:
            n_bsk = override_n_baskets
        else:
            n_bsk = self.n_baskets
        sales_impr_mx = np.zeros((self.n_products, n_bsk, 2), dtype=int) # 3rd dim: 0=sales, 1=impressions
        sales_impr_mx[:, :, 1] = np.random.uniform(0, 1, (self.n_products, n_bsk)) < self.products_probs[:, np.newaxis]
        for leader in self.leaders_lst:
            demand = self.demands[leader, self.margins_to_idx_lst[leader][margins[leader]], 0] 
            sales_impr_mx[leader, :, 0] = np.random.uniform(0, 1, (n_bsk)) < demand
        for follower in self.followers_lst:
            corr_leader = self.follower_to_leader_dict[follower]
            demand = self.demands[follower, self.margins_to_idx_lst[follower][margins[follower]], 0] 
            enhancement_demand = self.demands[follower, self.margins_to_idx_lst[follower][margins[follower]], 1]
            mask = sales_impr_mx[corr_leader, :, 0] == 1
            sales_demand = np.random.uniform(0, 1, (n_bsk - np.sum(mask))) < demand
            sales_enhancement_demand = np.random.uniform(0, 1, (np.sum(mask))) < enhancement_demand 
            sales_impr_mx[follower, np.logical_not(mask), 0] = sales_demand
            sales_impr_mx[follower, mask, 0] = sales_enhancement_demand
        sales_impr_mx[:, :, 0] = np.multiply(sales_impr_mx[:, :, 0], sales_impr_mx[:, :, 1])
        return sales_impr_mx


    def compute_values(self):
        vals_dict = {}
        n_samples = int(self.mc_ep * self.n_baskets)
        for leader in self.leaders_lst:
            followers = self.graph_dict[leader]
            if len(followers) > 0:
                vals = -1 * np.ones((self.n_actions, self.n_actions))
                for leader_margin_i, leader_margin in enumerate(self.margins[leader, :]):
                    for followers_margin_i, followers_margin in enumerate(self.margins[followers[0], :]):
                        margins_action = self.margins[:, 0]
                        margins_action[leader] = leader_margin
                        for follower in followers:
                            margins_action[follower] = followers_margin
                        sales_impr_mx = self.step(margins_action, override_n_baskets=n_samples)
                        sales_mx = sales_impr_mx[:, :, 0]
                        mc_sales = np.sum(sales_mx, axis=1) / n_samples
                        assert mc_sales.shape == (self.n_products, ), "Error in sales shape" # to be removed
                        vals[leader_margin_i, followers_margin_i] = (self.alpha + leader_margin) * mc_sales[leader]
                        for follower in followers:
                            vals[leader_margin_i, followers_margin_i] = vals[leader_margin_i, followers_margin_i] + \
                                (self.alpha + followers_margin) * mc_sales[follower]
            else:
                vals = -1 * np.ones((self.n_actions, ))
                for leader_margin_i, leader_margin in enumerate(self.margins[leader, :]):
                    margins_action = self.margins[:, 0]
                    margins_action[leader] = leader_margin
                    sales_impr_mx = self.step(margins_action, override_n_baskets=n_samples)
                    empirical_demand = np.sum(sales_impr_mx[leader, :, 0]) / n_samples
                    vals[leader_margin_i] = (self.alpha + leader_margin) * empirical_demand
            vals_dict[leader] = vals  
        return vals_dict
    

    def compute_action_value(self, margins):
        assert margins.ndim == 1, "The action must be 1-dimensional"
        assert margins.shape[0] == self.n_products, "The action must be of dimension n_products"
        value = 0
        for leader in self.leaders_lst:
            leader_act_idx = self.margins_to_idx_lst[leader][margins[leader]]
            followers = self.graph_dict[leader]
            if len(followers) == 0:
                value += self.action_values[leader][leader_act_idx]
            else:
                followers_act_idx = self.margins_to_idx_lst[followers[0]][margins[followers[0]]]
                value += self.action_values[leader][leader_act_idx, followers_act_idx]
        return value


    def compute_best_action_value(self):
        value = 0
        for leader in self.leaders_lst:
            value += np.max(self.action_values[leader])
        return value


    def reset(self, seed=0):
        np.random.seed(seed)



"""
class ComplementaryPricingEnvOLD:

    
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
"""