import numpy as np



class ComplementaryPricingEnvironment:
    

    def __init__(self, n_products, n_actions, margins, demands, n_baskets, products_probs, alpha, graph_dict, mc_ep=1000, seed=0):
        assert demands.shape == (n_products, n_actions, 2), "Shape of the demand not coherent"
        assert margins.shape == (n_products, n_actions), "Shape of the margins not coherent"
        assert products_probs.shape == (n_products, ), "Shape of the products_probs not coherent"
        assert (demands <= 1).all() and (demands >= 0).all(), "Error in demand values"
        assert (products_probs <= 1).all() and (products_probs >= 0).all(), "Error in demand values"
        assert alpha >= 0 and alpha <= 1, "alpha must be in [0, 1]"   
        assert n_baskets >= 1, "n_baskets must be positive"  
        assert mc_ep >= 10, "mc_ep too low for reliable estimates"   
        self.n_products = n_products
        self.n_actions = n_actions
        self.n_baskets = n_baskets
        self.demands = demands
        self.margins = margins
        self.alpha = alpha # 1 revenue, 0 profit
        self.products_probs = products_probs
        self.graph_dict = graph_dict # elements for 0 to num_products-1
        self.mc_ep = mc_ep
        self.margins_to_idx_lst = []
        for prod in range(n_products):
            self.margins_to_idx_lst.append({self.margins[prod, idx]: idx for idx in range(0, self.n_actions)})
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
                assert isinstance(followers, list), "All graph_dict values must be of type list"
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
        sales_impr_mx = np.ones((self.n_products, n_bsk, 2), dtype=int) # 3rd dim: 0=sales, 1=impressions
        
        sales_impr_mx[:, :, 1] = np.random.uniform(0, 1, (self.n_products, n_bsk)) < self.products_probs[:, np.newaxis]
        
        for leader in self.leaders_lst:
            aux = self.margins_to_idx_lst[leader][margins[leader]]
            demand = self.demands[leader, aux, 0] 
            sales_impr_mx[leader, :, 0] = np.random.uniform(0, 1, (n_bsk)) < demand
        
        for follower in self.followers_lst:
            
            corr_leader = self.follower_to_leader_dict[follower]
            
            demand = self.demands[follower, self.margins_to_idx_lst[follower][margins[follower]], 0] 
            enhancement_demand = self.demands[follower, self.margins_to_idx_lst[follower][margins[follower]], 1]
            
            mask_leader_sales = sales_impr_mx[corr_leader, :, 0] == 1
            
            sales_demand = np.random.uniform(0, 1, (n_bsk - np.sum(mask_leader_sales))) < demand
            sales_enhancement_demand = np.random.uniform(0, 1, (np.sum(mask_leader_sales))) < enhancement_demand

            sales_impr_mx[follower, ~mask_leader_sales, 0] = sales_demand
            sales_impr_mx[follower, mask_leader_sales, 0] = sales_enhancement_demand
        
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
                        margins_action = self.margins[:, 0].copy() # ERRATO???
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
                    margins_action = self.margins[:, 0].copy() # ERRATO???
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
