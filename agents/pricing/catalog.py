import numpy as np
from agents.pricing.independent import OptIndepPricingAgent
from agents.pricing.complementaryset import OptComplementarySetPricingAgent



class CatalogPricingAgent:


    def __init__(self, n_products, n_actions, margins, alpha, kernel_L, horizon, graph_dict=None):
        assert margins.shape == (n_products, n_actions), "Shape of the margins not coherent"
        assert alpha >= 0 and alpha <= 1, "alpha must be in [0, 1]"     
        self.margins = margins # (action_number, action_dimension)
        self.n_products = n_products
        self.n_actions = n_actions
        self.kernel_L = kernel_L
        self.horizon = horizon
        self.alpha = alpha # 1 revenue, 0 profit
        if graph_dict is not None:
            self.known_graph = True
            self.graph_dict = graph_dict
            self.agents_dict = {}
            for leader in self.graph_dict.keys():
                if len(self.graph_dict[leader]) == 0:
                    self.agents_dict[leader] = OptIndepPricingAgent(actions=margins[leader,:,0].reshape(-1,1), kernel_L=kernel_L, horizon=horizon, alpha=alpha)
                else:
                    followers_lst = self.graph_dict[leader]
                    self.agents_dict[leader] = OptComplementarySetPricingAgent(n_actions, len(followers_lst),
                        self.margins[leader, :].reshape(self.n_actions, 1), self.margins[followers_lst[0], :].reshape(self.n_actions, 1), 
                        kernel_L, horizon, alpha)
        else:
            self.known_graph = False
            raise NotImplementedError("Unknown graph mode not implemented yet.")


    def pull(self):
        action = -1 * np.ones((self.n_products))
        if self.known_graph: 
            for leader in self.graph_dict.keys():
                if len(self.graph_dict[leader]) == 0:
                    action[leader] = self.agents_dict[leader].pull()
                else:
                    followers_lst = self.graph_dict[leader]
                    leader_price, followers_prices = self.agents_dict[leader].pull()
                    action[leader] = leader_price
                    action[followers_lst] = followers_prices
        else:
            raise NotImplementedError("Unknown graph mode not implemented yet.")
        return action
    

    def update(self, sales, impressions):
        if self.known_graph: 
            for leader in self.graph_dict.keys():
                if len(self.graph_dict[leader]) == 0:
                    self.agents_dict[leader].update(np.sum(sales[leader]), np.sum(impressions[leader]))
                else:
                    followers_lst = self.graph_dict[leader]
                    self.agents_dict[leader].update(sales[leader, :].ravel(), impressions[leader, :].ravel(), 
                                                    sales[followers_lst, :], impressions[followers_lst, :])
        else:
            raise NotImplementedError("Unknown graph mode not implemented yet.")