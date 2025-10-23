import numpy as np
from agents.bandits.gaussianprocess import IGPUCB



class OptIndepPricingAgent:


    def __init__(self, actions, kernel_L, horizon, alpha, iterative=True):
        self.actions = actions # (action_number, action_dimension)
        self.action_dim = self.actions.shape[1]
        self.n_actions = self.actions.shape[0]
        assert actions.ndim == 2 and actions.shape[0] > 1 and actions.shape[1] == 1, "actions must be a 2D array with one column"
        assert alpha >= 0 and alpha <= 1, "alpha must be in [0, 1]"
        self.alpha = alpha # 1 revenue, 0 profit
        self.iterative = iterative
        self.bandit_agent = IGPUCB(self.n_actions, 1, actions, kernel_L, 0.25, 1, 1/horizon, het=True)
        self.last_action = None


    def pull(self):
        demand_ucbs = self.bandit_agent.get_optimistic_estimates()
        obj_ucbs = (self.alpha + self.actions.ravel()) * demand_ucbs
        choice = np.argmax(obj_ucbs)
        self.last_action = self.actions[choice, :]
        return float(self.last_action)

    
    def update(self, sales, impressions):
        if self.iterative:
            self.bandit_agent.update_complete(self.last_action, float(sales/impressions), sample_weight=int(impressions))
        else:
            raise NotImplementedError("Non-iterative mode not implemented yet.")
