import numpy as np
from agents.bandits.gaussianprocess import IGPUCB



class OptComplementarySetPricingAgent:


    def __init__(self, n_actions, n_followers, margins_leader, margins_followers, kernel_L, horizon, alpha, 
                 cost_leader=None, cost_followers=None, iterative=True):
        assert margins_leader.shape == (n_actions, 1), "Shape of the margins_leader not coherent"
        assert margins_followers.shape == (n_actions, 1), "Shape of the margins_followers not coherent" # all the same
        assert alpha >= 0 and alpha <= 1, "alpha must be in [0, 1]"  
        self.n_followers = n_followers  
        self.n_actions = n_actions
        self.margins_leader = margins_leader
        self.margins_followers = margins_followers
        self.kernel_L = kernel_L
        self.horizon = horizon
        self.alpha = alpha # 1 revenue, 0 profit
        self.iterative = iterative
        self.leader_bandit_agent = IGPUCB(self.n_actions, 1, self.margins_leader, kernel_L, 0.25, 1, 1/horizon, het=True)
        self.LEAD = "lead"
        self.NOLEAD = "no_lead"
        self.followers_bandit_agent = []
        for i in range(n_followers):
            self.followers_bandit_agent.append({
                self.LEAD : IGPUCB(self.n_actions, 1, self.margins_followers, kernel_L, 0.25, 1, 1/horizon, het=True), 
                self.NOLEAD : IGPUCB(self.n_actions, 1, self.margins_followers, kernel_L, 0.25, 1, 1/horizon, het=True)
            })
        self.last_action = None
        self.prob_concurrency_followers = np.zeros((self.n_followers))
        self.w_concurrency_followers = np.zeros((self.n_followers))
        self.leader_avg_impressions = 0
        self.followers_avg_impressions = np.zeros((self.n_followers))
        self.iterations_count = 0
        if cost_leader is not None: 
            assert isinstance(cost_leader, float) and cost_leader > 0, "error in cost_leader"
            self.cost_leader = cost_leader
        else:
            self.cost_leader = 1
        if cost_followers is not None:
            assert cost_followers.shape == (self.n_followers) and np.all(cost_followers > 0), "error in cost_leader"
            self.cost_followers = cost_followers
        else:
            self.cost_followers = np.ones((self.n_followers))


    def pull(self):
        demand_ucbs_leader, demand_mu_leader, _ = self.leader_bandit_agent.get_optimistic_estimates(return_raw_info=True)
        obj_ucbs_leader = self.cost_leader * self.leader_avg_impressions * (
            self.alpha + self.margins_leader.ravel()) * demand_ucbs_leader
        demand_ucbs_followers = {}
        demand_ucbs_followers[self.LEAD] = -1 * np.ones((self.n_followers, self.n_actions))
        demand_ucbs_followers[self.NOLEAD] = -1 * np.ones((self.n_followers, self.n_actions))
        for i in range(self.n_followers):
            demand_ucbs_followers[self.LEAD][i, :] = self.followers_bandit_agent[i][self.LEAD].get_optimistic_estimates()
            demand_ucbs_followers[self.NOLEAD][i, :] = self.followers_bandit_agent[i][self.NOLEAD].get_optimistic_estimates()
        opt_mx = -1 * np.ones((self.n_actions, self.n_actions))
        for leader_act_i, _ in enumerate(list(self.margins_leader.ravel())):
            for followers_act_i, followers_act in enumerate(list(self.margins_followers.ravel())):
                opt_mx[leader_act_i, followers_act_i] = obj_ucbs_leader[leader_act_i]
                for fl_i in range(self.n_followers):
                    demand_composite = self.prob_concurrency_followers[fl_i] * demand_mu_leader[leader_act_i] * demand_ucbs_followers[self.LEAD][fl_i, followers_act_i] + ((1 - self.prob_concurrency_followers[fl_i]) + self.prob_concurrency_followers[fl_i] * (1 - demand_mu_leader[leader_act_i])) * demand_ucbs_followers[self.NOLEAD][fl_i, followers_act_i]
                    opt_mx[leader_act_i, followers_act_i] += self.cost_followers[fl_i] * (self.alpha + followers_act) * self.followers_avg_impressions[fl_i] * demand_composite
        opt_action_idx_leader, opt_action_idx_followers = np.unravel_index(np.argmax(opt_mx), opt_mx.shape)
        self.last_action = self.margins_followers[opt_action_idx_followers] * np.ones((self.n_followers + 1))
        self.last_action[0] = self.margins_leader[opt_action_idx_leader]
        return self.last_action[0], self.last_action[1]
    

    def update(self, sales_leader_vect, impressions_leader_vect, sales_followers_mx, impressions_followers_mx):
        if self.iterative:
            assert sales_leader_vect.ndim == 1, "sales_leader_mx shape error"
            assert impressions_leader_vect.ndim == 1, "impressions_leader_mx must be an integer"
            assert sales_followers_mx.ndim == 2 and sales_followers_mx.shape[0] == self.n_followers, "sales_followers shape error"
            assert impressions_followers_mx.ndim == 2 and impressions_followers_mx.shape[0] == self.n_followers, "impressions_followers shape error"
            assert sales_leader_vect.shape == impressions_leader_vect.shape and \
                    sales_leader_vect.shape[0] == sales_followers_mx.shape[1] and \
                    sales_followers_mx.shape == impressions_followers_mx.shape, "error in shapes"
            probs_vector = np.mean(np.multiply(impressions_followers_mx, impressions_leader_vect), axis=1).ravel()
            probs_weight = impressions_leader_vect.shape[0]
            self.prob_concurrency_followers = (self.prob_concurrency_followers * self.w_concurrency_followers + probs_vector * probs_weight) / (probs_weight + self.w_concurrency_followers)
            self.w_concurrency_followers = self.w_concurrency_followers + probs_weight
            self.leader_bandit_agent.update_complete(self.last_action[0], float(np.sum(sales_leader_vect)/ np.sum(impressions_leader_vect)), sample_weight=int(np.sum(impressions_leader_vect)))
            self.leader_avg_impressions = (self.leader_avg_impressions * self.iterations_count + np.sum(impressions_leader_vect)) / (self.iterations_count + 1)
            mask_leader_sales = sales_leader_vect == 1
            for fl_i in range(self.n_followers):
                sales_with_leader = np.sum(sales_followers_mx[fl_i, mask_leader_sales])
                impressions_with_leader = np.sum(impressions_followers_mx[fl_i, mask_leader_sales])
                sales_no_leader = np.sum(sales_followers_mx[fl_i, np.logical_not(mask_leader_sales)])
                impressions_no_leader = np.sum(impressions_followers_mx[fl_i, np.logical_not(mask_leader_sales)])
                if impressions_with_leader > 0:
                    self.followers_bandit_agent[fl_i][self.LEAD].update_complete(self.last_action[fl_i+1], float(sales_with_leader/impressions_with_leader), sample_weight=int(impressions_with_leader))
                if impressions_no_leader > 0:
                    self.followers_bandit_agent[fl_i][self.NOLEAD].update_complete(self.last_action[fl_i+1], float(sales_no_leader/impressions_no_leader), sample_weight=int(impressions_no_leader))
                self.followers_avg_impressions = (self.followers_avg_impressions * self.iterations_count + np.sum(impressions_followers_mx[fl_i])) / (self.iterations_count + 1)
            self.iterations_count += 1
        else:
            raise NotImplementedError("update() cannot be used in non-iterative mode yet.")
