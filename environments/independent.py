import numpy as np


class IndependentPricingEnvironment:
    """
    Simulates a multi-product pricing environment with independent demands.
    This environment models a scenario (e.g., e-commerce) where a seller
    must set prices for multiple products. The demand for each product is
    independent of the prices of other products.

    The simulation proceeds in discrete steps. In each step:
    1.  A random number of users (potential customers) arrives for each
        product, sampled uniformly from a specified range.
    2.  The agent provides a margin (price) for each product.
    3.  A binomial trial is run for each product to determine the number
        of sales, based on the number of users and the demand probability
        associated with the chosen margin.
    """


    def __init__(self, n_products, n_actions, margins, demands, user_ranges, alpha, seed=0):
        """
        Initializes the pricing environment.

        Args:
            n_products (int): The number of independent products.
            n_actions (int): The number of discrete price/margin actions
                             available for each product.
            margins (numpy.ndarray): A 2D array of shape (n_products, n_actions)
                                     specifying the margin (profit) for each
                                     product-action pair.
            demands (numpy.ndarray): A 2D array of shape (n_products, n_actions)
                                     specifying the demand probability [0, 1]
                                     for each product-action pair.
            user_ranges (numpy.ndarray): A 2D array of shape (n_products, 2)
                                         where each row is `[min_users, max_users]`
                                         defining the range of potential
                                         customers in a step for each product.
            alpha (float): The objective weighting parameter, between 0 and 1.
                           `0` = Profit, `1` = Revenue.
            seed (int, optional): Seed for the random number generator.
                                  Defaults to 0.
        
        Raises:
            ValueError: If input shapes are inconsistent or values are
                        outside their expected ranges.
        """
        
        if not (isinstance(demands, np.ndarray) and demands.shape == (n_products, n_actions)):
            raise ValueError("Shape of 'demands' not coherent with n_products and n_actions.")
        if not (isinstance(margins, np.ndarray) and margins.shape == (n_products, n_actions)):
            raise ValueError("Shape of 'margins' not coherent with n_products and n_actions.")
        if not ((demands <= 1).all() and (demands >= 0).all()):
            raise ValueError("All 'demands' must be in the range [0, 1].")
        if not (isinstance(user_ranges, np.ndarray) and user_ranges.shape == (n_products, 2)):
            raise ValueError("Shape of 'user_ranges' not coherent with n_products.")
        if not (user_ranges > 0).all():
            raise ValueError("'user_ranges' must be strictly greater than zero.")
        if not (user_ranges[:, 0] <= user_ranges[:, 1]).all():
            raise ValueError("In 'user_ranges', min (col 0) must be <= max (col 1).")
        if not (alpha >= 0 and alpha <= 1):
            raise ValueError("'alpha' must be in the range [0, 1].")

        self.alpha = alpha # 1 revenue, 0 profit
        self.n_products = n_products
        self.n_actions = n_actions
        self.demands = demands
        self.margins = margins
        self.user_ranges = user_ranges
        
        # Create a reverse lookup (margin value -> action index) for each product
        self.margins_to_idx_lst = []
        for prod in range(n_products):
            self.margins_to_idx_lst.append({margins[prod, act]: act for act in range(n_actions)})
        
        self.reset(seed)


    def step(self, margins):
        """
        Simulates one step of the environment based on the chosen margins.

        Args:
            margins (numpy.ndarray): A 1D array of shape (n_products,)
                containing the *margin value* (not index) chosen for each
                product.

        Returns:
            numpy.ndarray: A 2D array of shape (n_products, 2), where each
                row is `[number_of_sales, number_of_users]`.

        Raises:
            ValueError: If the `margins` action has an incorrect shape.
            KeyError: If a margin value in `margins` is not found in the
                      environment's predefined `margins` matrix.
        """
        if not (isinstance(margins, np.ndarray) and margins.ndim == 1 and
                margins.shape[0] == self.n_products):
            raise ValueError(f"The action 'margins' must be a 1D array of "
                             f"dimension n_products ({self.n_products}). "
                             f"Got shape {margins.shape}.")

        # [sales, users]
        sales = np.zeros((self.n_products, 2), dtype=int)
        
        # 1. Determine the number of users (opportunities) for each product
        sales[:, 1] = (np.random.uniform(0, 1, self.n_products) * (
            self.user_ranges[:, 1] - self.user_ranges[:, 0]) + 
            self.user_ranges[:, 0]).astype(np.int64)
        
        # 2. Look up the demand probability for the chosen margin
        demand_probs = -1 * np.ones(self.n_products)
        for i in range(self.n_products):
            action_idx = self.margins_to_idx_lst[i][margins[i]]
            demand_probs[i] = self.demands[i, action_idx]
        
        # 3. Simulate sales using a binomial draw
        # (n_trials = num_users, p = demand_prob)
        sales[:, 0] = np.random.binomial(sales[:, 1], demand_probs)
        
        return sales

        
    def compute_optimal_actions_and_values(self):
        """
        Computes the "oracle" optimal policy and expected values.

        This calculation is based on the weighted objective:
        `ExpectedValue = Demand * (alpha + Margin)`

        Returns:
            tuple: A tuple `(optimal_indices, optimal_values, all_values)`
                - **optimal_indices (numpy.ndarray)**: 1D array (n_products,)
                  of the *indices* of the best action for each product.
                - **optimal_values (numpy.ndarray)**: 1D array (n_products,)
                  of the *expected value* of the best action for each product.
                - **all_values (numpy.ndarray)**: 2D array (n_products, n_actions)
                  of the expected value for all possible actions.
        """
        # Calculate the expected objective value for all product-action pairs
        vals = np.multiply(self.demands, (self.alpha + self.margins))
        
        # Find the best action index and its value for each product
        optimal_indices = np.argmax(vals, axis=1)
        optimal_values = np.max(vals, axis=1)
        
        return optimal_indices, optimal_values, vals
    

    def reset(self, seed=0):
        """
        Resets the random number generator seed for reproducibility.

        Args:
            seed (int, optional): The seed to use. Defaults to 0.
        """
        np.random.seed(seed)
