import numpy as np



def generate_curves(num_curves, num_actions=10, base_variation=2, min_start=0.6, max_start=1.0):
    """
    Generates demand curves
    """

    L = base_variation / num_actions
    
    demands = np.zeros((num_curves, num_actions))
    demands[:, 0] = np.random.uniform(min_start, max_start, num_curves)
    demands[:, 1:] = np.random.uniform(0, -L, (num_curves, num_actions-1))
    demands = np.maximum(np.cumsum(demands, axis=1), 0)

    return demands



def generate_user_ranges(num_users, min_range, max_range, low_to_high_mix=0.5):
    """
    Generates user ranges for low and high selling products
    """
    
    user_ranges = np.zeros((num_users, 2), dtype=int)
    low_mask = np.random.uniform(0, 1, num_users) > low_to_high_mix
    user_ranges[low_mask, 0] = min_range
    user_ranges[np.logical_not(low_mask), 1] = max_range
    user_ranges[low_mask, 1] = np.random.randint(low=min_range, high=max_range, size=np.sum(low_mask))
    user_ranges[np.logical_not(low_mask), 0] = np.random.randint(low=min_range, high=max_range, size=num_users-np.sum(low_mask))
    
    return user_ranges




def generate_graph(num_products, num_clusters):
    """
    Generates random clusters and for each trainee product a probability of concurrency
    """
    
    trainers = np.random.choice(num_products, num_clusters, replace=False)
    
    clusters_dict = {trainers[i] : [] for i in range(len(trainers))}

    to_assign = []
    for i in range(num_products):
        if i not in list(clusters_dict.keys()):
            to_assign.append(i)
    
    assignments = np.random.choice(list(clusters_dict.keys()), num_products - num_clusters)
    
    for i in range(len(to_assign)):
        clusters_dict[assignments[i]].append(to_assign[i])

    probabilities = np.random.uniform(0, 1, num_products)

    return clusters_dict, probabilities
