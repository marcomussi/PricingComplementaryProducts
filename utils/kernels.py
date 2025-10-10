import numpy as np



def kernel_rbf(a, b, L): 
    """
    Radial Basis Function Kernel 
    Lower "L" means that we are considering smoother functions, acting as a Lipschitz constant
    """
    
    output = -1 * np.ones((a.shape[0], b.shape[0]))
    
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            output[i, j] = np.power(np.linalg.norm(a[i, :] - b[j, :], 2), 2)
    
    return np.exp(- L * output)
s