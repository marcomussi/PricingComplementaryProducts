import numpy as np
from utils.kernels import kernel_rbf



class GaussianProcessRegressor: 


    def __init__(self, kernel_L, sigma_sq_process, input_dim):
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim


    def load_data(self, x, y):
        n = x.shape[0]
        self.x_vect = np.array([x]).reshape(n, self.input_dim)
        self.y_vect = np.array([y]).reshape(n, 1)
        self.K_matrix_inv = np.linalg.inv(kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.sigma_sq_process * np.eye(n))
    
    
    def compute(self, x):
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        n = x.shape[0]
        mu = np.zeros(n)
        sigma = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            sigma[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        return mu, sigma
    


class HeteroscedasticGaussianProcessRegressor: 


    def __init__(self, kernel_L, input_dim=1):
        self.kernel_L = kernel_L
        self.input_dim = input_dim


    def load_data(self, x, y, sigmas):
        n = x.shape[0]
        self.x_vect = np.array([x]).reshape(n, self.input_dim)
        self.y_vect = np.array([y]).reshape(n, 1)
        self.sigmas = np.array([sigmas]).reshape(n,)


    def compute(self, x):
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        n = x.shape[0]
        K = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + np.diag(self.sigmas)
        K_inv = np.linalg.inv(K)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ K_inv @ self.y_vect
            sigma[i] = 1 - K_star.T @ K_inv @ K_star
        return mu, sigma
