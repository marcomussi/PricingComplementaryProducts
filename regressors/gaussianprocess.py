import numpy as np
from utils.kernels import kernel_rbf
from utils.math import incr_inv



class GaussianProcessRegressorRBF: 


    def __init__(self, kernel_L, sigma_sq_process, input_dim=1, keep_info_gain_estimate=False):
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim
        self.keep_info_gain_estimate = keep_info_gain_estimate
        self.n_samples = 0
        self.info_gain = None
        self.x_vect = None
        self.y_vect = None


    def load_data(self, x, y):
        self.n_samples = x.shape[0]
        self.x_vect = np.array([x]).reshape(self.n_samples, self.input_dim)
        self.y_vect = np.array([y]).reshape(self.n_samples, 1)
        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.sigma_sq_process * np.eye(self.n_samples)
        self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.n_samples))
        if self.keep_info_gain_estimate: 
            _, value = np.linalg.slogdet(self.K_matrix/self.sigma_sq_process)
            self.info_gain = 0.5 * value
    

    def add_sample(self, x, y):
        if isinstance(x, np.ndarray):
            assert x.ndim == 2 and x.shape == (1, self.input_dim), "add_sample() function: Error in input"
        if isinstance(y, np.ndarray):
            assert (y.ndim == 1 and y.shape == (1, )) or (y.ndim == 2 and y.shape == (1, 1)), "add_sample() function: Error in input"
        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)
        self.n_samples += 1
        if self.x_vect is None:
            self.x_vect = x
            self.y_vect = y
            self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + self.sigma_sq_process
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(1))
            if self.keep_info_gain_estimate:
                self.info_gain = (0.5 * np.log(1 + 1 / self.sigma_sq_process))
        else:
            self.x_vect = np.vstack((self.x_vect, x))
            self.y_vect = np.vstack((self.y_vect, y))
            K_star = kernel_rbf(self.x_vect[:-1, :].reshape(-1, self.input_dim), self.x_vect[-1, :].reshape(1, self.input_dim), self.kernel_L)
            elem = kernel_rbf(self.x_vect[-1, :].reshape(1, self.input_dim), self.x_vect[-1, :].reshape(1, self.input_dim), self.kernel_L)
            if self.keep_info_gain_estimate:
                sigma_i = elem - K_star.T @ self.K_matrix_inv @ K_star
                self.info_gain += (0.5 * np.log(1 + sigma_i / self.sigma_sq_process))
            elem = np.array(elem + self.sigma_sq_process).reshape(1, 1) 
            self.K_matrix = np.vstack((np.hstack((self.K_matrix, K_star)), np.hstack((K_star.T, elem))))
            self.K_matrix_inv = incr_inv(self.K_matrix_inv, K_star, K_star.T, elem)
            

    def compute(self, x):
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "compute() function: Error in input"
        n = x.shape[0]
        mu = np.zeros(n)
        sigma = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            sigma[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        return mu, sigma
    

    def get_info_gain(self):
        if self.keep_info_gain_estimate:
            return self.info_gain[0, 0] if isinstance(self.info_gain, np.ndarray) else self.info_gain
        else:
            raise ValueError("Info Gain not computed, use flag keep_info_gain_estimate=True during initialization")
    
    

class HeteroscedasticGaussianProcessRegressorRBF: 


    def __init__(self, kernel_L, sigma_sq_process, input_dim=1, one_sample_mod=False):
        self.kernel_L = kernel_L
        self.sigma_sq_process = sigma_sq_process  
        self.input_dim = input_dim
        self.one_sample_mod = one_sample_mod
        self.x_vect = None
        self.y_vect = None
        if one_sample_mod:
            self.num_samples = []


    def load_data(self, x, y, sigmasqs):
        if self.one_sample_mod:
            raise ValueError("load_data() cannot be used with one_sample_mod=True")
        n = x.shape[0]
        self.x_vect = np.array([x]).reshape(n, self.input_dim)
        self.y_vect = np.array([y]).reshape(n, 1)
        self.sigmasqs = np.array([sigmasqs]).reshape(n,)
        self.K_matrix = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L) + np.diag(self.sigmasqs)
        self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(n))

    
    def add_sample(self, x, y, sample_weight=1):
        if not self.one_sample_mod:
            raise ValueError("add_sample() cannot be used with one_sample_mod=False")
        if isinstance(x, np.ndarray):
            assert x.ndim == 2 and x.shape == (1, self.input_dim), "add_sample() function: Error in input"
        if isinstance(y, np.ndarray):
            assert (y.ndim == 1 and y.shape == (1, )) or (y.ndim == 2 and y.shape == (1, 1)), "add_sample() function: Error in input"
        x = np.array([x]).reshape(1, self.input_dim)
        y = np.array([y]).reshape(1, 1)
        if self.x_vect is None:
            self.x_vect = x
            self.y_vect = y
            self.K_matrix_noreg = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L)
            self.K_matrix = self.K_matrix_noreg + (self.sigma_sq_process / sample_weight)
            self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(1))
            self.num_samples = [sample_weight]
        else:
            matches = np.where(np.all(self.x_vect == x, axis=1))[0]
            if len(matches) > 0:
                pos_first_found = matches[0]
                self.y_vect[pos_first_found] = (self.y_vect[pos_first_found] * self.num_samples[pos_first_found] + (y * sample_weight)) / (self.num_samples[pos_first_found] + sample_weight)
                self.num_samples[pos_first_found] = self.num_samples[pos_first_found] + sample_weight
                self.K_matrix = self.K_matrix_noreg + np.diag(self.sigma_sq_process / np.array(self.num_samples))
                self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.K_matrix.shape[0]))
            else: 
                self.x_vect = np.vstack((self.x_vect, x))
                self.y_vect = np.vstack((self.y_vect, y))
                self.num_samples.append(sample_weight)
                self.K_matrix_noreg = kernel_rbf(self.x_vect, self.x_vect, self.kernel_L)
                self.K_matrix = self.K_matrix_noreg + np.diag(self.sigma_sq_process / np.array(self.num_samples))
                self.K_matrix_inv = np.linalg.solve(self.K_matrix, np.eye(self.K_matrix.shape[0]))


    def compute(self, x):
        assert x.ndim == 2, "compute() function: Error in input dimension"
        assert x.shape[1] == self.input_dim, "compute() function: Error in input dimension"
        n = x.shape[0]
        mu = np.zeros(n)
        sigmasq = np.zeros(n)
        for i in range(n):
            K_star = kernel_rbf(self.x_vect, x[i, :].reshape(1, self.input_dim), self.kernel_L)
            mu[i] = K_star.T @ self.K_matrix_inv @ self.y_vect
            sigmasq[i] = kernel_rbf(x[i, :].reshape(1, self.input_dim), x[i, :].reshape(1, self.input_dim), self.kernel_L) - K_star.T @ self.K_matrix_inv @ K_star
        return mu, sigmasq


    def get_info_gain(self):
        if self.one_sample_mod:
            temp = self.sigma_sq_process / np.array(self.num_samples)
        else:
            temp = self.sigmasqs
        D = np.diag(temp ** -0.5)
        _, value = np.linalg.slogdet(D @ (self.K_matrix - np.diag(temp)) @ D + np.eye(D.shape[0]))
        return 0.5 * value
