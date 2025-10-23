from ortools.linear_solver import pywraplp
import numpy as np


def complementary_products(V_mx, verbose=False):
    """
    Solves a binary optimization problem to find optimal pairings between
    "base" products and "complementary" products to maximize total value.

    This function uses a Mixed-Integer Programming (MIP) solver (SCIP) from
    Google OR-Tools. Given an n x n value matrix `V_mx`, it partitions
    `n` items into two sets: Base products and Complementary products

    The goal is to maximize the total value from pairings, defined by the
    input matrix `V_mx`.

    Args:
        V_mx (numpy.ndarray): An `n x n` square matrix where `V_mx[i, j]`
                              represents the value of pairing base product `i`
                              with complementary product `j`.
        verbose (bool, optional): If `True`, prints solver status, objective
                                  value, and the resulting decision variables.
                                  Defaults to `False`.

    Returns:
        tuple: A tuple `(x_values, y_values)`:
            - x_values (numpy.ndarray): An `n x n` binary matrix.
            - y_values (numpy.ndarray): An `n`-element binary vector.

    Raises:
        ValueError: If `V_mx` is not a 2-dimensional or not a square matrix.
        RuntimeError: If the SCIP solver is not available.
    """

    if not isinstance(V_mx, np.ndarray):
        raise ValueError(f"Input V_mx must be a numpy array. Got {type(V_mx)}")
    if V_mx.ndim != 2:
        raise ValueError(f"Solver Error: Matrix must be 2-dimensional. Got {V_mx.ndim} dimensions.")
    if V_mx.shape[0] != V_mx.shape[1]:
        raise ValueError(f"Solver Error: Matrix must be square. Got shape {V_mx.shape}")

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("SCIP solver not available. Check OR-Tools installation.")
    
    n = V_mx.shape[0]
    P = range(n)
    
    x = {}
    y = {}
    for i in P:
        for j in P:
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")
        y[i] = solver.BoolVar(f"y_{i}")
    
    solver.Maximize(solver.Sum(x[i, j] * V_mx[i, j] for i in P for j in P))
    
    for i in P:
        solver.Add(solver.Sum(x[i, j] for j in P if j != i) >= 1 - n * y[i])
        solver.Add(solver.Sum(x[j, i] for j in P if j != i) <= n * y[i])
        solver.Add(x[i, i] <= 1 + n * y[i])
        solver.Add(x[i, i] >= 1 - n * y[i])
        solver.Add(solver.Sum(x[j, i] for j in P) >= 1 - n * (1 - y[i]))
        solver.Add(solver.Sum(x[j, i] for j in P) <= 1 + n * (1 - y[i]))
        solver.Add(solver.Sum(x[i, j] for j in P if j != i) <= n * (1 - y[i]))

    if verbose:
        print("Starting Solver with input matrix: ")
        print(V_mx)
    
    status = solver.Solve()

    x_values = np.zeros((n, n), dtype=int)
    y_values = np.zeros(n, dtype=int)

    if verbose:
        if status == pywraplp.Solver.OPTIMAL:
            print("Solver Status: OPTIMAL")
            print("Objective value:", solver.Objective().Value())
        else:
            print("Solver did not find an optimal solution.")
    
    for i in P:
        for j in P:
            x_values[i, j] = int(x[i, j].solution_value())
    
    y_values = np.array([int(y[i].solution_value()) for i in P])
        
    if verbose:
        print("\nDecision Variables (x[i][j]):")
        print(x_values)
        print("\nBinary Variables y:")
        print(y_values)

    return x_values, y_values