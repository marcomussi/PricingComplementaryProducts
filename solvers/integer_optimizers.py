from ortools.linear_solver import pywraplp
import numpy as np

def complementary_products(V_mx, verbose=False):

    assert V_mx.ndim == 2, "Solver Error: Matrix must be 2-dimensional"
    assert V_mx.shape[0] == V_mx.shape[1], "Solver Error: Matrix must be square"
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
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
        print("Starting Solver with inout matrix: ")
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