import numpy as np



def incr_inv(A_inv, B, C, D):
    """
    Incremental inverse of blockwise matrix [[A, B], [C, D]]
    We consider only square matrices, no pseudo-inverses
    """

    n = A_inv.shape[0] # original number of rows/cols
    n_add = C.shape[0] # number of rows/cols to add
    
    assert A_inv.shape == (n, n), "incremental matrix inverse: error in input dimensions"
    assert B.shape == (n, n_add), "incremental matrix inverse: error in input dimensions"
    assert C.shape == (n_add, n), "incremental matrix inverse: error in input dimensions"
    assert D.shape == (n_add, n_add), "incremental matrix inverse: error in input dimensions"

    temp = np.linalg.solve(D - C @ A_inv @ B, np.eye(n_add))

    block1 = A_inv + A_inv @ B @ temp @ C @ A_inv
    block2 = - A_inv @ B @ temp
    block3 = - temp @ C @ A_inv
    block4 = temp

    res1 = np.concatenate((block1, block2), axis=1)
    res2 = np.concatenate((block3, block4), axis=1)

    return np.concatenate((res1, res2), axis=0)