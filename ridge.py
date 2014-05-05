
import numpy as np

def regression(inputs, outputs, lmbd):
    """Implements ridge regression using the closed formula"""
    # Computes min_A || inputs * A - outputs ||^2 + lmbd * || A ||^2
    # Where || A ||^2 = tr (A^t A)
    #
    # the solution is A = (inputs^t * inputs + lmbd*I)^{-1} *  inputs^t * outputs
    
    p = np.shape(inputs)[0]
    q = np.shape(inputs)[1]
    lmbdid = np.diag(lmbd * np.ones(q))
    inputst = np.transpose(inputs)
    A = np.dot(np.dot(np.linalg.pinv(np.dot(inputst, inputs) + lmbdid), inputst), outputs)

    return A


