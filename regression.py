
import numpy as np

def ridge(inputs, outputs, lmbd):
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

def incremental_ridge(inputs, outputs, lmbd, iterations, stepsize):
    """Implements ridge regression using gradient descent"""
    # Computes min_A || inputs * A - outputs ||^2 + lmbd * || A ||^2
    # Where || A ||^2 = tr (A^t A)
    #
    # the solution is A = (inputs^t * inputs + lmbd*I)^{-1} *  inputs^t * outputs

    p = np.shape(inputs)[0]
    q = np.shape(inputs)[1]
    r = np.shape(outputs)[1]
    A = np.random.rand(q,r)

    lmbdid = np.diag(lmbd * np.ones(q))
    inputst = np.transpose(inputs)
    utu = np.dot(inputst, inputs)
    utuid = utu + lmbdid
    uv = np.dot(inputst, outputs)

    for i in range(iterations):
        # TODO: How to choose the step size?
        gradient = np.dot(utuid,A) - uv
        A = A - stepsize * gradient

    return A

def incremental_tracenorm(inputs, outputs, lmbd, iterations, stepsize):
    """ Implements trace norm regression using the first gradient descent from (Ji and Ye, 2009)"""
    # Computes min_A || inputs * A - outputs ||^2 + lmbd * || A ||*
    # Where || A ||* is the sum of the singular values of A

    p = np.shape(inputs)[0]
    q = np.shape(inputs)[1]
    r = np.shape(outputs)[1]
    A = np.random.rand(q,r)

    #unused: lmbdid = np.diag(lmbd * np.ones(q))
    inputst = np.transpose(inputs)
    utu = np.dot(inputst, inputs)
    #unused:  utuid = utu + lmbdid
    uv = np.dot(inputst, outputs)

    for i in range(iterations):
        # TODO: How to choose the step size?
        gradient = np.dot(utu,A) - uv
        C = A - stepsize * gradient
        U, s, V = np.linalg.svd(C)
        s = s - lmbd*np.ones(np.shape(s)[0])
        sz = np.array([s, np.zeros(np.shape(s)[0])])
        final_s = sz.max(0)
        lu = np.shape(U)[1]
        lv = np.shape(V)[0]
        S = np.zeros((lu, lv), dtype=complex)
        rk = min(lu, lv)
        S[:rk,:rk] = np.diag(final_s)
        A = np.dot(U, np.dot(S, V))

    return A




