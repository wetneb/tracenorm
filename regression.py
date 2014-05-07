
import numpy as np
import math

def frobenius_norm_squared(A):
    A2 = np.multiply(A,A)
    return np.real(np.sum(A2))

def fitness(inputs, outputs, A):
    return frobenius_norm_squared(np.dot(inputs, A) - outputs)

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

    cost = [fitness(inputs, outputs, A), lmbd*frobenius_norm_squared(A)]

    return A, cost


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

    costs = []

    for i in range(iterations):
        # TODO: How to choose the step size?
        costs.append([frobenius_norm_squared(np.dot(inputs,A) - outputs),
                      lmbd* frobenius_norm_squared(A)])
        gradient = np.dot(utuid,A) - uv
        A = A - stepsize * gradient

    return A, costs

def tracenorm(A):
    U, s, V = np.linalg.svd(A)
    return sum(s)

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

    costs = []

    for i in range(iterations):
        # TODO: How to choose the step size?

        costs.append([frobenius_norm_squared(np.dot(inputs,A) - outputs),
                      lmbd* tracenorm(A)])

        gradient = np.dot(utu,A) - uv
        C = A - stepsize * gradient
        U, s, V = np.linalg.svd(C)

        debugging = i==1 or i == 500
        if debugging:
            print "i = "+str(i)
            print "SVD decomposition:"
            print U
            print s
            print V
        s = s - (lmbd/2)*np.ones(np.shape(s)[0])
        if debugging:
            print "S minus lambda"
            print s

        sz = np.array([s, np.zeros(np.shape(s)[0])])
        if debugging:
            print "sz"
            print sz

        final_s = sz.max(0)
        if debugging:
            print "final_s"
            print final_s

        lu = np.shape(U)[1]
        lv = np.shape(V)[0]
        S = np.zeros((lu, lv), dtype=complex)
        rk = min(lu, lv)
        S[:rk,:rk] = np.diag(final_s)
        if debugging:
            print "in a matrix"
            print S
            print "\n"
        A = np.dot(U, np.dot(S, V))

    return A, costs


