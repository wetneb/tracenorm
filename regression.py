
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

def next_tracenorm_guess(inputs, outputs, lmbd, mu, current_A):
    # Computes the next estimate of A using the first gradient algorithm
    p = np.shape(inputs)[0]
    q = np.shape(inputs)[1]
    r = np.shape(outputs)[1]
    A = current_A

    inputst = np.transpose(inputs)
    utu = np.dot(inputst, inputs)
    uv = np.dot(inputst, outputs)

    gradient = np.dot(utu,A) - uv
    C = A - mu * gradient
    U, s, V = np.linalg.svd(C)

    s = s - (lmbd/2)*np.ones(np.shape(s)[0])
    sz = np.array([s, np.zeros(np.shape(s)[0])])
    final_s = sz.max(0)
    lu = np.shape(U)[1]
    lv = np.shape(V)[0]
    S = np.zeros((lu, lv), dtype=complex)
    rk = min(lu, lv)
    S[:rk,:rk] = np.diag(final_s)
    return np.dot(U, np.dot(S, V))


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
        A = next_tracenorm_guess(inputs, outputs, lmbd, stepsize, A)

    return A, costs

def costfunction_tracenorm(inputs, outputs, A, lmbd):
    return fitness(inputs, outputs, A) + lmbd*tracenorm(A)

def intermediate_cost(inputs, outputs, new_A, old_A, lmbd, mu):
    diff_A = new_A - old_A
    grad_f = 2* (np.dot(inputs.transpose(), np.dot(inputs,old_A)) - np.dot(inputs.transpose(), outputs))
    return (fitness(inputs, outputs, old_A) +
            np.trace(np.dot(diff_A.transpose(), grad_f)) + (mu/2)*frobenius_norm_squared(diff_A))

def accelerated_tracenorm(inputs, outputs, lmbd, iterations, L0, gamma):
    L = L0

    p = np.shape(inputs)[0]
    q = np.shape(inputs)[1]
    r = np.shape(outputs)[1]
    A = np.random.rand(q,r)

    costs = []

    for i in range(iterations):
        # Cost tracking
        costs.append([fitness(inputs, outputs, A),
                      lmbd* tracenorm(A)])

        next_A = next_tracenorm_guess(inputs, outputs, lmbd, L, A)
        while (costfunction_tracenorm(inputs, outputs, next_A, lmbd) > intermediate_cost(inputs, outputs, next_A, A, lmbd, L)):
                 L = gamma*L
                 next_A = next_tracenorm_guess(inputs, outputs, lmbd, L, A)
        
        A = next_A

    return A, costs

