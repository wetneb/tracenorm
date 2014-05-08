
import numpy as np
import matplotlib.pyplot as plt
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
    C = A - (1/mu) * gradient
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

def fitness_gradient(inputs, outputs, A):
    return 2* (np.dot(inputs.transpose(), np.dot(inputs,A)) - np.dot(inputs.transpose(), outputs))

# This quantity is the "P_mu" from the article
def intermediate_cost(inputs, outputs, new_A, old_A, mu):
    diff_A = new_A - old_A
    grad_f = fitness_gradient(inputs, outputs, old_A)
    return (fitness(inputs, outputs, old_A) +
            np.trace(np.dot(diff_A.transpose(), grad_f)) +
            (mu/2)*frobenius_norm_squared(diff_A))

def frobenius_norm(A):
    return math.sqrt(frobenius_norm_squared(A))

def gradient_tracenorm(inputs, outputs, lmbd, iterations):
    # Epsilon is here to ensure that the Lipschitz constant is big enough
    # (because the expression of L is tight)
    epsilon = 0.05
    L = (1+epsilon)*2*math.sqrt(frobenius_norm_squared(np.dot(inputs.transpose(), inputs)))
    print "Computed L: "+str(L)

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

        if(fitness(inputs, outputs, next_A) > intermediate_cost(inputs, outputs, next_A, A, L)):
            print "Numerical error detected."
            break
       
        A = next_A

    return A, costs

def extended_gradient_tracenorm(inputs, outputs, lmbd, iterations, initial=0):
    # Epsilon is here to ensure that the Lipschitz constant is big enough
    # (because the expression of L is tight)
    epsilon = 0.05
    L_bound = (1+epsilon)*2*math.sqrt(frobenius_norm_squared(np.dot(inputs.transpose(), inputs)))
    L = 1
    gamma = 1.5

    p = np.shape(inputs)[0]
    q = np.shape(inputs)[1]
    r = np.shape(outputs)[1]
    if type(initial) == int:
        A = np.random.rand(q,r)
    else:
        A = initial

    costs = []

    for i in range(iterations):
        # Cost tracking
        costs.append(fitness(inputs, outputs, A) +
                      lmbd* tracenorm(A))

        next_A = next_tracenorm_guess(inputs, outputs, lmbd, L, A)

        while(fitness(inputs, outputs, next_A) > intermediate_cost(inputs, outputs, next_A, A, L)):
            if L > L_bound:
                print "Numerical error detected at iteration "+str(i)
                break

            L = gamma * L
            next_A = next_tracenorm_guess(inputs, outputs, lmbd, L, A)
       
        A = next_A

    print "Final L: "+str(L)
    print "Bound on L: "+str(L_bound)
    return A, costs

def accelerated_gradient_tracenorm(inputs, outputs, lmbd, iterations, initial=0):
    L = 1
    gamma = 1.5
    alpha = 1

    epsilon = 0.1
    L_bound = (1+epsilon)*2*math.sqrt(frobenius_norm_squared(np.dot(inputs.transpose(), inputs)))

    p = np.shape(inputs)[0]
    q = np.shape(inputs)[1]
    r = np.shape(outputs)[1]
    if type(initial) == int:
        W = np.random.rand(q,r)
    else:
        W = initial
    Z = W

    costs = []

    for i in range(iterations):
        # Cost tracking
        costs.append(fitness(inputs, outputs, W)+
                      lmbd* tracenorm(W))

        next_W = next_tracenorm_guess(inputs, outputs, lmbd, L, Z)

        while(fitness(inputs, outputs, next_W) > intermediate_cost(inputs, outputs, next_W, Z, L)):
            if L > L_bound:
                print "Numerical error detected at iteration "+str(i)
                break

            L = gamma * L
            next_W = next_tracenorm_guess(inputs, outputs, lmbd, L, Z)

        previous_W = W
        W = next_W
        previous_alpha = alpha
        alpha = (1 + math.sqrt(1 + 4*alpha*alpha))/2
        Z = W + ((alpha - 1)/alpha)*(W - previous_W)

    print "Final L: "+str(L)
    print "Bound on L: "+str(L_bound)
    return W, costs


