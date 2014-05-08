
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

def intermediate_cost(inputs, outputs, new_A, old_A, lmbd, mu):
    diff_A = new_A - old_A
    grad_f = fitness_gradient(inputs, outputs, old_A)
    return (fitness(inputs, outputs, old_A) +
            np.trace(np.dot(diff_A.transpose(), grad_f)) +
            (mu/2)*frobenius_norm_squared(diff_A) +
            (tracenorm(new_A)))

def frobenius_norm(A):
    return math.sqrt(frobenius_norm_squared(A))

def accelerated_tracenorm(inputs, outputs, lmbd, iterations, L0, gamma):
    L = L0

    L = 4*math.sqrt(frobenius_norm_squared(np.dot(inputs.transpose(), inputs)))
    print "New L: "+str(L)

    p = np.shape(inputs)[0]
    q = np.shape(inputs)[1]
    r = np.shape(outputs)[1]
    A = np.random.rand(q,r)

    costs = []

    for i in range(iterations):
        debugging = True #i == 1 or i == 100
        if debugging:
            print "Iteration "+str(i)
            print "A = "
            print A

        # Cost tracking
        costs.append([fitness(inputs, outputs, A),
                      lmbd* tracenorm(A)])

        if debugging:
            print "Fitness: "+str(fitness(inputs, outputs, A))
            print "Tracenorm: "+str(tracenorm(A))

        next_A = next_tracenorm_guess(inputs, outputs, lmbd, L, A)
        if debugging:
            print "Current L: "+str(L)
            print "Next_A: "
            print next_A
            print "costfunction_tracenorm on next_A: "+str(costfunction_tracenorm(inputs, outputs, next_A, lmbd))
            print "intermediate_cost: "+str(intermediate_cost(inputs, outputs, next_A, A, lmbd, L))
        if (costfunction_tracenorm(inputs, outputs, next_A, lmbd) > intermediate_cost(inputs, outputs, next_A, A, lmbd, L)):
            print "Numerical error detected."
            print "Diagnosis:"
            print "1/ Cost function:"
            print "- Fitness on next_A: "+str(fitness(inputs, outputs, next_A))
            print "- Trace Norm of next_A: "+str(tracenorm(next_A))
            print "- Lambda: "+str(lmbd)
            print "TOTAL: "+str(costfunction_tracenorm(inputs, outputs, next_A, lmbd))
            print "2/ Q_L:"
            print "- fitness of old_A: "+str(fitness(inputs, outputs, A))
            gradient = 2*(np.dot(inputs.transpose(), np.dot(inputs, A)) - np.dot(inputs.transpose(), outputs))
            print "- gradient"
            print gradient
            print "- first-order term: "+str(np.trace(np.dot(next_A.transpose() - A.transpose(), gradient)))
            print "- difference between next_A and A: "
            print (next_A - A)
            print "- quadratic term: "+str((L/2)*frobenius_norm_squared(A - next_A))
            print "- Trace Norm of next_A: "+str(tracenorm(next_A))
            print "TOTAL: "+str(intermediate_cost(inputs, outputs, next_A, A, lmbd, L))
            print "Lipschitz check:"
            diffgrad = fitness_gradient(inputs,outputs,next_A) - fitness_gradient(inputs,outputs,A)
            print "Difference between gradients (norm): "+str(frobenius_norm(diffgrad))
            print "Difference between A and next_A: "+str(frobenius_norm(A - next_A))
            print "Difference between A and next_A times L: "+str(L*(frobenius_norm(A - next_A)))

            t = np.linspace(-0.2,1.2)
            At = []
            fitness_t = []
            grad_t = []
            fa = fitness(inputs,outputs,A)
            grad_fa = np.trace(np.dot(fitness_gradient(inputs,outputs,A).transpose(), next_A - A))
            n2diff = frobenius_norm_squared(A - next_A)
            for ti in t:
                point = A + ti*(next_A - A)
                At.append(point)
                fitness_t.append([fitness(inputs,outputs,point), fa + ti*grad_fa, fa+ti*grad_fa + (L/2)*ti*ti*n2diff])

            At = np.array(At)
            fitness_t = np.array(fitness_t)
            grad_t = np.array(grad_t)
            fitness_t
            plt.plot(t, fitness_t)
            plt.show()
            
            break
#                 L = gamma*L
#                 if debugging:
#                     print "Current L: "+str(L)
#                     print "Next_A: "
#                     print next_A
#                     print "costfunction_tracenorm on next_A: "+str(costfunction_tracenorm(inputs, outputs, next_A, lmbd))
#                     print "intermediate_cost: "+str(intermediate_cost(inputs, outputs, next_A, A, lmbd, L))
#                 next_A = next_tracenorm_guess(inputs, outputs, lmbd, L, A)
        
        A = next_A
        if debugging:
            print "\n"

    return A, costs

