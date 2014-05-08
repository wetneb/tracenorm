import numpy as np
import matplotlib.pyplot as plt
import regression

inputs = np.array([ [1,1,4], [2,3,0], [4,3,1], [3,2,9], [4.5,3.9,2], [5.1,4,6], [6,7,1] ])
outputs = np.array( [[5, 1.5], [4, 1.5], [3.5, 3], [13, 3], [6, 4], [10.5, 4], [7, 6.5] ])

lmbd = 1
iterations = 1000

def print_stats(res):
    print "Cost: "+str(regression.fitness(inputs, outputs, res))
    print "Frobenius norm^2: "+str(regression.frobenius_norm_squared(res))
    print "Trace norm: "+str(regression.tracenorm(res))

print "Result with closed-form ridge regression:"
res, cost = regression.ridge(inputs, outputs, lmbd)
print res
print_stats(res)

bestL2Cost = cost[0]+cost[1]
upperBoundTNCost = cost[0]+lmbd*regression.tracenorm(res)

print ""

print "Result with gradient descent ridge regression:"
res, costIR = regression.incremental_ridge(inputs, outputs, lmbd, iterations, 0.001)
print res
print_stats(res)

for i in range(len(costIR)):
    total_cost = costIR[i][0] + costIR[i][1]
    costIR[i].append(total_cost)
    costIR[i].append(bestL2Cost)

print ""

print "Result with gradient descent trace norm regression:"
res, costTN = regression.incremental_tracenorm(inputs, outputs, lmbd, iterations, 1000)
print res
print_stats(res)

for i in range(len(costTN)):
    total_cost = costTN[i][0] + costTN[i][1]
    costTN[i].append(total_cost)
    costTN[i].append(upperBoundTNCost)

#plt.plot(costIR)
#plt.show()
#plt.plot(costTN)
#plt.show()

print "Result with accelerated gradient descent:"
res, costATN = regression.accelerated_tracenorm(inputs, outputs, lmbd, 200, 1, 1.2)
print res
print_stats(res)

for i in range(len(costATN)):
    total_cost = costATN[i][0] + costATN[i][1]
    costATN[i].append(total_cost)
    costATN[i].append(upperBoundTNCost)

plt.plot(costATN)
plt.show()

