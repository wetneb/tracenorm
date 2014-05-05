import numpy as np
import ridge

inputs = np.array([ [1,1,4], [2,3,0], [4,3,1], [3,2,9], [4.5,3.9,2], [5.1,4,6], [6,7,1] ])
outputs = np.array( [[5, 1.5], [4, 1.5], [3.5, 3], [13, 3], [6, 4], [10.5, 4], [7, 6.5] ])

lmbd = 0.01

print "Result with closed-form ridge regression:"
print ridge.regression(inputs, outputs, lmbd)

print "Result with gradient descent ridge regression:"
print ridge.incremental(inputs, outputs, lmbd, 1000, 0.001)

print "Result with gradient descent trace norm regression:"
print ridge.tracenorm(inputs, outputs, lmbd, 1000, 0.001)

