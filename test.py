import numpy as np
import ridge

inputs = np.array([ [1,1,4], [2,3,0], [4,3,1], [3,2,9], [4.5,3.9,2], [5.1,4,6], [6,7,1] ])
outputs = np.array( [[5, 1.5], [4, 1.5], [3.5, 3], [13, 3], [6, 4], [10.5, 4], [7, 6.5] ])

print ridge.regression(inputs, outputs, 0.01)

