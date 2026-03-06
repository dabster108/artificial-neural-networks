#neural network activation function
# Input - Layer 
# Hidden - Layer 
# Output - Layer 
#inputs
import numpy as np 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



x1 =2 
x2 = 3 
w1 = 0.5 
w2 = -0.4 

# Bias
b = 0.1

# Weighted sum
z = (w1 * x1) + (w2 * x2) + b
# Activation output
a = sigmoid(z)

print("z value:", z)
print("Activated output:", a)



