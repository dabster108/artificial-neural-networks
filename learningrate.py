''' 2 NEURONS INPUT - > 1 OUT PUT NEURON'''
import math 
x1 = 1 
x2 =2 
y =1 # actual output 


w1 = 0.4
w2 = 0.6
b = 0
alpha = 0.1  # learning rate 


z = w1 * x1 + w2 * x2 + b
print(f"Weighted sum (z): {z}") # weighted sum 

def sigmoid(x):
     return 1 / (1 + math.exp(-x))


y_hat = sigmoid(z)
print(f"Prediction (ŷ): {y_hat:.3f}") 


loss = (y - y_hat) ** 2
print(f"Loss (MSE): {loss:.3f}") # loss function 


dL_dw1 = -0.02
dL_dw2 = -0.04   # assuming gradients from backpropagation 

# gradient descent update 
w1_new = w1 - alpha * dL_dw1
w2_new = w2 - alpha * dL_dw2


print(f"Updated w1: {w1_new:.3f}")
print(f"Updated w2: {w2_new:.3f}") # updated weights 




