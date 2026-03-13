import math
import matplotlib.pyplot as plt

x1 = 1
x2 = 2
y = 1

w1 = 0.4
w2 = 0.6
b = 0
alpha = 0.1
epochs = 20

loss_history = []

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

for epoch in range(epochs):
    z = w1 * x1 + w2 * x2 + b
    y_hat = sigmoid(z)
    loss = (y - y_hat) ** 2
    loss_history.append(loss)
    
    dL_dyhat = -2 * (y - y_hat)
    dyhat_dz = y_hat * (1 - y_hat)
    
    dL_dw1 = dL_dyhat * dyhat_dz * x1
    dL_dw2 = dL_dyhat * dyhat_dz * x2
    
    w1 = w1 - alpha * dL_dw1
    w2 = w2 - alpha * dL_dw2
    
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, w1={w1:.3f}, w2={w2:.3f}, ŷ={y_hat:.3f}")

plt.plot(range(1, epochs+1), loss_history, marker='o')
plt.title("Loss over Epochs 🧠📉")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

#epoch - forward pass / back propagation 