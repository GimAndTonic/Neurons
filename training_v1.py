# %% [markdown]
# # One Neuron for your water Consumption

# %%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image

# %% [markdown]
# ## Water Function
# Calculate your water

# %%
WATER = lambda DAY_TIME, WORKOUT_TIME : DAY_TIME * 0.2 - 1.4 + WORKOUT_TIME * 0.5

# %%
X = np.linspace(7, 20, 14)
Y = np.linspace(0, 2, 4)
X, Y = np.meshgrid(X, Y)
Z = X * 0.2 - 1.4 + Y * 0.5

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title('How much should I drink?')
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Workout hours')
ax.set_zlabel('Liter of water')

ax.view_init(15, -90, 0)

plt.show()

# %% [markdown]
# ## Build Architecture

# %%


# %% [markdown]
# ## Create Trainingsdata

# %% [markdown]
# ## Synthetic Neuron

# %%
# configuration
learning_rate = 0.01

## preset/initials
# initial weights
w1, w2 = 0, 0

# initial bias
bias = 0

# %%
## Neuron Function
# Activation Function ReLU (Rectified Linear Unit)
def ReLU(x) :
    return max(0, x)

# Inner neuron
def compute_neuron(X, W, bias) :
    return np.dot(X, W) + bias
    #return ReLU( y_ )

# Loss (Error)
def compute_loss(Y_true, Y_pred) :
    return np.mean( (Y_true - Y_pred) ** 2 )

# Gradients for Learning
def compute_gradients(X, Y_true, Y_pred, W, bias) :
    dW = -2 * np.dot(X.T, (Y_true - Y_pred) / len(X))
    dB = -2 * np.sum(Y_true - Y_pred) / len(X)
    return dW, dB

# Update the weights
def update_weights(W, bias, dW, dB, learning_rate) :
    W    -= learning_rate * dW
    bias  = -0.07#-= learning_rate * dB
    return W, bias

# %%
sample_count = 10

x1 = list()
x2 = list()
y = list()

for i in range(sample_count) :
    x1.append( random.random() * (20-7) + 7 )
    x2.append( random.random() * 7)
    y.append( WATER(x1[-1], x2[-1]) )

X = np.vstack((np.array(x1), np.array(x2))).T
Y = np.array(y)

# %%
X /= 20
Y /= 20

# Initialize weights and bias
weights = np.random.rand(2)  # Two weights for two input features
bias = np.random.rand()

# Training parameters
learning_rate = 0.01
epochs = 1000

weight_history = list()
loss_history   = list()

# Training loop
for epoch in range(epochs):
    # Compute neuron output
    Y_pred = compute_neuron(X, weights, bias)

    # Compute loss
    loss = compute_loss(Y, Y_pred)

    # Compute gradients
    dW, dB = compute_gradients(X, Y, Y_pred, weights, bias)

    # Update weights and bias
    weights, bias = update_weights(weights, bias, dW, dB, learning_rate)
    # Optionally, print the loss at certain intervals
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
        print(weights)
        weight_history.append((weights[0], weights[1]))
        loss_history.append(loss)

# Final weights and bias
print("Final weights:", weights)
print("Final bias:", bias)


# %%
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
x, y = np.meshgrid(x, y)

z = list()
for i in range(100) :
    z_i = list()
    for j in range(100):
        z_i.append(
            compute_loss(Y, compute_neuron(
                X, 
                np.array([x[i, j],y[i, j]]), 
                -0.07)
            )
        )
    z.append(z_i)

z = np.array(z)


# %%

fig = plt.figure(figsize=(10, 30))
ax = fig.add_subplot(111, projection='3d')
w1_history, w2_history = zip(*weight_history)
ax.plot(w1_history, w2_history, np.log(loss_history) + 0.1, 
        marker='o', color='b', markersize=5, linewidth=2)

ax.plot_surface(x, y, np.log(z), color='viridis', alpha=0.5)

ax.set_title('Optimisation Problem')
ax.set_xlabel('w_1')
ax.set_ylabel('w_2')
ax.set_zlabel('loss')

#ax.view_init(0, 0, 0)

plt.show()
