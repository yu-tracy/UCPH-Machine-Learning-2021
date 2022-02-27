import numpy as np
import matplotlib.pyplot as plt

# get weights
def compute_weights(X, y):
    X = np.column_stack((X**2, X))   
    inv = np.linalg.inv(np.dot(X.T, X))
    W = np.dot(np.dot(inv, X.T), y)
    return W

def prediction(X, W):
    X = np.column_stack((X**2, X))
    return np.dot(X, W)

def draw_plot():
    trajectory_of_cannon = np.array([[0, 0], [1, 14], [2, 21], [3, 25], [4, 35], [5, 32]])
    weights = compute_weights(trajectory_of_cannon[:,0], trajectory_of_cannon[:,1])
    x_sets = np.linspace(0, 10, 100)  # value of x from 0 to 10
    y = prediction(x_sets, weights)
    print(weights)
    plt.plot(x_sets, y, color = 'blue', label = 'the predicted parabolic trajection')
    plt.scatter(trajectory_of_cannon[:,0], trajectory_of_cannon[:,1], color = 'red', label = 'the observed locations')
    plt.axhline(y=0,ls="--",c="black")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('regression.png')
    plt.close()