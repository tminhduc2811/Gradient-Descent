import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from file
data = pd.read_csv('Salary_Data.csv').values
N = data.shape[0]

# Reshape data
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Plot data set
plt.scatter(x, y)
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
# plt.show()

# Add a column with value = 1 to X matrix
x = np.hstack((np.ones((N, 1)), x))

# Create W matrix
w = np.array([1., 1.]).reshape(-1, 1)

# Number of iteration
iteration = 200

# Create an array to save learning rate
cost = np.zeros((iteration, 1))

# Set learning rate
learning_rate = 0.000001

# Start training
for i in range(1, iteration):
    # Calculate yr - y
    r = np.dot(x, w) - y
    # Calculate J
    cost[i] = 0.5*np.sum(r*r)/N
    # Update w0 & w1
    w[0] -= learning_rate*np.sum(r)
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:, 1]))
    print('Loss = ', cost[i])

predict = np.dot(x, w)
# With 15 years experience, predict the amount of salary:
print('Salary for 15 years of experience: ', w[0] + w[1] * 15)
print('Salary for 6 years of experience: ', w[0] + w[1] * 6)
plt.plot((x[0][1], x[N-1][1]), (predict[0], predict[N-1]), 'r')
plt.show()
