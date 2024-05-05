import numpy as np
import matplotlib.pyplot as plt

#sigmoid activation function

def sigmoid(x):
  return 1/ (1 + np.exp(-x))

#derivative of sigmoid

def sigmoid_derivative(x):
  return x * (1-x)

#tanh activation function

def tanh(x):
  return np.tanh(x)

#derivative of tanh function

def tanh_derivative(x):
  return 1 - x ** 2


#input data
X = np.array([[0,0], [0,1], [1,0], [1,1]])

#expected output
y = np.array([[0], [1], [1], [0]])

#seed for random weights initially

np.random.seed(31)

#initialize weights randomly
weights_0 = 2 * np.random.random((2,3)) - 1 #weights for input to hidden layer
weights_1 = 2 * np.random.random((3,1)) - 1 #weights for hidden to output layer

#store errors for plotting
errors = []

for epoch in range(20001):
  #forward prop
  layer_0 =X
  layer_1 = sigmoid(np.dot(layer_0, weights_0))
  layer_2 = sigmoid(np.dot(layer_1, weights_1))

  #calculate error
  layer_2_error = y - layer_2
  errors.append(np.mean(np.abs(layer_2_error)))

  #backprop
  layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
  layer_1_error = layer_2_delta.dot(weights_1.T)
  layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

  #update weights
  weights_1 += layer_1.T.dot(layer_2_delta)
  weights_0 += layer_0.T.dot(layer_1_delta)

  #print error every 1000 epoch
  if epoch % 1000 == 0:
    print(f'Error at epoch with sigmoid {epoch}: {np.mean(np.abs(layer_2_error))}')

#plot the errors to show gradient descent
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Error over Epochs - Sigmoid')
plt.show()

#seed for random weights initially; use tanh derivative

np.random.seed(27)

#initialize weights randomly
weights_0 = 2 * np.random.random((2,3)) - 1 #weights for input to hidden layer
weights_1 = 2 * np.random.random((3,1)) - 1 #weights for hidden to output layer

#store errors for plotting
errors = []

for epoch in range(20001):
  #forward prop
  layer_0 =X
  layer_1 = tanh(np.dot(layer_0, weights_0))
  layer_2 = tanh(np.dot(layer_1, weights_1))

  #calculate error
  layer_2_error = (y - layer_2)
  errors.append(np.mean(np.abs(layer_2_error)))

  #backprop
  layer_2_delta = layer_2_error * tanh_derivative(layer_2)
  layer_1_error = layer_2_delta.dot(weights_1.T)
  layer_1_delta = layer_1_error * tanh_derivative(layer_1)

  #update weights
  weights_1 += layer_1.T.dot(layer_2_delta)
  weights_0 += layer_0.T.dot(layer_1_delta)

  #print error every 1000 epoch
  if epoch % 1000 == 0:
    print(f'Error at epoch with tanh {epoch}: {np.mean(np.abs(layer_2_error))}')

#plot the errors to show gradient descent
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Error over Epochs - tanh')
plt.show()
