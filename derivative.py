# pylint: skip-file
import math
import matplotlib.pyplot as plt
import numpy as np


def f(x):
  return 3 * x ** 2 - 4 * x + 5

def derivativex(x):
  return 6*x - 4

#sigmoid activation function

def sigmoid(x):
  return 1/ (1 + np.exp(-x))

#derivative of sigmoid

def sigmoid_derivative(x):
  return sigmoid(x) * sigmoid(1-x)

#tanh activation function

def tanh(x):
  return np.tanh(x)

#derivative of tanh function

def tanh_derivative(x):
  return 1 - tanh(x) ** 2


#plot a parabola and its derivative
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
ydashx = derivativex(xs)
plt.plot(xs, ys)
plt.plot(xs, ydashx)
plt.show()

#plot sigmoid function and its derivative.
xs = np.arange(-15, 15, 0.25)
ys_sigmoid = sigmoid(xs)
ydashx_sigmoid = sigmoid_derivative(xs)


#plot tanh and it's derivative.
ys_tanh = tanh(xs)
ydashx_tanh = tanh_derivative(xs)

plt.plot(xs, ys_sigmoid)
plt.plot(xs, ys_tanh)
plt.xlabel('x-values')
plt.ylabel('sigmoid (blue), tanh(orange)')
plt.title('sigmoid and tanh functions with x-values')
plt.show()

plt.plot(xs, ydashx_sigmoid)
plt.plot(xs, ydashx_tanh)
plt.xlabel('x-values')
plt.ylabel('sigmoid derivative (blue), tanh derivative(orange)')
plt.title('sigmoid and tanh derivatives with x-values')
plt.show()
