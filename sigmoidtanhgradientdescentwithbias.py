# pylint: skip-file
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.errors = []
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)  # Weight matrix for input to hidden
        self.b1 = np.zeros((1, self.hidden_size))  # Bias for hidden layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size)  # Weight matrix for hidden to output
        self.b2 = np.zeros((1, self.output_size))  # Bias for output layer
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
      
    def tanh_derivative(self, x):
        return 1 - x ** 2

    def forward_tanh(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.tanh(self.z2)
        return self.a2

    def backward_tanh(self, X, y, output):
        # Backpropagation
        self.error = output - y
        self.delta_output = self.error * self.tanh_derivative(output)
        self.error_hidden = np.dot(self.delta_output, self.W2.T)
        self.delta_hidden = self.error_hidden * self.tanh_derivative(self.a1)
        
        # Update weights and biases
        self.W2 -= np.dot(self.a1.T, self.delta_output)
        self.b2 -= np.sum(self.delta_output, axis=0, keepdims=True)
        self.W1 -= np.dot(X.T, self.delta_hidden)
        self.b1 -= np.sum(self.delta_hidden, axis=0, keepdims=True)


    def forward_sigmoid(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward_sigmoid(self, X, y, output):
        # Backpropagation
        self.error = output - y
        self.delta_output = self.error * self.sigmoid_derivative(output)
        self.error_hidden = np.dot(self.delta_output, self.W2.T)
        self.delta_hidden = self.error_hidden * self.sigmoid_derivative(self.a1)
        
        # Update weights and biases
        self.W2 -= np.dot(self.a1.T, self.delta_output)
        self.b2 -= np.sum(self.delta_output, axis=0, keepdims=True)
        self.W1 -= np.dot(X.T, self.delta_hidden)
        self.b1 -= np.sum(self.delta_hidden, axis=0, keepdims=True)
    
    def train_tanh(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward_tanh(X)
            self.backward_tanh(X, y, output)
            error = np.mean(np.abs(self.error))
            self.errors.append(error)
            if epoch % 1000 == 0:
                print(f'Error: {np.mean(np.abs(self.error))}')

    def train_sigmoid(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward_sigmoid(X)
            self.backward_sigmoid(X, y, output)
            error = np.mean(np.abs(self.error))
            self.errors.append(error)
            if epoch % 1000 == 0:
                print(f'Error: {np.mean(np.abs(self.error))}')

    def plot_errors(self):
        plt.plot(range(len(self.errors)), self.errors)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error vs. Epochs')
        plt.show()

#main function
if __name__ == "__main__":
    # Define training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # output is xor of input data
    y = np.array([[0], [1], [1], [0]])
    
    # Initialize neural network
    input_size = 2
    hidden_size = 4
    output_size = 1
    nn_sigmoid = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Train neural network
    nn_sigmoid.train_sigmoid(X, y, epochs=20001)

    # Plot errors
    nn_sigmoid.plot_errors()
    
    # Test the trained network using sigmoid activation function
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print("Predictions after training:")
    for input_data in test_data:
        output = nn_sigmoid.forward_sigmoid(input_data)
        print(f'Input: {input_data}, Output: {output}')

    nn_tanh = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Train neural network
    nn_tanh.train_tanh(X, y, epochs=20001)

    # Plot errors
    nn_tanh.plot_errors()

    # Test the trained network using tanh activation function
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print("Predictions after training:")
    for input_data in test_data:
        output = nn_tanh.forward_sigmoid(input_data)
        print(f'Input: {input_data}, Output: {output}')
