import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.errors = []
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        
        # Initialize Adam optimizer parameters
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.t = 0
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + self.epsilon)) / m
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        
        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # ADAM optimization
        self.t += 1
        
        # Update first layer weights and biases
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (dW1 ** 2)
        mW1_corr = self.mW1 / (1 - self.beta1 ** self.t)
        vW1_corr = self.vW1 / (1 - self.beta2 ** self.t)
        self.W1 -= self.learning_rate * mW1_corr / (np.sqrt(vW1_corr) + self.epsilon)
        
        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * (db1 ** 2)
        mb1_corr = self.mb1 / (1 - self.beta1 ** self.t)
        vb1_corr = self.vb1 / (1 - self.beta2 ** self.t)
        self.b1 -= self.learning_rate * mb1_corr / (np.sqrt(vb1_corr) + self.epsilon)
        
        # Update second layer weights and biases
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (dW2 ** 2)
        mW2_corr = self.mW2 / (1 - self.beta1 ** self.t)
        vW2_corr = self.vW2 / (1 - self.beta2 ** self.t)
        self.W2 -= self.learning_rate * mW2_corr / (np.sqrt(vW2_corr) + self.epsilon)
        
        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * (db2 ** 2)
        mb2_corr = self.mb2 / (1 - self.beta1 ** self.t)
        vb2_corr = self.vb2 / (1 - self.beta2 ** self.t)
        self.b2 -= self.learning_rate * mb2_corr / (np.sqrt(vb2_corr) + self.epsilon)
    
    def train(self, X, y, epochs=10001):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.cross_entropy_loss(y, y_pred)
            self.errors.append(loss)
            self.backward(X, y, y_pred)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def plot_errors(self):
        plt.plot(self.errors)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        plt.show()

#main function
if __name__ == "__main__":

  # 3-bit XOR Dataset
  X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
  Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0],
              [0, 1], [1, 0], [1, 0], [0, 1]], dtype=np.float32)

  # Initialize and train the neural network
  nn = NeuralNetwork(input_size=3, hidden_size=8, output_size=2)
  nn.train(X, Y, epochs=10001)
  nn.plot_errors()

  # Predict
  predictions = nn.forward(X)
  print("3-bit Predictions (Full):")
  print(predictions)
