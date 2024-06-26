# pylint: skip-file
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set NumPy printing options to display full arrays without truncation
np.set_printoptions(precision=4, suppress=True, threshold=np.inf)

#update acticationtype to 'sigmoid, 'relu', 'tanh' etc. as needed
activationtype = 'selu'

# 3-bit XOR Dataset
X_3bit = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
Y_3bit = np.array([[1, 0], [0, 1], [0, 1], [1, 0],
                   [0, 1], [1, 0], [1, 0], [0, 1]], dtype=np.float32)

# Model function with tf.function to reduce retracing
@tf.function(reduce_retracing=True)
def predict_model(model, data):
    return model(data)

# Building a model for 3-bit XOR with a single hidden layer
model_3bit = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=3, activation=activationtype),  # Hidden layer with 8 neurons
    tf.keras.layers.Dense(2, activation='softmax')             # Output layer with Softmax
])

# Compile the model
model_3bit.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Train the model for 4,000 epochs
history_3bit = model_3bit.fit(X_3bit, Y_3bit, epochs=4000, verbose=0)

# Evaluate the model
loss_3bit, accuracy_3bit = model_3bit.evaluate(X_3bit, Y_3bit, verbose=0)
print(f"3-bit Final Loss: {loss_3bit}, Final Accuracy: {accuracy_3bit}")

# Plot training loss over epochs
plt.figure()
plt.plot(history_3bit.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('3-bit Training Loss Over Epochs with' + activationtype)
plt.grid(True)
plt.show()

# Using tf.function to reduce retracing for predictions
predictions_3bit = predict_model(model_3bit, X_3bit)

print("3-bit Predictions (Full):")
print(predictions_3bit.numpy())
