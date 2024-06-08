import tensorflow as tf

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Print the shapes of the loaded data
print("Training data shapes:")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("\nTest data shapes:")
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)