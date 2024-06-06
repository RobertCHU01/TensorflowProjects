import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# mnist = tf.keras.datasets.mnist.read_data_sets("MNIST_data/", one_hot = True)
mnist = tf.keras.datasets.mnist.load_data()

# shape of train/test data/label
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape)
# print(mnist.validation.images.shape)
# print(mnist.validation.labels.shape)
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape)
# shape of one picture
print(x_train[0 , :].shape) # (28, 28)