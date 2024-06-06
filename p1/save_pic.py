'''This file stores images so user can see it with eyes.'''
import tensorflow as tf
import scipy.misc
import os
from PIL import Image
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# mnist = tf.keras.datasets.mnist.read_data_sets("MNIST_data/", one_hot = True)
mnist = tf.keras.datasets.mnist.load_data()

save_dir = 'MNIST_data/raw'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in range (3):
    image_array = x_train[i , :]
    image = image_array.reshape(28, 28)

    filename = os.path.join(save_dir, 'mnist_train_%d.jpg' % i)
    Image.fromarray(image).save(filename)