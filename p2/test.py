# coding:utf-8
import os
if not os.path.exists('read'):
    os.makedirs('read/')

# Import TensorFlow
import tensorflow as tf
import numpy

# Specify the directory where the images are located
image_dir = './'

# List of image filenames with full paths
filenames = [os.path.join(image_dir, 'A.jpg'),
             os.path.join(image_dir, 'B.jpg'),
             os.path.join(image_dir, 'C.jpg')]

# Rest of the code remains the same
# List of image filenames
filenames = ['A.jpg', 'B.jpg', 'C.jpg']

# Create a dataset from the filenames
dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Repeat the dataset 5 times
dataset = dataset.repeat(5)

# Create an iterator
iterator = iter(dataset)

i = 0
while True:
    try:
        # Get the next filename from the iterator
        filename = next(iterator)
        
        # Read the image data
        filename = filename.numpy().decode('utf-8')
        image_data = tf.io.read_file(filename)
        
        i += 1
        
        # Save the image data
        with open(f'read/test_{i}.jpg', 'wb') as f:
            f.write(image_data.numpy())
    except StopIteration:
        # End of dataset
        break