import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Define input and output tensors
inputs = tf.keras.Input(shape=(784,), dtype=tf.float32)
outputs = tf.keras.layers.Dense(10, activation='softmax')(inputs)

# Define model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile model with cross-entropy loss and SGD optimizer
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')

