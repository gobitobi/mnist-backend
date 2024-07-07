import os
import tensorflow as tf
from tensorflow import keras

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    return (X_train, y_train), (X_test, y_test)

def create_model():
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)), # input tensor
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=10):
    checkpoint_path = "training_1/cp.ckpt.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    
    model.save_weights('test_weights.weights.h5')
    
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[cp_callback])
    
    return os.listdir(checkpoint_dir)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    model = create_model()
    model.summary()
    checkpoint_dir_contents = train_model(model, X_train, y_train, X_test, y_test, epochs=10)
    print(checkpoint_dir_contents)

# import os
# import tensorflow as tf
# from tensorflow import keras

# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
        
#         # Load and preprocess the data
#         (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
#         self.X_train = self.X_train.reshape(-1, 28, 28, 1) / 255.0
#         self.X_test = self.X_test.reshape(-1, 28, 28, 1) / 255.0
        
#         # Define the model layers
#         self.conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu')
#         self.bn1 = keras.layers.BatchNormalization()
#         self.pool1 = keras.layers.MaxPooling2D((2, 2))
#         self.drop1 = keras.layers.Dropout(0.25)

#         self.conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu')
#         self.bn2 = keras.layers.BatchNormalization()
#         self.pool2 = keras.layers.MaxPooling2D((2, 2))
#         self.drop2 = keras.layers.Dropout(0.25)

#         self.flatten = keras.layers.Flatten()
#         self.dense1 = keras.layers.Dense(128, activation='relu')
#         self.bn3 = keras.layers.BatchNormalization()
#         self.drop3 = keras.layers.Dropout(0.5)
#         self.dense2 = keras.layers.Dense(10, activation='softmax')
        
#         # Compile the model
#         self.compile(optimizer='adam',
#                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    
#     def build_graph(self):
#         input_shape = (28, 28, 1)
#         inputs = tf.keras.Input(shape=input_shape)
#         return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))

#     def call(self, inputs, training=False):
#         x = self.conv1(inputs)
#         x = self.bn1(x, training=training)
#         x = self.pool1(x)
#         x = self.drop1(x, training=training)

#         x = self.conv2(x)
#         x = self.bn2(x, training=training)
#         x = self.pool2(x)
#         x = self.drop2(x, training=training)

#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.bn3(x, training=training)
#         x = self.drop3(x, training=training)
#         return self.dense2(x)
    
#     def summary(self):
#         inputs = tf.keras.Input(shape=(28, 28, 1))
#         _ = self.call(inputs)  # Build the model
#         super().summary()
        
#     def train(self, epochs=5):
#         # Define a callback for saving the model checkpoints
#         # checkpoint_name = "cp.ckpt.weights.h5"
#         checkpoint_path = "training_1/cp.ckpt.weights.h5"
#         # checkpoint_path = os.path.join("backend/model", checkpoint_name)
#         checkpoint_dir = os.path.dirname(checkpoint_path)
        
#         # Ensure the checkpoint directory exists
#         if not os.path.exists(checkpoint_dir):
#             os.makedirs(checkpoint_dir)
        
#         cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                          save_weights_only=True,
#                                                          verbose=1)
#         model.save_weights('test_weights.weights.h5')
        
#         # Train the model
#         self.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_test, self.y_test), callbacks=[cp_callback])
        
#         # List the checkpoint directory contents
#         return os.listdir(checkpoint_dir)

# if __name__ == "__main__":
#     # Instantiate and use the MyModel class
#     model = MyModel()
#     model.summary()
#     checkpoint_dir_contents = model.train(epochs=10)
#     print(checkpoint_dir_contents)

