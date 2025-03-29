import numpy as np
import tensorflow as tf

class TradingModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 actions: Hold, Buy, Sell
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=50, batch_size=32):
        x_train = np.array(x_train, dtype=np.float32)
        print(f"x_train shape: {x_train.shape}")  # Should print (num_samples, 2)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, state):
        """
        Predict the best action for a given state.
        :param state: Current state of the environment.
        :return: Action index (0=Hold, 1=Buy, 2=Sell).
        """
        state = np.array(state).reshape(1, -1)  # Reshape for a single sample
        probabilities = self.model.predict(state, verbose=0)  # Suppress verbose output
        return int(np.argmax(probabilities))  # Return the action with the highest probability as an integer

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)