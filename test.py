import tensorflow as tf
import numpy as np

data = np.array([[1,2,3,4,5,6,7,8,9,10],
                 [2,3,4,5,6,7,8,9,10,11],
                 [3,4,5,6,7,8,9,10,11,12],
                 [4,5,6,7,8,9,10,11,12,13],
                 [5,6,7,8,9,10,11,12,13,14],
                 [6,7,8,9,10,11,12,13,14,15],
                 [7,8,9,10,11,12,13,14,15,16],
                 [8,9,10,11,12,13,14,15,16,17],
                 [9,10,11,12,13,14,15,16,17,18],
                 [10,11,12,13,14,15,16,17,18,19],
                 [11,12,13,14,15,16,17,18,19,20]])

x_train = data[:, 0:8]
y_train = data[:, 8:10]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(2, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100)

new_data = np.array([[9,10,11,12,13,14,15,16,17,18],
                     [10,11,12,13,14,15,16,17,18,19]])

predictions = model.predict(new_data[:, 0:8])
print(predictions)
