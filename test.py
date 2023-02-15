import tensorflow as tf
import numpy as np

data = np.array([[2,4,6,8,10,12,14,16,18,20],
                 [4,6,8,10,12,14,16,18,20,22],
                 [6,8,10,12,14,16,18,20,22,24],
                 [8,10,12,14,16,18,20,22,24,26],
                 [10,12,14,16,18,20,22,24,26,28],
                 [12,14,16,18,20,22,24,26,28,30],
                 [14,16,18,20,22,24,26,28,30,32],
                 [16,18,20,22,24,26,28,30,32,34],
                 [18,20,22,24,26,28,30,32,34,36],
                 [20,22,24,26,28,30,32,34,36,38],
                 [22,24,26,28,30,32,34,36,38,40]])

x_train = data[:, 0:8]
y_train = data[:, 8:10]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(2, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100)

new_data = np.array([[30,32,34,36,38,40,42,44,46,48],
                     [40,42,44,46,48,50,52,54,56,58]])

predictions = model.predict(new_data[:, 0:8])
print(predictions)
