import csv
import numpy as np
from tensorflow import keras

# Chargement des données
numeros = []
with open('euromillions.csv') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        numeros.append(row[:5])

# Préparation des données pour l'entraînement
x_train = np.array(numeros[:-1], dtype=int)
y_train = np.array(numeros[1:], dtype=int)

# Création du réseau de neurones
model = keras.Sequential([
    keras.layers.Reshape((5, 1), input_shape=(5,)),
    keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)
])

model.compile(optimizer='adam', loss='mse')


# Entraînement du réseau de neurones
model.fit(x_train, y_train, epochs=100)

# Prédiction des numéros gagnants pour le prochain tirage
prochain_tirage = np.array([[-1, -1, -1, -1, -1]], dtype=int)  # valeurs inconnues
prediction = model.predict(prochain_tirage)[0]

# Empêcher deux numéros similaires d'être prédits
derniers_numeros = np.array(numeros[-1], dtype=int)
for i in range(5):
    while np.abs(prediction[i] - derniers_numeros[i]) <= 1:
        prediction[i] = np.random.randint(1, 51)

# Affichage de la prédiction et des numéros réels pour le dernier tirage
dernier_tirage = np.array([numeros[-1]], dtype=int)
print("Prédiction: ", np.sort(np.around(prediction).astype(int)))
print("Dernier tirage: ", dernier_tirage[0])
