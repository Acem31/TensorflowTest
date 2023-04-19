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
    keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(5)
])
model.compile(optimizer='adam', loss='mse')

# Entraînement du réseau de neurones
model.fit(x_train, y_train, epochs=100)

# Prédiction des numéros gagnants pour le prochain tirage
prochain_tirage = np.array([[-1, -1, -1, -1, -1]], dtype=int)  # valeurs inconnues
prediction = model.predict(prochain_tirage)

# Affichage de la prédiction et des numéros réels pour le dernier tirage
dernier_tirage = np.array([numeros[-1]], dtype=int)
print("Prédiction: ", np.around(prediction[0]).astype(int))
print("Dernier tirage: ", dernier_tirage[0])
