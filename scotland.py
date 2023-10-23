import pandas as pd
import numpy as np
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow import keras
from tensorflow.keras import layers
from data import read_euromillions_data  # Importez la fonction read_euromillions_data depuis data.py

# Charger les données en utilisant la fonction read_euromillions_data
euromillions_data = read_euromillions_data('euromillions.csv')

X = np.array([tuple[:5] for tuple in euromillions_data[:-1]]) 
y = np.array([tuple[:5] for tuple in euromillions_data[:-1]])  

# Sélectionner la dernière ligne du CSV
last_row = np.array(euromillions_data[-1][:5])

# Les 5 numéros à prédire
to_predict = last_row.reshape(1, -1)

best_accuracy = 0.0  # Initialisation du meilleur taux de précision
best_params = {}
iteration = 0

while best_accuracy < 0.3:  # Le seuil est de 30%
    iteration += 1

    # Diviser les données en ensemble d'apprentissage
    X_train = X
    y_train = y

    # ... (le reste du code pour l'optimisation des hyperparamètres)

# Réentraîner le modèle en incluant la dernière ligne
model.fit(X, y, epochs=best_hps.get('epochs'))

# Prédire le dernier tuple de data.py
last_tuple = euromillions_data[-1][:5]
prediction = model.predict(last_tuple.reshape(1, -1))

print("Dernier tuple de data.py :")
print(last_tuple)
print("Prédiction pour le dernier tuple : ", prediction[0])
print("Erreur absolue moyenne finale : {0:.2f}".format(best_accuracy))
