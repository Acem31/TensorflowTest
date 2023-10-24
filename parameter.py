import pandas as pd
import numpy as np
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow import keras
from tensorflow.keras import layers
from data import read_euromillions_data

def update_batch_size(current_accuracy, hyperparameters):
    # Mettez en place votre logique pour calculer la nouvelle valeur de batch_size
    # en fonction de la précision actuelle et des hyperparamètres.
    # Par exemple, vous pouvez doubler la taille du lot si la précision actuelle
    # est supérieure à un seuil donné.
    if current_accuracy > 0.7:  # Exemple de seuil de précision
        new_batch_size = hyperparameters['batch_size'] * 2
    else:
        new_batch_size = hyperparameters['batch_size']

    return new_batch_size

# Charger les données en utilisant la fonction read_euromillions_data
euromillions_data = read_euromillions_data('euromillions.csv')

# Préparer les données d'entraînement
X = np.array([tuple[:5] for tuple in euromillions_data[:-1]])
y = np.array(euromillions_data[1:])  # Nous prédisons la ligne suivante par rapport à celle précédente

# Définir la fonction de modèle Keras pour Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(5,)))
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(layers.Dense(5)  # 5 sorties pour prédire les 5 numéros
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse', metrics=['mae'])
    return model

# Configurer le tuner Keras
tuner = RandomSearch(
    build_model,
    objective='mae',
    max_trials=10,  # Nombre d'essais pour la recherche d'hyperparamètres
    directory='my_dir',  # Répertoire pour enregistrer les résultats de la recherche
    project_name='my_project'
)

# Effectuer la recherche des hyperparamètres
tuner.search(X, y, epochs=50, batch_size=1, verbose=2)

# Récupérer les meilleurs hyperparamètres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres :")
print(best_hps.get_config())
