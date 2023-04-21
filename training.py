import csv
import numpy as np
from tensorflow import keras
from sklearn.model_selection import GridSearchCV, ParameterGrid

# Chargement des données
numeros = []
with open('euromillions.csv') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        numeros.append(row[:5])

# Préparation des données pour l'entraînement
x_train = np.array(numeros[:-1], dtype=int)
y_train = np.array(numeros[1:], dtype=int)

def create_model(neurons=[16], layers=1, activation='relu', optimizer='adam', dropout=0.0):
    model = keras.Sequential()
    for i in range(layers):
        if i == 0:
            model.add(keras.layers.Dense(neurons[i], input_dim=5, activation=activation))
        else:
            model.add(keras.layers.Dense(neurons[i], activation=activation))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(5))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Définition des hyperparamètres à tester
param_grid = {
    'neurons': [[16], [32], [64], [128], [256], [512], [16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
    'layers': [1, 2, 3, 4],
    'activation': ['relu', 'tanh', 'sigmoid', 'linear'],
    'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad'],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}

# Utilisation de GridSearchCV pour tester toutes les combinaisons d'hyperparamètres
grid_search = GridSearchCV(estimator=create_model(),
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)
grid_search.fit(x_train, y_train)

# Affichage des résultats
print('Meilleurs hyperparamètres trouvés:', grid_search.best_params_)
print('Score de la meilleure combinaison:', grid_search.best_score_)
