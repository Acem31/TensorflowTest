import csv
import numpy as np
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

# Chargement des données
numeros = []
with open('euromillions.csv') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        numeros.append(row[:5])

# Préparation des données pour l'entraînement
x_train = np.array(numeros[:-1], dtype=int)
y_train = np.array(numeros[1:], dtype=int)

# Fonction de création du modèle
def create_model(neurons=[16], layers=1, activation='relu', optimizer='adam', dropout=0.0):
    model = keras.Sequential()
    model.add(keras.layers.Reshape((5, 1), input_shape=(5,)))
    for i in range(layers):
        model.add(keras.layers.Conv1D(neurons[i], kernel_size=3, activation=activation))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(neurons[-1], activation=activation))
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
    'batch_size': [16, 32, 64, 128],
    'epochs': [50, 100, 150, 200]
}

param_list = list(ParameterGrid(param_grid))

for params in param_list:
    model = create_model(**params)
    model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0)
    score = model.evaluate(x_train, y_train, verbose=0)
    print(params, score)
