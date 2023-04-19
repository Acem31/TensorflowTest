import csv
import numpy as np
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

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
def create_model(neurons=[16], layers=1):
    model = keras.Sequential()
    model.add(keras.layers.Reshape((5, 1), input_shape=(5,)))
    for i in range(layers):
        model.add(keras.layers.Conv1D(neurons[i], kernel_size=3, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(neurons[-1], activation='relu'))
    model.add(keras.layers.Dense(5))
    model.compile(optimizer='adam', loss='mse')
    return model

# Définition des hyperparamètres à tester
param_grid = {
    'neurons': [[16], [32], [64], [128], [16, 16], [32, 32], [64, 64], [128, 128]],
    'layers': [1, 2]
}

# Recherche des hyperparamètres optimaux
model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
grid_result = grid.fit(x_train, y_train)

# Affichage des résultats
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Entraînement du modèle avec les hyperparamètres optimaux
best_model = create_model(neurons=grid_result.best_params_['neurons'], layers=grid_result.best_params_['layers'])
best_model.fit(x_train, y_train, epochs=100)

# Prédiction des numéros gagnants pour le prochain tirage
prochain_tirage = np.array([[-1, -1, -1, -1, -1]], dtype=int)  # valeurs inconnues
prediction = best_model.predict(prochain_tirage)[0]

# Empêcher deux numéros similaires d'être prédits
derniers_numeros = np.array(numeros[-1], dtype=int)
for i in range(5):
    while np.abs(prediction[i] - derniers_numeros[i]) <= 1:
        prediction[i] = np.random.randint(1, 51)

# Affichage de la prédiction et des numéros réels pour le dernier tirage
dernier_tirage = np.array([numeros[-1]], dtype=int)
print("Prédiction: ", np.sort(np.around(prediction).astype(int)))
print("Dernier tirage: ", dernier_tirage[0])
