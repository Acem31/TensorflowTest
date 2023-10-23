import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow import keras
from tensorflow.keras import layers

# Charger le fichier CSV
data = pd.read_csv('euromillions.csv', delimiter=';', header=None)

X = data.iloc[:-1, :5] 
y = data.iloc[:-1, :5]  

# Sélectionner la dernière ligne du CSV
last_row = data.iloc[-1, :5] 

# Les 5 numéros à prédire
to_predict = last_row.to_numpy().reshape(1, -1)

best_accuracy = 0.0
best_params = {}
iteration = 0

while best_accuracy < 30:
    iteration += 1

    # Diviser les données en ensemble d'apprentissage
    X_train = X
    y_train = y

    # Définir la fonction de modèle Keras pour Keras Tuner
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Dense(units=hp.Int('units', min_value=1, max_value=50, step=1), activation='softmax'))
        model.add(layers.Dense(50, activation='softmax'))
        model.add(layers.Dense(5, activation='linear'))  # Utilisez 'linear' pour la régression
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='mean_squared_error',  # Utilisez 'mean_squared_error' pour la régression
                      metrics=['mae'])  # Utilisez 'mae' pour mesurer l'erreur absolue moyenne
    
        # Ajouter l'hyperparamètre 'epochs'
        epochs = hp.Int('epochs', min_value=5, max_value=30, step=5)
        model.fit(X_train, y_train, epochs=epochs)
    
        return model


    # Configurer le tuner Keras
    tuner = RandomSearch(
        build_model,
        objective='mae',  # Utilisez l'erreur absolue moyenne pour la régression
        max_trials=10,
        directory='my_dir',
        project_name='my_project'
    )

    # Effectuer la recherche des hyperparamètres
    tuner.search(X_train, y_train)

    # Récupérer les meilleurs hyperparamètres
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Construire le modèle avec les meilleurs hyperparamètres
    model = tuner.hypermodel.build(best_hps)

    # Faire des prédictions pour la dernière ligne
    y_pred = model.predict(to_predict)

    # Calculer l'erreur absolue moyenne entre les vrais numéros et les prédictions
    mae = np.mean(np.abs(last_row.to_numpy() - y_pred[0]))

    print(f"Itération {iteration} - Erreur absolue moyenne : {mae:.2f}")
    print("Dernière ligne du CSV :", last_row)
    print("Prédiction pour la dernière ligne : ", y_pred[0])

    if mae < best_accuracy:
        best_accuracy = mae

# Réentraîner le modèle en incluant la dernière ligne
model.fit(X, y, epochs=best_hps.get('epochs'))

last_row = data.iloc[-1, :-5]  # Sélectionner les 5 premières colonnes de la dernière ligne
prediction = model.predict(last_row.to_numpy().reshape(1, -1))
print("Dernière ligne du CSV :")
print(last_row)
print("Prédiction pour la dernière ligne : ", prediction[0])
print("Erreur absolue moyenne finale : {0:.2f}".format(best_accuracy))
