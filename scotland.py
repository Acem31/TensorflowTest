import numpy as np
from tensorflow import keras
from data import read_euromillions_data
from kerastuner.tuners import RandomSearch
from parameter import update_batch_size  # Importez la fonction depuis parameter.py
import os

# Charger les données en utilisant la fonction read_euromillions_data
euromillions_data = read_euromillions_data('euromillions.csv')

# Initialisation du taux de précision
best_accuracy = 0.0

# Initialisation du nombre d'itérations
iteration = 0

batch_size = 1

# Liste des fonctions d'activation à tester
activation_functions = ['sigmoid', 'tanh']
model = None

while best_accuracy < 0.3:  # Le seuil est de 30%
    iteration += 1

    # Sélectionner la dernière ligne du CSV
    last_row = np.array(euromillions_data[-1][:5])

    best_activation = None
    best_accuracy_for_activation = 0.0

    for activation in activation_functions:
        # Définir une fonction pour prédire un tuple de 5 numéros
        def predict_next_tuple(last_tuple, hps):
            # Extraire les valeurs optimisées d'hyperparamètres
            best_units = hps.get('units')
            best_learning_rate = hps.get('learning_rate')
            # Construire le modèle ANN avec les hyperparamètres actuels
            model = keras.Sequential([
                keras.layers.Dense(hps.Int('units', min_value=32, max_value=512, step=32), activation=activation, input_shape=(5,)),
                keras.layers.Dense(hps.Int('units', min_value=32, max_value=512, step=32), activation=activation),
                keras.layers.Dense(5)  # 5 sorties pour prédire les 5 numéros
            ])

            # Compiler le modèle
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=hps.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                          loss='mse', metrics=['mae'])

            # Entraîner le modèle avec le nombre d'epochs actuel
            X = np.array(euromillions_data[:-1])
            y = np.array(euromillions_data[1:])
            model.fit(X, y, epochs=iteration * 50, batch_size=batch_size, verbose=2)

            model.save_weights(f'model_weights_iteration_{iteration}.h5')

            # Prédire le prochain tuple
            prediction = model.predict(last_tuple.reshape(1, 5))

            return prediction[0]

        # Créer un tuner Keras pour la recherche d'hyperparamètres
        tuner = RandomSearch(
            predict_next_tuple,
            objective='mae',
            max_trials=10,
            directory='my_dir',
            project_name='my_project'
        )

        # Chercher les meilleurs hyperparamètres pour cette itération
        tuner.search(last_row, num_trials=10)  # Effectuer la recherche d'hyperparamètres

        # Obtenir les meilleurs hyperparamètres de la recherche
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Prédire le prochain tuple en utilisant les meilleurs hyperparamètres actuels
        next_tuple = predict_next_tuple(last_row, best_hps)

        print(f"Itération {iteration}, Activation: {activation} - Prédiction pour le prochain tuple : {next_tuple}")

        # Calculer la précision (à adapter selon le type de problème)

        current_directory = os.getcwd()
        files = os.listdir(current_directory)
        weight_files = [file for file in files if file.startswith('model_weights_iteration_')]
        latest_weight_file = max(weight_files)

        # Pour la régression, utilisez une métrique appropriée
        model.load_weights(os.path.join(current_directory, latest_weight_file))
        accuracy = model.evaluate(X, y, verbose=0)  # Évaluez le modèle sur vos données
        mse = accuracy[0]

        print(f"Précision pour l'itération {iteration}, Activation: {activation}, Précision: {accuracy:.2f}")

        if accuracy > best_accuracy_for_activation:
            best_accuracy_for_activation = accuracy
            best_activation = activation

    if best_accuracy_for_activation > best_accuracy:
        best_accuracy = best_accuracy_for_activation
        best_activation_final = best_activation

print(f"Taux de précision atteint : {best_accuracy:.2f} avec Activation: {best_activation_final}")
