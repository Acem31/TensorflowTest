import numpy as np
from tensorflow import keras
from data import read_euromillions_data
from parameter import best_hps  # Importer les meilleurs hyperparamètres depuis parameter.py

# Charger les données en utilisant la fonction read_euromillions_data
euromillions_data = read_euromillions_data('euromillions.csv')

# Initialisation du taux de précision
best_accuracy = 0.0

# Initialisation du nombre d'itérations
iteration = 0

# Définir une fonction pour prédire un tuple de 5 numéros
def predict_next_tuple(last_tuple):
    # Construire le modèle ANN avec les meilleurs hyperparamètres
    model = keras.Sequential([
        keras.layers.Dense(best_hps.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(5,)),
        keras.layers.Dense(best_hps.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        keras.layers.Dense(5)  # 5 sorties pour prédire les 5 numéros
    ])

    # Compiler le modèle
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hps.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse', metrics=['mae'])

    # Entraîner le modèle
    X = np.array(euromillions_data[:-1])
    y = np.array(euromillions_data[1:])
    model.fit(X, y, epochs=50, batch_size=1, verbose=2)

    # Prédire le prochain tuple
    prediction = model.predict(last_tuple.reshape(1, 5))

    return prediction[0]

while best_accuracy < 0.3:  # Le seuil est de 30%
    iteration += 1

    # Sélectionner la dernière ligne du CSV
    last_row = np.array(euromillions_data[-1][:5])

    # Prédire le prochain tuple
    next_tuple = predict_next_tuple(last_row)

    print(f"Itération {iteration} - Prédiction pour le prochain tuple : {next_tuple}")

    # Calculer la précision (à adapter selon le type de problème)
    # Par exemple, si vous effectuez une classification, utilisez accuracy_score
    # Pour la régression, utilisez une métrique appropriée
    accuracy = 0.0  # Calculez ici votre taux de précision

    print(f"Précision pour l'itération {iteration} : {accuracy:.2f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy

# Prédiction finale
if best_accuracy >= 0.3:
    last_row = np.array(euromillions_data[-1][:5])
    final_prediction = predict_next_tuple(last_row)
    print(f"Prédiction finale : {final_prediction}")

print("Taux de précision atteint : {0:.2f}".format(best_accuracy))
