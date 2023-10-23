import numpy as np
from tensorflow import keras
from data import read_euromillions_data
from parameter import best_hps  # Importer les meilleurs hyperparamètres depuis parameter.py

# Charger les données en utilisant la fonction read_euromillions_data
euromillions_data = read_euromillions_data('euromillions.csv')

# Préparer les données d'entraînement
X = np.array([tuple[:5] for tuple in euromillions_data[:-1]])
y = np.array(euromillions_data[1:])  # Nous prédisons la ligne suivante par rapport à celle précédente

# Initialisation du taux de précision
best_accuracy = 0.0

# Initialisation du nombre d'itérations
iteration = 0

while best_accuracy < 0.3:  # Le seuil est de 30%
    iteration += 1

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
    model.fit(X, y, epochs=50, batch_size=1, verbose=2)

    # Prédire le prochain tuple
    last_tuple = np.array(euromillions_data[-1][:5]).reshape(1, -1)
    prediction = model.predict(last_tuple)

    print(f"Itération {iteration} - Prédiction pour le prochain tuple : ", prediction[0])

    # Calculer la précision (à adapter selon le type de problème)
    # Par exemple, si vous effectuez une classification, utilisez accuracy_score
    # Pour la régression, utilisez une métrique appropriée
    accuracy = 0.0  # Calculez ici votre taux de précision

    print(f"Précision pour l'itération {iteration} : {accuracy:.2f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy

print("Taux de précision atteint : {0:.2f}".format(best_accuracy))
