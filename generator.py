import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow import keras
from tensorflow.keras import layers

# Charger le fichier CSV
data = pd.read_csv('euromillions.csv', delimiter=';', header=None)
X = data.iloc[:, :-2]  # Sélectionner les 5 premières colonnes
y = data.iloc[:, -1]  # Dernière colonne à prédire

best_accuracy = 0.0
best_params = {}
iteration = 0

while best_accuracy < 30:
    iteration += 1
    # Diviser les données en ensemble d'apprentissage et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Définir la fonction de modèle Keras pour Keras Tuner
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Dense(units=hp.Int('units', min_value=1, max_value=50, step=1), activation='softmax'))
        model.add(layers.Dense(50, activation='softmax'))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Ajouter l'hyperparamètre 'epochs'
        epochs = hp.Int('epochs', min_value=5, max_value=30, step=5)
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
        return model

    # Configurer le tuner Keras
    tuner = RandomSearch(
        build_model,
        objective='val_acc',
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

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1)) * 100

    last_row = data.iloc[-1].values[:-2]
    prediction = model.predict(np.array([last_row]))[0]

    print(f"Itération {iteration} - Taux de précision : {accuracy:.2f}%")
    print("Dernière ligne du CSV :", last_row)
    print("Prédiction pour la dernière ligne : ", np.argmax(prediction))

    if accuracy > best_accuracy:
        best_accuracy = accuracy

# Réentraîner le modèle en incluant la dernière ligne
model.fit(X, y, epochs=best_hps.get('epochs'))
last_row = X.iloc[[-1]]
prediction = model.predict(np.array(last_row))[0]
print("Dernière ligne du CSV :")
print(data.iloc[-1])
print("Prédiction pour la dernière ligne : ", np.argmax(prediction))
print("Taux de précision final : {0:.2f}%".format(best_accuracy))
