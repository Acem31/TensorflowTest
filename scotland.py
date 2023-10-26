import csv
import random
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Fonction pour lire les données du fichier CSV
def read_euromillions_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            numbers = list(map(int, row[:5]))
            data.append(numbers)
    return data

# Fonction objectif pour Optuna
def objective(trial):
    # Séparation des données en caractéristiques (X) et cible (y)
    X = [row[:-1] for row in euromillions_data]
    y = [row[-1] for row in euromillions_data]

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Paramètres à optimiser
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 10, 30)

    # Initialisation du modèle de régression avec les paramètres suggérés par Optuna
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédiction du dernier tuple
    last_tuple = X_test[-1]
    predicted_last_value = model.predict([last_tuple])[0]

    # Calcul du score
    mse = mean_squared_error([y_test[-1]], [predicted_last_value])
    accuracy = 1 - mse / np.var(y)

    return -mse  # Minimiser l'erreur quadratique moyenne
    
predicted_last_value = None  # Initialisation en dehors de la boucle

if __name__ == "__main__":
    file_path = 'euromillions.csv'
    euromillions_data = read_euromillions_data(file_path)

    # Création d'une étude Optuna
    study = optuna.create_study(direction='minimize')

    while True:
        # Optimisation des hyperparamètres avec Optuna
        study.optimize(objective, n_trials=100)

        best_params = study.best_params
        best_score = -study.best_value

        print(f"Meilleurs hyperparamètres : {best_params}")
        print(f"Meilleur score de précision : {best_score * 100}%")

        if best_score >= 0.5:
            break

    # Afficher la dernière ligne du CSV
    last_actual_value = euromillions_data[-1][-1]
    print(f"Dernière ligne du CSV : {euromillions_data[-1]}")

    # Prédiction du dernier tuple pour l'itération actuelle
    predicted_last_value = predict_last_tuple(euromillions_data)[0]

    print(f"Prédiction pour la dernière ligne : {predicted_last_value}")
    print(f"Score de précision actuel : {best_score * 100}%")
