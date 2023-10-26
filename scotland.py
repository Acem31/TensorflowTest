import csv
import random
import numpy as np
import optuna
from bayes_opt import BayesianOptimization
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

    n_jobs = 5

    # Initialisation du modèle de régression avec les paramètres suggérés par Optuna
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        n_jobs=n_jobs
    )
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

    # Création d'une étude d'optimisation bayésienne
    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            "n_estimators": (10, 500),
            "max_depth": (5, 50),
            "min_samples_split": (0.1, 1.0),
            "min_samples_leaf": (0.1, 0.5),
            "max_features": (1, len(euromillions_data[0])-1),
            "bootstrap": (0, 1),
        },
        random_state=42,
    )
    
    best_score = None

    while True:
        # Optimisation des hyperparamètres avec BayesianOptimization
        optimizer.maximize(init_points=5, n_iter=10)
        
        best_params = optimizer.max['params']
        current_best_score = -optimizer.max['target']
        
        print(f"Meilleurs hyperparamètres : {best_params}")
        print(f"Meilleur score de précision : {current_best_score * 100}%")
    
        # Si le score n'augmente pas, vous pouvez définir une condition d'arrêt personnalisée
        if best_score is not None and current_best_score <= best_score:
            print("Arrêt de l'optimisation : le score n'augmente plus.")
            break
    
        best_score = current_best_score
    
        # Afficher la dernière ligne du CSV
        last_actual_value = euromillions_data[-1][-1]
        print(f"Dernière ligne du CSV : {euromillions_data[-1]}")
    
        # Prédiction du dernier tuple pour l'itération actuelle
        predicted_last_value = predict_last_tuple(euromillions_data)[0]
    
        print(f"Prédiction pour la dernière ligne : {predicted_last_value}")
        print(f"Score de précision actuel : {best_score * 100}%")
    
    # Afficher les meilleurs hyperparamètres une fois la boucle terminée
    print("Meilleurs hyperparamètres finaux :")
    print(best_params)
