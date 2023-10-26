import csv
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fonction pour lire les données du fichier CSV
def read_euromillions_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            numbers = list(map(int, row[:5]))  # 5 premières colonnes sont les numéros
            result = int(row[5])  # Dernière colonne est le résultat
            data.append((numbers, result))
    return data

# Fonction objectif pour BayesianOptimization
def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap):
    # Séparation des données en caractéristiques (X) et cible (y)
    X = [row[0] for row in data]  # Caractéristiques (les numéros)
    y = [row[1] for row in data]  # Cible (résultat)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialisation du modèle de classification avec les hyperparamètres suggérés par BayesianOptimization
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=int(max_features),
        bootstrap=bool(bootstrap),
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation du modèle (précision)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

if __name__ == "__main":
    # Lecture des données
    data = read_euromillions_data('euromillions.csv')

    # Initialisation des variables
    target_accuracy = 0.5  # Taux de précision cible
    best_accuracy = 0.0  # Initialisation
    best_params = None
    iteration = 0

    while best_accuracy < target_accuracy:
        iteration += 1

        # Création d'une étude d'optimisation bayésienne
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={
                "n_estimators": (10, 500),
                "max_depth": (5, 50),
                "min_samples_split": (0.1, 1.0),
                "min_samples_leaf": (0.1, 0.5),
                "max_features": (1, len(data[0][0])),
                "bootstrap": (0, 1),
            },
            random_state=42,
        )

        # Optimisation des hyperparamètres avec BayesianOptimization
        optimizer.maximize(init_points=5, n_iter=10)

        best_params = optimizer.max['params']
        best_accuracy = optimizer.max['target']

        print(f"Iteration {iteration}:")
        print("Meilleurs hyperparamètres :", best_params)
        print("Meilleure précision :", best_accuracy * 100, "%")

        # Afficher la dernière ligne du CSV
        last_row = data[-1]
        print("Dernière ligne du CSV :", last_row[0])

        # Prédiction avec les hyperparamètres optimaux
        X = [row[0] for row in data]
        y = [row[1] for row in data]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(
            n_estimators=int(best_params["n_estimators"]),
            max_depth=int(best_params["max_depth"]),
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=int(best_params["max_features"]),
            bootstrap=bool(best_params["bootstrap"]),
            random_state=42
        )
        model.fit(X_train, y_train)
        predicted_value = model.predict([last_row[0]])
        print("Prédiction avec les hyperparamètres optimaux :", predicted_value)

    print("Taux de précision cible atteint !")
