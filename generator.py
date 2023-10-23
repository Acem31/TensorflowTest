import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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

    # Créer un modèle (Random Forest, par exemple)
    model = RandomForestClassifier()

    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None],  # Ajout de 'log2' et 'None'
        'bootstrap': [True, False]
    }


    # Utiliser GridSearchCV pour rechercher les meilleurs hyperparamètres
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Obtenir les meilleurs hyperparamètres
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Faire des prédictions sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # Calculer la précision en pourcentage
    accuracy = accuracy_score(y_test, y_pred) * 100

    last_row = data.iloc[-1].values[:-2]
    prediction = best_model.predict([last_row])[0]

    print(f"Itération {iteration} - Taux de précision : {accuracy:.2f}%")
    print("Dernière ligne du CSV :", last_row)
    print("Prédiction pour la dernière ligne : ", prediction)

    if accuracy > best_accuracy:
        best_accuracy = accuracy

    # Mettre à jour les hyperparamètres pour la prochaine boucle
    param_grid['n_estimators'] = [n * 2 for n in param_grid['n_estimators']]
    param_grid['max_depth'] = [d + 10 if d is not None else None for d in param_grid['max_depth']]

# Réentraîner le modèle en incluant la dernière ligne
best_model.fit(X, y)
last_row = X.iloc[[-1]]
prediction = best_model.predict(last_row)
print("Dernière ligne du CSV :")
print(data.iloc[-1])
print("Prédiction pour la dernière ligne : ", prediction[0])
print("Taux de précision final : {0:.2f}%".format(best_accuracy))
