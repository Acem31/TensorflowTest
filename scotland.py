import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

# Charger les données depuis le fichier CSV (5 premières colonnes)
data = pd.read_csv('euromillions.csv', delimiter=';', usecols=range(5))

# Définir une fonction pour l'optimisation bayésienne des hyperparamètres
def optimize_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)

    # Créer le modèle RandomForestClassifier avec les hyperparamètres
    model = RandomForestClassifier(n_estimators=int(n_estimators), 
                                  max_depth=int(max_depth), 
                                  min_samples_split=int(min_samples_split), 
                                  min_samples_leaf=int(min_samples_leaf), 
                                  random_state=42)

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer le taux de précision
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Définir les limites des hyperparamètres pour l'optimisation bayésienne
pbounds = {'n_estimators': (10, 200),
           'max_depth': (1, 32),
           'min_samples_split': (2, 20),
           'min_samples_leaf': (1, 20)}

# Initialiser l'optimiseur bayésien
optimizer = BayesianOptimization(f=optimize_rf, pbounds=pbounds, random_state=42)

# Boucle pour optimiser les hyperparamètres jusqu'à atteindre un taux de précision de 50%
best_accuracy = 0
while best_accuracy < 0.5:
    # Optimiser les hyperparamètres
    optimizer.maximize(init_points=5, n_iter=10, acq='ei')
    
    # Obtenir les meilleurs hyperparamètres
    best_params = optimizer.max['params']
    
    # Afficher les meilleurs hyperparamètres
    print("Meilleurs hyperparamètres:", best_params)
    
    # Entraîner le modèle avec les meilleurs hyperparamètres sur l'ensemble de données complet
    best_model = RandomForestClassifier(n_estimators=int(best_params['n_estimators']), 
                                       max_depth=int(best_params['max_depth']), 
                                       min_samples_split=int(best_params['min_samples_split']), 
                                       min_samples_leaf=int(best_params['min_samples_leaf']), 
                                       random_state=42)
    best_model.fit(data.iloc[:, :-1], data.iloc[:, -1])
    
    # Lire la dernière ligne du CSV pour la prédiction
    last_row = data.iloc[[-1], :-1]
    actual_result = data.iloc[-1, -1]
    
    # Faire une prédiction sur la dernière ligne
    prediction = best_model.predict(last_row)
    
    # Calculer le taux de précision
    last_accuracy = accuracy_score([actual_result], prediction)
    
    # Afficher le taux de précision de la prédiction
    print("Taux de précision de la prédiction:", last_accuracy)
    
    # Mettre à jour le meilleur taux de précision
    best_accuracy = max(best_accuracy, last_accuracy)

print("Objectif de taux de précision atteint (50% ou plus)!")
