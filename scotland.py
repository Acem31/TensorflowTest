import numpy as np
import pandas as pd
import pymc3 as pm
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données depuis le CSV
data = pd read_csv('euromillions.csv', header=None, sep=';')

# Sélectionner les 5 premières colonnes
data = data.iloc[:, :5]

# Diviser les données en caractéristiques (X) et cibles (y)
X = data.iloc[:-1, :-1].values
y = data.iloc[:-1, -1].values
X_pred = data.iloc[-1, :-1].values
y_true = data.iloc[-1, -1]

# Initialiser le taux de précision à 0
accuracy = 0

# Nombre de nœuds (num_nodes)
num_nodes = 10

# Boucle d'apprentissage et d'optimisation
while accuracy < 0.5:
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with pm.Model() as bbn_model:
        # Créer des nœuds/variables en fonction de num_nodes
        variables = [pm.Normal(f'variable_{i}', mu=0, sigma=1) for i in range(num_nodes)]

        # Définir les variables observées en fonction des données d'entraînement
        observations = [pm.Normal(f'observation_{i}', mu=variables[i], sigma=0.1, observed=X_train[:, i]) for i in range(num_nodes)]

    # Optimisation des hyperparamètres avec Bayesian Optimization
    param_space = {
        'learning_rate': (0.001, 0.1),  # Taux d'apprentissage si applicable
        'num_iterations': (100, 1000),  # Nombre d'itérations d'apprentissage
        'regularization_weight': (0.001, 0.1),  # Poids de régularisation
        'num_mcmc_samples': (100, 1000),  # Nombre d'échantillons MCMC pour l'inférence
        'threshold': (0.1, 0.9)  # Seuil de décision pour la classification si applicable
        # Vous pouvez ajouter d'autres hyperparamètres ici
    }

    bbn_search = BayesSearchCV(bbn_model, param_space, n_iter=50, cv=5, random_state=42)

    # Entraîner le modèle avec les données d'entraînement
    bbn_search.fit(X_train, y_train)

    # Faire des prédictions sur les données de test
    y_pred = bbn_search.predict(X_test)

    # Calculer le taux de précision
    accuracy = accuracy_score(y_test, y_pred)

    # Faire une prédiction avec le modèle actuel sur les données de prédiction
    y_pred_current = bbn_search.predict(X_pred.reshape(1, -1))

    # Afficher la prédiction actuelle
    print("Prédiction actuelle:", y_pred_current)

# Faire une prédiction sur la dernière ligne du CSV
y_pred_final = bbn_search.predict(X_pred.reshape(1, -1))

# Calculer le taux de précision final
final_accuracy = accuracy_score([y_true], y_pred_final)

# Afficher les valeurs du modèle et son taux de précision
print("Modèle BBN entraîné avec succès.")
print("Taux de précision final : {:.2%}".format(final_accuracy))
