import numpy as np
import pandas as pd
from sklearn.neighbors import NearestCentroid
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données à partir du CSV
data = pd.read_csv('votre_fichier.csv', header=None, delimiter=';')

# Garder uniquement les 5 premiers numéros de chaque ligne
data = data.iloc[:, :5]

# Définir X (caractéristiques) et y (étiquettes)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Initialiser le modèle LVQ
model = NearestCentroid()

# Créer un espace d'hyperparamètres pour l'optimisation bayésienne
param_space = {
    'shrink_threshold': Real(0.0, 2.0),
    'n_neighbors': Integer(1, 10),
    'split_ratio': Real(0.1, 0.9),
    'initialization_method': Categorical(['random', 'kmeans']),
    'distance_metric': Categorical(['euclidean', 'manhattan']),
    'learning_rate': Real(0.01, 0.5),
    'max_iter': Integer(10, 100),
}

# Initialiser l'optimisation bayésienne
opt = BayesSearchCV(
    model,
    param_space,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42  # Ajoutez ceci pour rendre les résultats reproductibles
)

accuracy = 0

while accuracy < 0.5:
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Effectuer l'optimisation bayésienne des hyperparamètres
    opt.fit(X_train, y_train)

    # Obtenir les meilleurs paramètres
    best_params = opt.best_params_

    # Entraîner le modèle LVQ avec les meilleurs paramètres
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)

# Prédire la dernière ligne du CSV
last_row = data.iloc[-1, :-1].values.reshape(1, -1)
prediction = model.predict(last_row)

# Comparer avec les vrais numéros et calculer la précision
true_numbers = data.iloc[-1, -1]
correct_predictions = (prediction == true_numbers).sum()
precision = correct_predictions / len(true_numbers)

# Afficher les résultats
print("Modèle LVQ avec les meilleurs hyperparamètres:", best_params)
print("Précision du modèle:", accuracy)
print("Prédiction:", prediction)
print("Vrais numéros:", true_numbers)
print("Précision de la prédiction:", precision)
