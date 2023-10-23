import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Lire le CSV
data = pd.read_csv('euromillions.csv', sep=';', header=None)

# Séparer les données en fonction des colonnes d'entrée (X) et des colonnes de sortie (y)
X = data.iloc[:, 0:5]
y = data.iloc[:, 0:5]

# Créer un modèle RandomForestRegressor avec des hyperparamètres prédéfinis
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Utiliser MultiOutputRegressor pour gérer plusieurs sorties
model = MultiOutputRegressor(model)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'estimator__n_estimators': randint(10, 200),  # Nombre d'arbres dans la forêt
    'estimator__max_depth': randint(1, 20),     # Profondeur maximale des arbres
    'estimator__min_samples_split': [2, 5, 10],  # Nombre minimum d'échantillons requis pour diviser un nœud
    'estimator__min_samples_leaf': [1, 2, 4],    # Nombre minimum d'échantillons requis dans une feuille
    'estimator__max_features': ['auto', 'sqrt', 'log2']  # Nombre maximum de fonctionnalités à considérer lors de la recherche de la meilleure division
}


# Créer l'objet RandomizedSearchCV
search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42)

# Effectuer la recherche sur les hyperparamètres
search.fit(X_train, y_train)

# Obtenir les meilleurs hyperparamètres
best_params = search.best_params_

# Obtenir le modèle avec les meilleurs hyperparamètres
best_model = search.best_estimator_

# Prédire les 5 premières colonnes de la dernière ligne du CSV
derniere_ligne = X.iloc[-1, :].values.reshape(1, -1)
prediction = best_model.predict(derniere_ligne)

# Calculer la précision de la prédiction sur l'ensemble de test (RMSE pour chaque sortie)
y_pred = best_model.predict(X_test)
precision = mean_squared_error(y_test, y_pred, squared=False)

# Imprimer les meilleurs hyperparamètres, la prédiction et la précision
print("Meilleurs hyperparamètres:", best_params)
print("Prédiction pour la dernière ligne du CSV:", prediction)
print("Précision de la prédiction sur l'ensemble de test (RMSE):", precision)
