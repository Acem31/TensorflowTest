import numpy as np
import pandas as pd
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split

# Charger les données à partir du CSV
data = pd.read_csv('euromillions.csv', header=None, delimiter=';')

# Garder uniquement les 5 premières colonnes de chaque ligne
data = data.iloc[:, :5]

# Séparer les données en X et y
X = data.iloc[:-1, :]  # Les 5 premières colonnes de toutes les lignes, sauf la dernière
y = data.iloc[-1, :]   # Les 5 premières colonnes de la dernière ligne

# Créer un espace d'hyperparamètres pour l'optimisation bayésienne
param_space = {
    'num_leaves': Integer(30, 1000),
    'learning_rate': Real(0.01, 0.5),
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(3, 50),
    'min_child_samples': Integer(2, 100),
    'subsample': Real(0.1, 1.0),
    'colsample_bytree': Real(0.1, 1.0),
    'reg_alpha': Real(0.0, 10.0),
    'reg_lambda': Real(0.0, 10.0),
}

# Initialiser l'optimisation bayésienne
opt = BayesSearchCV(
    lgb.LGBMRegressor(),  # Utiliser LGBMRegressor pour une tâche de régression
    param_space,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',  # Métrique de régression
    random_state=42
)

# Effectuer l'optimisation bayésienne des hyperparamètres
opt.fit(X, y)

# Obtenir les meilleurs paramètres
best_params = opt.best_params_

# Entraîner le modèle LightGBM avec les meilleurs paramètres
model = lgb.LGBMRegressor(**best_params)
model.fit(X, y)

# Prédire la dernière ligne du CSV
last_row = data.iloc[-1, :].values.reshape(1, -1)
prediction = model.predict(last_row)

# Afficher les résultats
print("Meilleurs hyperparamètres LightGBM:", best_params)
print("Prédiction pour la dernière ligne du CSV:", prediction)
