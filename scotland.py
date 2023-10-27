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

# Diviser les données en X_train, X_test (80% des données) et y_train, y_test (dernière ligne)
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:-1, :-1], data.iloc[:-1, -1:], test_size=0.2, random_state=42)

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
    lgb.LGBMClassifier(),  # Utiliser LGBMClassifier pour une tâche de classification
    param_space,
    n_iter=50,
    cv=5,
    scoring='accuracy',  # Métrique de classification
    random_state=42
)

# Créer une liste pour stocker les modèles
models = []

# Entraîner un modèle LightGBM pour chaque colonne
for i in range(5):
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train.iloc[:, i])
    models.append(model)

# Prédire la dernière ligne du CSV avec chaque modèle
predictions = [model.predict(X_test) for model in models]

# Afficher les résultats
print("Meilleurs hyperparamètres LightGBM:", best_params)
print("Prédiction pour la dernière ligne du CSV (les 5 colonnes) :", predictions)
