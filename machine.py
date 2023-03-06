# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Charger les données à partir d'un fichier CSV
data = pd.read_csv("euromillions.csv", header=None, delimiter=";")

# Renommer les colonnes
data.columns = ["num1", "num2", "num3", "num4", "num5", "etoile1", "etoile2"]

# Sélectionner les numéros principaux de la dernière ligne pour la prédiction des numéros étoile
derniere_ligne_principaux = data[["num1", "num2", "num3", "num4", "num5"]].iloc[-1:].values

# Sélectionner les numéros étoile de la dernière ligne pour la prédiction des numéros principaux
derniere_ligne_etoiles = data[["etoile1", "etoile2"]].iloc[-1:].values

# Diviser les données en ensemble d'entraînement et ensemble de test
X = data.drop(["num1", "num2", "num3", "num4", "num5"], axis=1)  # Les caractéristiques sont les deux numéros étoiles
y = data[["num1", "num2", "num3", "num4", "num5"]]  # La variable cible sont les cinq numéros principaux
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer un modèle d'apprentissage automatique pour la prédiction des numéros étoile
model_etoiles = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
model_etoiles.fit(y_train, X_train)

# Prédire les numéros étoile pour la dernière ligne de données
prediction_etoiles = model_etoiles.predict(derniere_ligne_principaux)
print("Les numéros étoile prédits sont :", prediction_etoiles)

# Créer un modèle d'apprentissage automatique pour la prédiction des numéros principaux
model_principaux = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
model_principaux.fit(X_train, y_train)

# Prédire les numéros principaux pour la dernière ligne de données
prediction_principaux = model_principaux.predict(derniere_ligne_etoiles)
print("Les numéros principaux prédits sont :", prediction_principaux)

score_etoiles = model_etoiles.score(y_test, X_test)
score_principaux = model_principaux.score(X_test, y_test)
print("Score pour la prédiction des numéros étoile :", score_etoiles)
print("Score pour la prédiction des numéros principaux :", score_principaux)
