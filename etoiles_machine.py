# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Charger les données à partir d'un fichier CSV
data = pd.read_csv("euromillions.csv", header=None, delimiter=";")

# Renommer les colonnes
data.columns = ["num1", "num2", "num3", "num4", "num5", "etoile1", "etoile2"]

# Sélectionner les numéros étoile de chaque ligne pour la prédiction des numéros étoile
etoiles = data[["etoile1", "etoile2"]].values

# Sélectionner les numéros principaux de chaque ligne pour la prédiction des numéros étoile
principaux = data[["num1", "num2", "num3", "num4", "num5"]].values

# Diviser les données en ensemble d'entraînement et ensemble de test
X = data.drop(["etoile1", "etoile2"], axis=1)  # Les caractéristiques sont les cinq numéros principaux
y = data[["etoile1", "etoile2"]]  # La variable cible sont les deux numéros étoiles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer un modèle d'apprentissage automatique pour la prédiction des numéros étoile
model_etoiles = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
model_etoiles.fit(X_train, y_train)

# Prédire les numéros étoile pour le prochain tirage
derniere_ligne_principaux = principaux[-1:].reshape(1,-1)
prediction_etoiles = model_etoiles.predict(derniere_ligne_principaux)
print("Les numéros étoile prédits sont :", prediction_etoiles)

score_etoiles = model_etoiles.score(X_test, y_test)
print("Score pour la prédiction des numéros étoile :", score_etoiles)
