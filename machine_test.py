# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Charger les données à partir d'un fichier CSV
data = pd.read_csv("euromillions.csv", header=None, delimiter=";")

# Renommer les colonnes
data.columns = ["num1", "num2", "num3", "num4", "num5", "etoile1", "etoile2"]

# Diviser les données en ensemble d'entraînement et ensemble de test
X = data.drop(["num1", "num2", "num3", "num4", "num5"], axis=1)  # Les caractéristiques sont les deux numéros étoiles
y = data[["num1", "num2", "num3", "num4", "num5"]]  # La variable cible sont les cinq numéros principaux
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer un modèle d'apprentissage automatique pour la prédiction des numéros principaux
model_principaux = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
model_principaux.fit(X_train, y_train)

# Prédire les numéros principaux pour le prochain tirage
prochain_tirage = pd.DataFrame(columns=["etoile1", "etoile2"])
prochain_tirage.loc[0] = [3, 8]  # Exemple de numéros étoiles pour le prochain tirage
prediction_principaux = model_principaux.predict(prochain_tirage)
print("Les numéros principaux prédits pour le prochain tirage sont :", prediction_principaux)
