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

# Créer un modèle d'apprentissage automatique avec les noms de colonnes spécifiés
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train, feature_names=["etoile1", "etoile2"])

# Évaluer les performances du modèle sur l'ensemble de test
score = model.score(X_test, y_test)
print("Score de précision du modèle :", score)

# Prédire les résultats pour un nouveau jeu de données
derniere_ligne = data.tail(1)
nouveaux_resultats = [[derniere_ligne["etoile1"].values[0], derniere_ligne["etoile2"].values[0]],
                      [derniere_ligne["num1"].values[0], derniere_ligne["num2"].values[0], 
                       derniere_ligne["num3"].values[0], derniere_ligne["num4"].values[0], 
                       derniere_ligne["num5"].values[0]]]
prediction_etoiles = model.predict(nouveaux_resultats[0])
prediction_numeros = model.predict(nouveaux_resultats[1])
print("Les numéros étoiles prédits sont :", prediction_etoiles)
print("Les numéros gagnants prédits sont :", prediction_numeros)
