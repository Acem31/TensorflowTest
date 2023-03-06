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

# Créer un modèle d'apprentissage automatique
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Évaluer les performances du modèle sur l'ensemble de test
score = model.score(X_test, y_test)
print("Score de précision du modèle :", score)

# Prédire les résultats pour un nouveau jeu de données
nouveaux_resultats = [[7, 9], [2, 5]]  # Exemple de nouveaux résultats pour les numéros étoiles
prediction = model.predict(nouveaux_resultats)
print("Les numéros gagnants prédits sont :", prediction)
