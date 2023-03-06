# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Charger les données à partir d'un fichier CSV
data = pd.read_csv("euromillions.csv")

# Diviser les données en ensemble d'entraînement et ensemble de test
X = data.drop("numeros_gagnants", axis=1)  # Les caractéristiques sont toutes sauf le résultat final
y = data["numeros_gagnants"]  # La variable cible est le résultat final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer un modèle d'apprentissage automatique
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Évaluer les performances du modèle sur l'ensemble de test
score = model.score(X_test, y_test)
print("Score de précision du modèle :", score)

# Prédire les résultats pour un nouveau jeu de données
nouveaux_resultats = [[12, 29, 33, 37, 47], [5, 10]]  # Exemple de nouveaux résultats
prediction = model.predict(nouveaux_resultats)
print("Les numéros gagnants prédits sont :", prediction)
