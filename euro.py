import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Chargement des données depuis le fichier CSV
data = pd.read_csv("euromillions.csv", sep=";", header=None)

# Sélection des 5 premières colonnes
data = data.iloc[:, :5]

# Séparation des fonctionnalités (X) et de la variable cible (y)
X = data.iloc[:, :4]
y = data.iloc[:, 4]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choix du modèle et entraînement
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Prédiction avec le modèle optimisé
new_data = pd.DataFrame([[16, 29, 32, 36], [7, 13, 39, 47]], columns=X.columns)
predictions = model.predict(new_data)
print("Prédictions:", predictions)
