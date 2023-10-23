import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Lire le CSV
data = pd.read_csv('euromillions.csv', sep=';', header=None)

# Séparer les données en fonction des colonnes d'entrée (X) et des colonnes de sortie (y)
# X doit contenir les numéros tirés (par exemple, 5 numéros principaux), et y doit contenir les étoiles (par exemple, 2 étoiles).
X = data.iloc[:, 0:5]
y = data.iloc[:, 5:7]  # Assurez-vous d'ajuster ces indices en fonction de votre fichier CSV.

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle RandomForestRegressor avec des hyperparamètres prédéfinis
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Liste pour stocker les prédictions pour chaque colonne
predictions = []

# Entraîner le modèle et faire des prédictions pour chaque colonne
for col in range(2):  # Nous avons 2 colonnes de sortie (étoiles)
    model.fit(X_train, y_train.iloc[:, col])
    y_pred = model.predict(X_test)
    predictions.append(y_pred)

# Calculer la précision pour chaque colonne (RMSE)
accuracies = [mean_squared_error(y_test.iloc[:, col], predictions[col], squared=False) for col in range(2)]

# Imprimer les prédictions et précisions pour chaque colonne
for col in range(2):
    print(f"Prédiction pour la colonne {col + 5} :", predictions[col])  # Nous commençons l'index à partir de 5
    print(f"Précision de la prédiction pour la colonne {col + 5} (RMSE) : {accuracies[col]:.2f}")
