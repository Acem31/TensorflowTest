import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Charger le fichier CSV avec un délimiteur spécifique pour les nombres et le point-virgule
df = pd.read_csv('euromillions.csv', delimiter='[;|\n]')

# Supprimer les lignes avec des valeurs manquantes
df.dropna(inplace=True)

# Diviser les données en caractéristiques d'entrée (X) et de sortie (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Initialiser le modèle de régression linéaire avec descente de gradient stochastique (SGD)
model = SGDRegressor()

# Initialiser un scaler pour normaliser les données
scaler = StandardScaler()

# Parcourir chaque ligne du CSV pour l'apprentissage incrémentiel
for index in range(1, len(df)):
    # Données d'entrée
    X_train = X.iloc[:index, :]

    # Donnée de sortie
    y_train = y.iloc[:index]

    # Normaliser les données
    X_train_scaled = scaler.fit_transform(X_train)

    # Mettre à jour le modèle avec la nouvelle ligne
    model.partial_fit(X_train_scaled, y_train)

# Sélectionner la dernière ligne du CSV pour la prédiction future
future_data = X.iloc[[-1], :]

# Normaliser les données pour la prédiction
future_data_scaled = scaler.transform(future_data)

# Faire la prédiction pour la future ligne
future_prediction = model.predict(future_data_scaled)
print(f'Prédiction pour la future ligne : {future_prediction}')
