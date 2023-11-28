import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Charger le fichier CSV avec un délimiteur spécifique pour les nombres et le point-virgule
df = pd.read_csv('euromillions.csv', delimiter='[;|\n]', engine='python')

# Supprimer les lignes avec des valeurs manquantes
df.dropna(inplace=True)

# Diviser les données en caractéristiques d'entrée (X) et de sortie (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Initialiser le modèle de régression neuronal (MLP)
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Initialiser un scaler pour normaliser les données
scaler = StandardScaler()

# Liste pour stocker les tirages précédents
previous_draws = []

# Parcourir chaque ligne du CSV pour l'apprentissage incrémentiel
for index in range(len(df)):
    # Ajouter le tirage actuel à la liste des tirages précédents
    previous_draws.append(X.iloc[index, :].values)

    if len(previous_draws) > 1:  # Vérifier si la liste n'est pas vide
        # Utiliser les tirages précédents pour l'apprentissage
        X_train = previous_draws[:-1]
        y_train = y.iloc[:index]

        # Normaliser les données
        X_train_scaled = scaler.fit_transform(X_train)

        # Mettre à jour le modèle avec la nouvelle ligne
        model.partial_fit(X_train_scaled, y_train)

# Sélectionner la dernière ligne du CSV pour la prédiction future
future_data = [X.iloc[-1, :].values]

# Normaliser les données pour la prédiction
future_data_scaled = scaler.transform(future_data)

# Faire la prédiction pour la future ligne
future_prediction = model.predict(future_data_scaled)
print(f'Prédiction pour la future ligne : {future_prediction}')
