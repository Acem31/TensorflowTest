import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Charger les données
data = pd.read_csv("euromillions.csv", header=None, delimiter=";")
X = data.iloc[:, :5].values # numéros principaux
y = data.iloc[:, 5:].values # numéros étoiles
X_test = X[-1] # dernière ligne pour les prédictions
X = X[:-1]
y = y[:-1]

# 2. Préparer les données
X_train, X_val, y_train, y_val = train_test_split(X, y[:, 0], test_size=0.2, random_state=42)
# Normalisation des données si nécessaire

# 3. Créer le modèle
model = LogisticRegression()

# 4. Entraîner le modèle
model.fit(X_train, y_train)

# 5. Évaluer le modèle
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

# 6. Prédire les résultats
y_test_pred = model.predict([X_test])
print("Prédiction:", y_test_pred[0])
print("Résultat réel:", data.iloc[-1, :5].values)
