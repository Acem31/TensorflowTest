import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Charger les données
data = pd.read_csv("euromillions.csv", header=None, usecols=[0, 1, 2, 3, 4])

# Séparer les données en entrée et en sortie
X = data.drop(columns=['n1', 'n2', 'n3', 'n4', 'n5'])
y = data[['n1', 'n2', 'n3', 'n4', 'n5']]

# Séparer les données en ensemble d'entraînement et ensemble de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle
model = LogisticRegression(max_iter=10000)

# Entraîner le modèle
model.fit(X_train, y_train)

# Évaluer la précision du modèle sur l'ensemble de validation
accuracy = model.score(X_val, y_val)
print(f'Accuracy: {accuracy}')

# Prédire les 5 numéros de la prochaine combinaison
next_draw = X.tail(1)
next_numbers = model.predict(next_draw)
print(f'Prédiction: {list(next_numbers[0])}')
