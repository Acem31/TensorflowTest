import pandas as pd
from sklearn.linear_model import LogisticRegression

# Import des données de l'historique des tirages
df = pd.read_csv("euromillions.csv")

# Préparation des données pour l'entraînement du modèle
X = df.iloc[:, :-5].values  # toutes les colonnes à l'exception des 5 dernières colonnes contenant les numéros
y = df.iloc[:, -5:].values  # les 5 dernières colonnes contenant les numéros

# Division des données en ensembles d'apprentissage et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de Régression Logistique Multinomiale
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Évaluation de la précision du modèle
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Prédiction d'une liste de 5 numéros pour le prochain tirage
last_draw = [3, 12, 21, 26, 34, 1, 11]
last_draw_features = last_draw[:-1]  # Les 5 premiers éléments sont les numéros principaux
next_number_predictions = model.predict([last_draw_features])
print(f"Prédiction: {next_number_predictions[0]}")
