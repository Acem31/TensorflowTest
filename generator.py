import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Lire le CSV
data = pd.read_csv('euromillions.csv', sep=';', header=None)

# Séparer les données en fonction des colonnes d'entrée (X) et de la colonne de sortie (y)
X = data.iloc[:, 0:5]
y = data.iloc[:, 5]

# Créer un modèle RandomForestClassifier avec des hyperparamètres prédéfinis
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédire les 5 premières colonnes de la dernière ligne du CSV
derniere_ligne = X.iloc[-1, :].values.reshape(1, -1)
prediction = model.predict(derniere_ligne)

# Calculer la précision de la prédiction sur l'ensemble de test
y_pred = model.predict(X_test)
precision = accuracy_score(y_test, y_pred)

# Imprimer la prédiction et la précision
print("Prédiction pour la dernière ligne du CSV :", prediction)
print("Précision de la prédiction sur l'ensemble de test :", precision)
