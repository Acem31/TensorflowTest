import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Charger les données
data = pd.read_csv("euromillions.csv", header=None, delimiter=";")
X = data.iloc[:, :5].values # numéros principaux
y = data.iloc[:, 5:].values # numéros étoiles
X_test = X[-1] # dernière ligne pour les prédictions
X = X[:-1]
y = y[:-1]

# 2. Préparer les données
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalisation des données si nécessaire

# 3. Créer le modèle
model = LogisticRegression()

#Convertir y_train
y_train = y_train.ravel()


# 4. Entraîner le modèle
model.fit(X_train, y_train)

# 5. Évaluer le modèle
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')
conf_matrix = confusion_matrix(y_val, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("Confusion matrix:\n", conf_matrix)

# 6. Prédire les résultats
y_test_pred = model.predict([X_test])
print("Prédiction:", y_test_pred[0])
print("Résultat réel:", data.iloc[-1, :].values)
