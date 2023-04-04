import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Charger les données
data = pd.read_csv("euromillions.csv")

# Trier les numéros
data.iloc[:, :5] = data.iloc[:, :5].apply(lambda x: sorted(map(int, x.split(';'))) if isinstance(x, str) else x, axis=1).apply(pd.Series)


# Récupérer le dernier tirage
last_draw = data.iloc[-1, :6]

# Sélectionner les données d'entraînement en excluant le dernier tirage
X_train = data.iloc[:-1, :6].apply(lambda x: list(map(int, x.str.split(';').str[0:5].tolist())), axis=1)
y_train = data.iloc[:-1, 6:].apply(lambda x: list(map(int, x.str.split(';').str[0:2].tolist())), axis=1)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Ajouter le dernier tirage aux données de test
X_test = scaler.transform([last_draw.str.split(';').str[0:5].apply(lambda x: list(map(int, x))).tolist()])

# Créer un réseau de neurones
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=[6]),
    layers.Dense(64, activation="relu"),
    layers.Dense(5, activation="linear"),
    layers.Dense(2, activation="linear")
])

# Compiler le modèle
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss="mean_absolute_error"
)

# Entraîner le modèle
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    verbose=1
)

# Obtenir les prédictions pour le dernier tirage
predictions = model.predict(X_test)

# Afficher les prédictions
print(predictions)
