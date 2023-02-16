import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Charger les données
data = pd.read_csv("euromillions.csv")

# Séparer les données en entrées (X) et sorties (y)
X = data.iloc[:, :6]
y = data.iloc[:, 6:]

# Normaliser les données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Séparer les données en ensembles de formation et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=50,
    verbose=1
)

# Obtenir les prédictions pour les données de test
predictions = model.predict(X_test)

# Afficher les prédictions
print(predictions)
