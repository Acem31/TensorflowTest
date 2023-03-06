import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Lecture du fichier csv
data = pd.read_csv("euro_millions.csv")

# Séparation des numéros et des étoiles
numbers = data.iloc[:, :7]
stars = data.iloc[:, 7:]

# Conversion des numéros et des étoiles en entiers
numbers = numbers.applymap(int)
stars = stars.applymap(int)

# Concaténation des numéros et des étoiles
data_processed = pd.concat([numbers, stars], axis=1)

# Conversion des données en listes
X = data_processed.values.tolist()
y = data_processed.values.tolist()

# Séparation des données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Définition du modèle
model = Sequential()
model.add(Dense(50, input_dim=14, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(7, activation='linear'))

# Compilation du modèle
model.compile(loss='mean_squared_error', optimizer='adam')

# Définition du callback early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Entraînement du modèle
model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping])

# Évaluation du modèle
loss = model.evaluate(X_test, y_test)
print(f"Loss : {loss}")
