import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from skopt import BayesSearchCV
from skopt.space import Integer

# Chargement des données
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row[:5]))
        data.append(numbers)

# Transformation des données en une série chronologique pandas
data = np.array(data)
date_range = pd.date_range(start='01-01-2004', periods=len(data), freq='W')
time_series = pd.Series(data[:, 0], index=date_range)

# Préparation des données pour l'apprentissage
def prepare_data_for_lstm(data, look_back=5):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 5
X, Y = prepare_data_for_lstm(time_series.values, look_back)

# Création de la structure du modèle LSTM
def create_lstm_model(look_back, units=50):
    model = Sequential()
    model.add(LSTM(units, input_shape=(look_back, 1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Recherche des meilleurs hyperparamètres avec skopt
param_space = {
    'batch_size': Integer(1, 32),
    'epochs': Integer(10, 100)
}

# Créez un objet KerasClassifier compatible avec scikit-learn
keras_classifier = KerasClassifier(build_fn=create_lstm_model, verbose=0)

# Utilisez cet objet dans la recherche des hyperparamètres
opt = BayesSearchCV(keras_classifier, param_space, n_iter=50, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
opt.fit(X, Y)

# Afficher les meilleurs hyperparamètres
best_params = opt.best_params_
print("Meilleurs hyperparamètres:", best_params)

# Entraînez le modèle avec les meilleurs hyperparamètres
best_model = create_lstm_model(look_back)
best_model.fit(X, Y, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Prédire les numéros futurs
forecast_steps = 5
forecast = []

for i in range(forecast_steps):
    input_sequence = time_series.values[-look_back:].reshape(1, look_back, 1)
    predicted_number = best_model.predict(input_sequence)
    forecast.append(int(predicted_number[0][0]))
    time_series = time_series.append(pd.Series([int(predicted_number[0][0])], index=[time_series.index[-1] + pd.DateOffset(1)]))

# Affichage des numéros prédits
print("Séquence prédite de 5 numéros:", forecast)

# Tracé des prédictions
plt.plot(time_series, label='Historical Data')
plt.plot(pd.date_range(start=time_series.index[-1], periods=forecast_steps, freq='W'), forecast, label='Predictions', color='red')
plt.legend()
plt.show()
