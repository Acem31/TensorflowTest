import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV

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
data_values = time_series.values

# Définir l'espace des hyperparamètres
param_space = {
    'p': (0, 30),
    'd': (0, 30),
    'q': (0, 30),
    'trend': ['n', 'c', 't', 'ct']
}

opt = BayesSearchCV(
    ARIMA(data_values, order=(0, 0, 0)),  # Modèle initial pour obtenir un objet ARIMA
    param_space,
    n_iter=50,  # Nombre d'itérations d'optimisation, ajustez au besoin
    cv=5,  # Nombre de validations croisées, ajustez au besoin
    n_jobs=-1,  # Utiliser tous les cœurs du processeur
)

opt.fit(data_values)
best_params = opt.best_params_

# Entraînement du modèle ARIMA avec les meilleurs hyperparamètres
best_p, best_d, best_q, best_trend = best_params['p'], best_params['d'], best_params['q'], best_params['trend']
model = ARIMA(data_values, order=(best_p, best_d, best_q), trend=best_trend)
model_fit = model.fit(disp=0)

# Prédiction des numéros futurs
forecast_steps = 5
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)

# Évaluation de la performance du modèle (ex. RMSE)
historical_data = time_series[-forecast_steps:]
rmse = np.sqrt(mean_squared_error(historical_data, forecast))
print(f"RMSE: {rmse}")

# Affichage des numéros prédits
print("Séquence prédite de 5 numéros:", forecast)

# Tracé des prédictions
plt.plot(time_series, label='Historical Data')
plt.plot(pd.date_range(start=time_series.index[-1], periods=forecast_steps, freq='W'), forecast, label='Predictions', color='red')
plt.fill_between(pd.date_range(start=time_series.index[-1], periods=forecast_steps, freq='W'), conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.show()
