import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

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

# Définir une fonction pour évaluer les hyperparamètres
def evaluate_arima(p, d, q, trend):
    try:
        model = ARIMA(data_values, order=(p, d, q), trend=trend)
        model_fit = model.fit()
        forecast_steps = 5
        forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)
        historical_data = time_series[-forecast_steps:]
        rmse = np.sqrt(mean_squared_error(historical_data, forecast))
        return rmse
    except:
        return float("inf")

# Recherche des meilleurs hyperparamètres
best_rmse = float("inf")
best_order = None

for p in range(5):
    for d in range(2):
        for q in range(5):
            for trend in ['n', 'c', 't', 'ct']:
                rmse = evaluate_arima(p, d, q, trend)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = (p, d, q, trend)

if best_order is not None:
    best_p, best_d, best_q, best_trend = best_order
    model = ARIMA(data_values, order=(best_p, best_d, best_q), trend=best_trend)
    model_fit = model.fit()

    # Prédiction des numéros futurs
    forecast_steps = 5
    forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)

    # Affichage des numéros prédits
    print("Séquence prédite de 5 numéros:", forecast)

    # Tracé des prédictions
    plt.plot(time_series, label='Historical Data')
    plt.plot(pd.date_range(start=time_series.index[-1], periods=forecast_steps, freq='W'), forecast, label='Predictions', color='red')
    plt.fill_between(pd.date_range(start=time_series.index[-1], periods=forecast_steps, freq='W'), conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.show()
else:
    print("Aucun meilleur ordre n'a été trouvé. Veuillez ajuster votre recherche d'hyperparamètres.")
