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

# Optimisation des hyperparamètres ARIMA
best_aic = float("inf")
best_order = None

for p in range(5):
    for d in range(2):
        for q in range(5):
            try:
                model = ARIMA(time_series, order=(p, d, q))
                model_fit = model.fit(disp=0)
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                continue

if best_order is not None:
    model = ARIMA(time_series, order=best_order)
    model_fit = model.fit(disp=0)
    # Reste du code pour la prédiction, l'évaluation et l'affichage
else:
    print("Aucun meilleur ordre n'a été trouvé. Veuillez ajuster votre recherche d'hyperparamètres.")
    
# Entraînement du modèle ARIMA avec les meilleurs hyperparamètres
model = ARIMA(time_series, order=best_order)
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
