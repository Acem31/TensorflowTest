import csv
from collections import defaultdict
from random import choices

# Lecture du fichier CSV
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    next(reader)  # Ignorer l'en-tête du fichier CSV
    data = [tuple(map(int, row[-2:])) for row in reader]

# Construction de la matrice de transition
transition_matrix = defaultdict(list)
for i in range(len(data) - 1):
    current_state = data[i]
    next_state = data[i + 1]
    transition_matrix[current_state].append(next_state)

# Fonction pour prédire la prochaine série de numéros
def predict_next_series():
    current_state = data[-1]  # Utiliser la dernière série connue
    next_series = choices(transition_matrix[current_state])[0]
    return next_series

# Prédiction de la prochaine série
next_series = predict_next_series()
print("Prochaine série prédite :", next_series)
