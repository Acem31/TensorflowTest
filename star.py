import csv
import random

# Chargement des données
data = []
with open('votre_fichier.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = tuple(map(int, row[:5]))
        extra_numbers = tuple(map(int, row[5:]))
        data.append((numbers, extra_numbers))

# Construction de la matrice de transition
transition_matrix = {}
for i in range(len(data) - 1):
    current_tuple = data[i]
    next_tuple = data[i + 1]

    if current_tuple in transition_matrix:
        if next_tuple in transition_matrix[current_tuple]:
            transition_matrix[current_tuple][next_tuple] += 1
        else:
            transition_matrix[current_tuple][next_tuple] = 1
    else:
        transition_matrix[current_tuple] = {next_tuple: 1}

    # Normaliser les probabilités de transition
    for current_tuple in transition_matrix:
        total_transitions = sum(transition_matrix[current_tuple].values())
        for next_tuple in transition_matrix[current_tuple]:
            transition_matrix[current_tuple][next_tuple] /= total_transitions

# Prédiction des prochains numéros
current_tuple = random.choice(data)
predicted_sequence = list(current_tuple[0]) + list(current_tuple[1])

for _ in range(2):  # Prédire deux numéros
    if current_tuple in transition_matrix:
        next_tuple = random.choices(
            list(transition_matrix[current_tuple].keys()),
            list(transition_matrix[current_tuple].values())
        )[0]
    else:
        next_tuple = random.choice(data)

    predicted_sequence.extend(next_tuple[0])  # Ajouter les chiffres de la série
    predicted_sequence.extend(next_tuple[1])  # Ajouter les chiffres des colonnes 6 et 7
    current_tuple = next_tuple

# Afficher la séquence prédite de deux numéros
predicted_sequence = predicted_sequence[-14:]  # Garder uniquement les 14 derniers chiffres (2 numéros x 7 colonnes)
print(predicted_sequence)
