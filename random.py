import csv
import random

# Charger les données à partir du fichier CSV
data = []
with open('euromillions.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        data.append([int(cell) for cell in row])

# Fonction pour générer une nouvelle ligne basée sur les tendances observées
def generate_new_row(data):
    new_row = []

    for i in range(len(data[0])):
        values = [row[i] for row in data]
        # Calculer la valeur moyenne et l'écart type des valeurs dans la colonne
        avg = sum(values) / len(values)
        std_dev = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5

        # Générer une nouvelle valeur basée sur une distribution normale
        new_value = int(random.gauss(avg, std_dev))
        new_row.append(new_value)

    return new_row

# Générer une nouvelle ligne
new_row = generate_new_row(data)
print("Nouvelle ligne générée:", new_row)
