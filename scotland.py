import csv
import numpy as np
import random
from sklearn.cluster import KMeans

# Chargement des données
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row[:5]))
        data.append(numbers)

# Transformation des données en tableau numpy
X = np.array(data)

# Hyperparamètres du modèle K-Means
n_clusters = 10  # Nombre de clusters
n_init = 20  # Nombre d'initialisations
init_method = 'k-means++'  # Méthode d'initialisation des centroïdes (autres options : 'random', 'k-means++', 'auto', etc.)
distance_metric = 'euclidean'  # Métrique de distance (autres options : 'manhattan', 'cosine', etc.)

# Création du modèle de clustering (K-Means) avec les hyperparamètres
kmeans = KMeans(n_clusters=n_clusters, init=init_method, n_init=n_init, random_state=0).fit(X)

# Prédiction de la dernière ligne
last_row = X[-1].reshape(1, -1)
predicted_cluster = kmeans.predict(last_row)

# Recherche des exemples dans le même cluster
similar_rows = [row for row, label in zip(data, kmeans.labels_) if label == predicted_cluster]

# Sélection d'un exemple aléatoire dans le cluster
predicted_sequence = random.choice(similar_rows)

print("Séquence prédite de 5 numéros:", predicted_sequence)
