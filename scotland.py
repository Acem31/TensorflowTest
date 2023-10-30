import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterSampler
from sklearn.metrics.pairwise import pairwise_distances

# Chargement des données
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row[:5]))
        data.append(numbers)

# Transformation des données en tableau numpy
X = np.array(data)

# Hyperparamètres initiaux
n_clusters = 10
n_init = 20
init_method = 'k-means++'
random_state = 0

# Critère d'arrêt
stop_condition = False

while not stop_condition:
    # Création du modèle de clustering (K-Means) avec les hyperparamètres actuels
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, n_init=n_init, random_state=random_state).fit(X)

    # Prédiction des 5 premiers numéros de la dernière ligne
    last_row = X[-1][:5]

    # Recherche des exemples dans le même cluster
    similar_rows = [row[:5] for row, label in zip(data, kmeans.labels_) if label == kmeans.predict([last_row])]

    # Si le modèle prédit correctement les 5 premiers numéros, arrêtez la boucle
    if last_row in similar_rows:
        stop_condition = True
        print("Modèle a réussi à prédire les 5 premiers numéros.")
    else:
        # Optimisation des hyperparamètres en utilisant une recherche aléatoire
        param_grid = {
            'n_clusters': range(n_clusters, n_clusters + 10),  # Ajustez cette plage comme vous le souhaitez
            'n_init': range(n_init, n_init + 10),  # Ajustez cette plage comme vous le souhaitez
        }

        param_combinations = list(ParameterSampler(param_grid, n_iter=10, random_state=random_state))

        best_combination = None
        best_score = float('inf')

        for params in param_combinations:
            kmeans_tmp = KMeans(n_clusters=params['n_clusters'], init=init_method, n_init=params['n_init'], random_state=random_state).fit(X)
            dist = pairwise_distances(kmeans_tmp.cluster_centers_, X)
            score = dist.min(axis=1).sum()

            if score < best_score:
                best_score = score
                best_combination = params

        n_clusters = best_combination['n_clusters']
        n_init = best_combination['n_init']

# Afficher les 5 numéros prédits
print("Séquence prédite de 5 numéros:", similar_rows[0])
