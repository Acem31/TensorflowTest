import csv
import numpy as np
from sklearn.cluster import KMeans
from skopt import BayesSearchCV
from sklearn.metrics import silhouette_score

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
n_init = 20

opt = BayesSearchCV(
    KMeans(),
    {
        'n_init': (10, 300),
        'max_iter': (10, 200),  # Ajoutez max_iter
    },
    random_state=0
)

# Meilleur K trouvé avec le coefficient de silhouette
best_silhouette_score = -1
best_k = None

for k in range(2, 200):
    kmeans = KMeans(n_clusters=k, n_init=best_params['n_init'], max_iter=best_params['max_iter'], random_state=43, init='k-means++').fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_k = k

# Maintenant, utilisez le meilleur K dans la boucle while
stop_condition = False

while not stop_condition:
    # Optimisation des hyperparamètres avec l'optimiseur bayésien
    opt.fit(X)
    best_params = opt.best_params_

    # Création du modèle de clustering (K-Means) avec les meilleurs hyperparamètres
    kmeans = KMeans(n_clusters=best_k, n_init=best_params['n_init'], max_iter=best_params['max_iter'], random_state=43, verbose=1).fit(X)
    
    # Prédiction des 5 premiers numéros de la dernière ligne
    last_row = X[-1][:5]

    # Recherche des exemples dans le même cluster
    similar_rows = [row[:5] for row, label in zip(data, kmeans.labels_) if label == kmeans.predict([last_row])]

    # Charger la dernière ligne du CSV sous forme de liste de nombres
    with open('euromillions.csv', 'r') as file:
        last_line = list(file.readlines())[-1]
        last_line_csv = [int(num) for num in last_line.strip().split(';')]

    if np.array_equal(last_row, last_line_csv):
        stop_condition = True
        print("Modèle a réussi à prédire les 5 premiers numéros.")

# Afficher les 5 numéros prédits
print("Séquence prédite de 5 numéros:", similar_rows[0])
