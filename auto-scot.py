import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import csv

# Charger les données depuis le fichier CSV
data = []
with open('euromillions.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        numbers = list(map(int, row[:5]))
        data.append(numbers)

# Créer un DataFrame pandas
df = pd.DataFrame(data, columns=['Num1', 'Num2', 'Num3', 'Num4', 'Num5'])

# Binariser les données pour l'algorithme Apriori
df_binarized = pd.get_dummies(df, columns=['Num1', 'Num2', 'Num3', 'Num4', 'Num5'])

# Utiliser l'algorithme Apriori pour trouver des motifs fréquents
frequent_itemsets = apriori(df_binarized, min_support=0.1, use_colnames=True)

# Afficher les règles d'association
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)
