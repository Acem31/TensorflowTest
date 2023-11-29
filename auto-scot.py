import pandas as pd
from itertools import combinations
from mlxtend.frequent_patterns import association_rules

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

# Générer toutes les combinaisons possibles de 5 numéros
all_combinations = list(combinations(df_binarized.columns, 5))

# Filtrer les combinaisons qui ont un support suffisant
frequent_combinations = [combo for combo in all_combinations if df_binarized[list(combo)].all(axis=1).mean() > 0.05]

# Convertir les combinaisons en un DataFrame
df_frequent_combinations = pd.DataFrame(frequent_combinations, columns=['Num1', 'Num2', 'Num3', 'Num4', 'Num5'])

# Utiliser l'algorithme Apriori pour trouver des motifs fréquents
frequent_itemsets = apriori(df_binarized[list(df_frequent_combinations.columns)], min_support=0.05, use_colnames=True)

# Afficher les règles d'association
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)

# Fonction pour enregistrer les règles dans un fichier CSV
def save_rules_to_csv(rules, file_path):
    rules.to_csv(file_path, index=False)

# Enregistrer les règles dans un fichier CSV
save_rules_to_csv(rules, 'association_rules.csv')
