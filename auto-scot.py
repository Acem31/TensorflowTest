# ...

# Seuil de distance pour continuer l'apprentissage
seuil_distance = 10.0
distance = 15

while distance > seuil_distance:
    # Prédiction avec le modèle
    next_numbers_prediction = best_model.predict(last_five_numbers.reshape(1, 1, -1))
    rounded_predictions = np.round(next_numbers_prediction)

    # Calcul de la distance euclidienne entre la prédiction et la dernière ligne du CSV
    distance = np.linalg.norm(rounded_predictions - data[-1])

    print("Prédiction pour les 5 prochains numéros :", rounded_predictions)
    print("Dernière ligne du CSV :", data[-1])
    print("Distance euclidienne avec la dernière ligne du CSV :", distance)

    # Ré-entraîner le modèle avec les nouvelles données
    best_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

    # Préparer les nouvelles données pour la prédiction
    last_five_numbers = np.array(data[-1]).reshape(1, 1, -1)
    last_five_numbers = np.squeeze([scaler.transform(last_five_numbers[:, i, :]) for i in range(last_five_numbers.shape[1])])

# Maintenant, après la fin de la boucle basée sur la distance, nous exécutons la boucle basée sur la perte

# Initialiser la perte à une valeur arbitrairement élevée pour entrer dans la boucle
loss = float('inf')

while loss > 40:
    all_data = []
    with open('euromillions.csv', 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            numbers = list(map(int, row[:5]))
            all_data.append(numbers)

    # Ajouter les nouvelles données au jeu de données existant
    X_extended = np.concatenate((X, np.array(all_data[:-1])))
    y_extended = np.concatenate((y, np.array(all_data[1:])))

    # Diviser les données étendues en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_extended, y_extended, test_size=0.2, random_state=42)

    # Normaliser les données
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Réorganiser les données pour qu'elles soient compatibles avec un modèle RNN
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Réentraîner le modèle avec les données étendues
    best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Réinitialiser le processus de prédiction avec la dernière ligne du CSV
    last_five_numbers = np.array(all_data[-1]).reshape(1, 1, -1)
    last_five_numbers = np.squeeze([scaler.transform(last_five_numbers[:, i, :]) for i in range(last_five_numbers.shape[1])])

    # Calculer la perte sur l'ensemble de test
    loss = best_model.evaluate(X_test, y_test)

# Une dernière prédiction après la fin de la boucle
final_prediction = best_model.predict(last_five_numbers.reshape(1, 1, -1))
final_rounded_prediction = np.round(final_prediction)

print("Dernière prédiction pour les 5 prochains numéros :", final_rounded_prediction)
