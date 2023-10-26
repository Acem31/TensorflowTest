if __name__ == "__main__":
    file_path = 'euromillions.csv'
    euromillions_data = read_euromillions_data(file_path)
    X = [row[:-1] for row in euromillions_data]
    
    # Créez une liste de toutes les cibles pour chaque numéro (de 1 à 50)
    y_tests = [create_target_columns(euromillions_data, num) for num in range(1, 51)]

    best_score = None
    last_tuple = euromillions_data[-1][:-1]

    while True:
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={
                "n_estimators": (10, 500),
                "max_depth": (5, 50),
                "min_samples_split": (0.1, 1.0),
                "min_samples_leaf": (0.1, 0.5),
                "max_features": (1, len(euromillions_data[0])-1),
                "bootstrap": (0, 1),
            },
            random_state=42,
        )

        optimizer.maximize(init_points=5, n_iter=10)

        best_params = optimizer.max['params']
        current_best_score = -optimizer.max['target']

        print(f"Meilleurs hyperparamètres : {best_params}")
        print(f"Meilleur score de précision : {current_best_score * 100}%")

        if best_score is not None and current_best_score <= best_score:
            print("Arrêt de l'optimisation : le score n'augmente plus.")
            break

        best_score = current_best_score

        last_actual_value = euromillions_data[-1][-1]
        print(f"Dernière ligne du CSV : {euromillions_data[-1]}")

        # Prédiction des 5 numéros avec les hyperparamètres optimaux
        predicted_numbers = [0] * 50
        for num in range(1, 51):
            y = y_tests[num - 1]
            model = RandomForestClassifier(
                n_estimators=int(best_params["n_estimators"]),
                max_depth=int(best_params["max_depth"]),
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                max_features=int(best_params["max_features"]),
                bootstrap=bool(best_params["bootstrap"]),
                n_jobs=5
            )
            model.fit(X, y)  # Utilisez l'ensemble complet pour former le modèle
            y_pred_prob = model.predict_proba([last_tuple])
            predicted_numbers[num - 1] = y_pred_prob[0][1]

        top_5_numbers = np.argsort(predicted_numbers)[-5:]
        print(f"Prédiction pour la dernière ligne : {top_5_numbers}")
        print(f"Score de précision actuel : {best_score * 100}%")

    print("Meilleurs hyperparamètres finaux :")
    print(best_params)
