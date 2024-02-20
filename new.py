import pandas as pd
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Charger les données
data = pd.read_csv('euromillions.csv', sep=';', header=None)

class EuromillionsEnv:
    def __init__(self, data):
        self.data = data
        self.reset()

    def reset(self):
        self.current_index = 0
        self.total_correct_numbers = 0
        self.current_sequence = []

    def step(self, action):
        # Action est la liste des numéros devinés par l'agent
        predicted_numbers = np.array(action)

        # Vérifier les numéros corrects
        correct_numbers = set(self.data.iloc[self.current_index, :6])
        correct_bonus = set(self.data.iloc[self.current_index, 6:8])

        correct_numbers_guessed = sum(num in correct_numbers for num in predicted_numbers[:5])
        correct_bonus_guessed = sum(num in correct_bonus for num in predicted_numbers[5:])

        total_correct = correct_numbers_guessed + correct_bonus_guessed
        self.total_correct_numbers += total_correct

        # Récompense basée sur le nombre de numéros corrects
        reward = total_correct / 7  # Récompense normalisée entre 0 et 1

        # Passage à l'étape suivante
        self.current_index += 1
        done = self.current_index >= len(self.data)

        # Retourner l'état, la récompense et si l'épisode est terminé
        return self.current_index, reward, done

    def get_state(self):
        # État actuel : les numéros tirés jusqu'à présent
        return self.data.iloc[self.current_index, :]

    def get_total_reward(self):
        return self.total_correct_numbers

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # facteur d'actualisation
        self.epsilon = 1.0  # taux d'exploration initial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Réseau neuronal pour l'approximation de la fonction Q
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.sample(range(1, 51), 7)  # Choisir aléatoirement 7 numéros entre 1 et 50 pour les principaux numéros et 2 numéros entre 1 et 12 pour les numéros bonus
        act_values = self.model.predict(state)
        return act_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialiser l'environnement et l'agent
env = EuromillionsEnv(data)
state_size = len(env.get_state())
action_size = 59  # 50 pour les numéros principaux + 12 pour les numéros bonus
agent = DQNAgent(state_size, action_size)

# Entraînement de l'agent
num_episodes = 1000
batch_size = 32
for episode in range(num_episodes):
    env.reset()
    state = np.reshape(env.get_state().values, [1, state_size])
    for timestep in range(len(data)):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state.values, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Prédiction avec l'agent entraîné
env.reset()
state = np.reshape(env.get_state().values, [1, state_size])
predicted_numbers = []

for timestep in range(len(data)):
    action = agent.act(state)
    predicted_numbers.append(action)
    next_state, reward, done = env.step(action)
    if done:
        break
    state = np.reshape(next_state.values, [1, state_size])

print("Numéros prédits (arrondis) pour le prochain tirage:")
print(predicted_numbers)
