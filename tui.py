import urwid
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

# Fonction pour lancer le programme
def run_program(button):
    # Lancer le programme externe "reinf_tuples.py"
    try:
        output = subprocess.check_output(['python', 'reinf_tuples.py'])
        text.set_text(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        text.set_text(f"Erreur : {e.returncode}\n{e.output.decode('utf-8')}")

# Fonction pour créer une fenêtre TUI
def create_tui_window():
    # Partie gauche : Affichage du programme
    global text
    text = urwid.Text("Cliquez sur le bouton 'Lancer le programme' pour exécuter le programme.")
    text = urwid.Pile([urwid.Text("Cliquez sur le bouton 'Lancer le programme' pour exécuter le programme."), text])
    text = urwid.Filler(text)

    # Partie droite : Bouton pour lancer le programme
    button = urwid.Button(u"Lancer le programme")
    urwid.connect_signal(button, 'click', run_program)
    button = urwid.AttrMap(button, None, focus_map='reversed')

    # Conteneur global
    columns = urwid.Columns([('weight', 2, text), ('fixed', 24, button)], dividechars=1)

    # Créer la boucle principale urwid
    main_loop = urwid.MainLoop(columns, unhandled_input=exit_on_q)
    main_loop.run()

# Fonction pour quitter le TUI en appuyant sur 'q'
def exit_on_q(key):
    if key in ('q', 'Q'):
        raise urwid.ExitMainLoop()

# Exécution de la fenêtre TUI
if __name__ == "__main__":
    create_tui_window()
