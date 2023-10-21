import urwid
import subprocess

# Fonction pour lancer le programme
def run_program(key):
    if key == 'f' or key == 'F':
        # Lancer le programme externe "reinf_tuples.py"
        try:
            output = subprocess.check_output(['python', 'reinf_tuples.py'])
            text_widget.set_text(output.decode('utf-8'))
        except subprocess.CalledProcessError as e:
            text_widget.set_text(f"Erreur : {e.returncode}\n{e.output.decode('utf-8')}")

# Fonction pour créer une fenêtre TUI
def create_tui_window():
    # Partie gauche : Affichage du programme
    program_text = urwid.Text("Programme en cours d'exécution...")
    program_frame = urwid.Frame(program_text)

    # Partie droite : Affichage des hyperparamètres et du taux de réussite
    hyperparams_text = urwid.Text("Hyperparamètres en cours d'utilisation:")
    accuracy_text = urwid.Text("Taux de réussite en cours: 0.0")

    right_pile = urwid.Pile([hyperparams_text, accuracy_text])
    right_frame = urwid.Frame(right_pile)

    # Conteneur global
    columns = urwid.Columns([program_frame, right_frame], dividechars=1)

    # Créer la boucle principale urwid
    main_loop = urwid.MainLoop(columns, unhandled_input=exit_on_q)

    try:
        main_loop.run()
    except Exception as e:
        print(f"Erreur dans la boucle principale Urwid : {e}")

# Exécution de la fenêtre TUI
if __name__ == "__main__":
    create_tui_window()
