import urwid
import subprocess

# Fonction pour créer une fenêtre TUI
def create_tui_window(stdscr):
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
    
    main_loop.run()

# Fonction pour quitter le TUI en appuyant sur 'q' ou 'Q'
def exit_on_q(key):
    if key in ('q', 'Q'):
        raise urwid.ExitMainLoop()

if __name__ == "__main__":
    curses.wrapper(create_tui_window)
    # Exécuter le script Python "reinf_tuples.py" lorsque F est pressé
    if key == 'F':
        output = subprocess.check_output(["python3.10", "reinf_tuples.py"])
