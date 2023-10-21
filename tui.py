import urwid
import subprocess

# Fonction pour lancer le programme
def run_program(button):
    # Lancer le programme externe "reinf_tuples.py"
    try:
        output = subprocess.check_output(['python', 'reinf_tuples.py'])
        text_widget.set_text(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        text_widget.set_text(f"Erreur : {e.returncode}\n{e.output.decode('utf-8')}")

# Fonction pour créer une fenêtre TUI
def create_tui_window():
    # Partie gauche : Affichage du programme
    global text_widget
    text_widget = urwid.Text("Cliquez sur le bouton 'Lancer le programme' pour exécuter le programme.")
    text_widget = urwid.Filler(text_widget)

    # Partie droite : Bouton pour lancer le programme
    button = urwid.Button(u"Lancer le programme")
    urwid.connect_signal(button, 'click', run_program)
    button = urwid.AttrMap(button, None, focus_map='reversed')

    # Conteneur global
    columns = urwid.Columns([text_widget, ('fixed', 30, button)], dividechars=1)

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
