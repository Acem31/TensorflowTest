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
    # Partie gauche : Instructions
    instructions_text = urwid.Text("Appuyez sur la touche 'F' pour lancer le programme.")
    instructions_text = urwid.Padding(instructions_text, align='left')
    instructions_text = urwid.AttrMap(instructions_text, 'body')

    # Partie droite : Affichage du programme
    global text_widget
    text_widget = urwid.Text("Attente de l'exécution du programme...")
    text_widget = urwid.Filler(text_widget)

    # Conteneur global
    columns = urwid.Columns([instructions_text, text_widget], dividechars=1)

    # Créer la boucle principale urwid
    main_loop = urwid.MainLoop(columns, unhandled_input=run_program)
    main_loop.run()

# Exécution de la fenêtre TUI
if __name__ == "__main__":
    create_tui_window()
