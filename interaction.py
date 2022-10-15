import random
from definitions import PlayerColour
from rendering import board as board_svg
from game import State, Piece
from IPython.display import clear_output, display
import ipywidgets as widgets
def interactive_random_game() -> None:
    history = [State.get_starting_state()]
    i = 0

    def show():
        nonlocal i
        clear_output()
        display(board_svg(history[i], size=500))
        display(prev_button)
        display(next_button)
        print(i)

    turn: PlayerColour = "white"
    def next_clicked(arg):
        nonlocal i
        nonlocal turn
        if i == len(history) - 1:
            candidates = history[-1].get_valid_moves(player=turn)
            turn = "black" if turn == "white" else "white"
            history.append(random.choice(candidates))
        i+=1
        show()
        
    def prev_clicked(arg):
        nonlocal i
        i -= 1
        if i < 0:
            i = 0
        show()


    prev_button = widgets.Button(description="prev")
    prev_button.on_click(prev_clicked)
    next_button = widgets.Button(description="next")
    next_button.on_click(next_clicked)

    show()


