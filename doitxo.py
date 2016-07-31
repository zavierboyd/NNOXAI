from stratagies import *
from playxo import play_game
board = """
...
...
...
"""
def doit(side,xai,oai,board):
    game = play_game(xai,oai,board)
    if game == "x wins!":
        if side == "x":
            return 1
        else:
            return -1
    elif game == "o wins!":
        if side == "x":
            return -1
        else:
            return 1
    else:
        return game

