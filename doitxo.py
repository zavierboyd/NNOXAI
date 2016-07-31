from stratagies import *
from playxo import play_game
from xo2 import board

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
        return 0


swx = 0
for i in range(20):
    swx += doit ("x",oneturnai(board,'x'),stupidai(board,'o'),board)

print swx