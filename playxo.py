from __future__ import division
from random import randint,random,choice
import numpy as np
from numpy import array
from math import sqrt,exp,e
from stratagies import *
board = """
...
...
...
"""

def newboard():
    return """
...
...
...
"""

def check(a,b,c,side,board):
    return board[a] == side and board[b] == side and board[c] == side

def anyonewon(board,side):
    positions = [(1,5,9),(2,6,10),(3,7,11),(1,2,3),(5,6,7),(9,10,11),(1,6,11),(3,6,9)]
    return any([check(p1,p2,p3,side,board) for p1,p2,p3 in positions])

phonetoboard = [0,1,2,3,5,6,7,9,10,11]
def play_game(xai,oai,board,see = False):
    board = list(board)
    listofboards=[]
    turn = 0
    win = 0
    while '.' in board:
        if turn%2 == 0:
            x_move = xai(board,'x')
            if board[phonetoboard[x_move]] == '.':
                board[phonetoboard[x_move]] = 'x'
            else:
                return -2
        if turn%2 == 1:
            o_move = oai(board,'o')
            if board[phonetoboard[o_move]] == '.':
                board[phonetoboard[o_move]] = 'o'
            else:
                return -2
        listofboards.append(board)
        xwin = anyonewon(board,'x')
        owin = anyonewon(board,'o')
        if see:
            print "".join(board),turn
        if xwin:
            return "x wins!",listofboards[turn-1:]
        if owin:
            return "o wins!"
        turn += 1
    return 0