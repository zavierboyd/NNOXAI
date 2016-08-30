from __future__ import division
from numpy.random import random, randint
from random import choice
import numpy as np
from numpy import dot, array
np.random.seed(1337)

def playgame(xai, oai, board, see=False):
    turn = 'x'
    while '.' in board:
        if turn == 'x':
            nextboard, move, turn = domove(board, turn, xai)
            if board[move] != '.':
                print board
                print move
                return -2
            board = nextboard
        else:
            nextboard, move, turn = domove(board, turn, oai)
            if board[move] != '.':
                print board
                print move
                return 2
            board = nextboard
        if see:
            printboard(board)
        if didwin(board, 'x'):
            if see:
                print 'x won'
            return 1
        elif didwin(board, 'o'):
            if see:
                print 'o won'
            return -1
    if see:
        print 'draw'
    return 0


def printboard(board):
    print board[0] + board[1] + board[2] + '\n' + board[3] + board[4] + board[5] + '\n' + board[6] + board[7] + board[8]
    print '-----'


class oneturnai2(object):
    def __init__(self):
        self.windata = {}
        self.blockdata = {}

    def __call__(self, board, side):
        if side == 'x':
            otherside = 'o'
        else:
            otherside = 'x'
        prediction = self.checkmove(board, otherside)
        move = self.checkmove(board, side)
        if move != -1:
            self.windata[''.join(board)] = move
            return move  # phone
        elif prediction != -1:
            self.blockdata[''.join(board)] = prediction
            return prediction  # phone
        else:
            return stupidai2(board, side)  # phone

    def checkmove(self, board, side):
        positions = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for i in range(len(positions)):
            p1, p2, p3 = positions[i]
            if board[p1] == side and board[p2] == side and board[p3] == '.':
                return p3  # phone
            if board[p1] == side and board[p3] == side and board[p2] == '.':
                return p2  # phone
            if board[p2] == side and board[p3] == side and board[p1] == '.':
                return p1  # phone
        return -1  # error


def stupidai2(board, side):
    solution = False
    while not solution:
        play = np.random.randint(0, 9)  # np.random.randint takes [low, high) ommiting the high number from poping up
        if board[play] == ".":
            solution = True
    return play  # phone

def domove(board, turn, ai):
    board = list(board)
    move = ai(board, turn)
    board[move] = turn
    board = "".join(board)
    nextturn = 'o' if turn == 'x' else 'x'
    return board, move, nextturn

def didwin(board, side):
    winning = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for a,b,c in winning:
        if board[a] == side and board[b] == side and board[c] == side:
            return True
    return False


def newboard(): return ['.' for i in range(9)]


class XOQLearning(object):
    def __init__(self, multi=5):
        self.database = datadict()
        self.multi = multi

    def __call__(self, board, side):
        state = ''.join(board)
        move = self.findmove(state)
        if board[move] != ".":
            print state, move, self.database[state]
        if side == self.side:
            self.gamekeys[state] = move
        return move

    def learn(self, ai, side, multi=0.2):
        self.gamekeys = {}
        self.side = side
        playmulti = self.multi
        self.multi = multi
        if side == 'x':
            score = playgame(self, ai, newboard())
        else:
            score = playgame(ai, self, newboard())
            score *= -1
        if score == 2:
            score = 0
        for state, move in self.gamekeys.items():
            self.database[state][move][0] += score
            self.database[state][move][1] += 1
        self.multi = playmulti

    def findmove(self, state):
        probs = self.database[state]
        winchance = np.array([prob[0] / prob[1] for prob in probs])
        y = np.exp(winchance * self.multi)
        s = np.sum(y)
        prob = y / s
        c = np.cumsum(prob)
        move = np.sum([c < np.random.random()])
        return move

    @staticmethod
    def newlist():
        return np.array([0 for i in range(9)])


class datadict(dict):
    def __missing__(self, key):
        new = np.array([[0, 1] if bit == '.' else [-1e100, 1] for bit in key])
        self[key] = new
        return new

