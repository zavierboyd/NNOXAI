#!/usr/bin/env python
#
# Copyright 2007 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import webapp2
import numpy as np
from awesomeai import frontmatrix, frontoffset, backmatrix, backoffset, percepmatrix, percepoffset
from neuralnet2 import *
from random import randint
from qlearning import database
from google.appengine.ext import ndb
import cPickle as cp
from urllib import quote, unquote


class QLearnState(ndb.Model):
    moveprobs = ndb.StringProperty()


class QLearnAI(ndb.Model):
    pass

def transboard(board, side):
    oside = 'o' if side == 'x' else 'x'
    side = [1 if bit == side else 0 for bit in board]
    oside = [1 if bit == oside else 0 for bit in board]
    space = [1 if bit == '.' else 0 for bit in board]
    return side + oside + space

def check(a,b,c,side,board):
    return board[a] == side and board[b] == side and board[c] == side

def anyonewon(board,side):
    positions = [(1,2,3),(4,5,6),(7,8,9),(1,4,7),(2,5,8),(3,6,9),(1,5,9),(3,5,7)]
    return any([check(p1,p2,p3,side,board) for p1,p2,p3 in positions])

def gengame(handler, board, urls, won):
    html = '<table class="board">'
    for row in board:
        html += '\n<tr>\n'
        for cell in row:
            if cell == '.':
                if won:
                    html += '''
                    <td>
                        .
                    </td>
                    '''
                else:
                    html += '''
                    <td>
                        <a class="choice" href={}>
                            .
                        </a>
                    </td>
                    '''.format(urls[0])
                    urls.pop(0)
            else:
                html += '\n<td>{}</td>\n'.format(cell)
        html += '\n</tr>\n'
    html += '</table>'
    return html

def stupidai2(board,side):
    solution = False
    while not solution:
        play = np.random.randint(0,9)
        if board[play] == ".":
            solution = True
    return play #phone

class oneturnai2(object):
    def __init__(self):
        pass

    def __call__(self, board, side):
        if side == 'x':
            otherside = 'o'
        else:
            otherside = 'x'
        prediction = self.checkmove(board, otherside)
        move = self.checkmove(board, side)
        if move != -1:
            return move  # phone
        elif prediction != -1:
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


class XOQLearning(object):
    def __init__(self, qtype ,multi=5):
        self.database = Qlearn(qtype)
        self.gamekeys = {}
        self.multi = multi

    def __call__(self, board, side):
        state = ''.join(board)
        self.side = side
        move = self.findmove(state)
        if board[move] != ".":
            print state, move, self.database[state]
        self.gamekeys[state] = move
        return move

    def learn(self, moves, score):
        self.gamekeys = moves
        for state, move in self.gamekeys.items():
            probs = self.database[state]
            probs[move][0] += score
            probs[move][1] += 1
            self.database[state] = probs

    def findmove(self, state):
        probs = self.database[state]
        winchance = np.array([prob[0] / prob[1] for prob in probs])
        y = np.exp(winchance * self.multi)
        s = np.sum(y)
        prob = y / s
        c = np.cumsum(prob)
        move = np.sum([c < np.random.random()])
        return move

class Qlearn(object):
    def __init__(self, key):
        self.aikey = ndb.Key(QLearnAI, key)
        if self.aikey.get() is None:
            top = QLearnAI(id=key)
            self.aikey = top.put()

    def __getitem__(self, state):
        statekey = ndb.Key(QLearnState, state, parent=self.aikey)
        data = statekey.get()
        if data is None:
            new = np.array([[0, 1] if bit == '.' else [-1e100, 1] for bit in state])
            self.__setitem__(state, new)
            probabilitys = new
        else:
            probabilitys = cp.loads(str(data.moveprobs))
        return probabilitys

    def __setitem__(self, state, probs):
        moveprobs = cp.dumps(probs)
        newdata = QLearnState(id=state, moveprobs=moveprobs, parent=self.aikey)
        newdata.put()

def domove(board, turn, ai):
    board = list(board)
    move = ai(board, turn)
    board[move] = turn
    board = "".join(board)
    nextturn = 'o' if turn == 'x' else 'x'
    return board, nextturn

def didwin(board, side):
    winning = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    board = list(board)
    print board
    for a,b,c in winning:
        if board[a] == side and board[b] == side and board[c] == side:
            return True
    return False

def createurls(prefix, board, nextturn):
    board = list(board)
    urls = []
    for i in range(len(board)):
        if board[i] == '.':
            url = list(board)
            url[i] = nextturn
            turn = 'x' if nextturn == 'o' else 'o'
            urls.append('{prefix}/{board}/{turn}'.format(prefix=prefix, board=''.join(url), turn=turn))
    return urls


class MainHandler(webapp2.RequestHandler):
    def get(self, ai):
        board = '.........'
        turn = 'o'
        playerturn = 'x'
        urls = createurls('/'+ai,board,playerturn)
        html = [list(board)]
        htmlboards = [[[a, b, c], [d, e, f], [g, h, i]] for a, b, c, d, e, f, g, h, i in html]
        snippet = gengame(self, htmlboards[0], urls, False)
        self.response.write(snippet)

class StupidHandler(webapp2.RequestHandler):
    def get(self, board, turn):
        playerside = 'x' if turn == 'o' else 'o'
        win = didwin(board, playerside)
        game = list(board)
        draw = True if ((not ('.' in game)) and (not win)) else False
        if (not win) and (not draw):
            board, nextturn = domove(board, turn, stupidai2)
            urls = createurls('/stupid', board, nextturn)
        else:
            urls = []
        lose = didwin(board, turn)
        if lose:
            urls = []
        board = [list(board)]
        htmlboards = [[[a, b, c], [d, e, f], [g, h, i]] for a, b, c, d, e, f, g, h, i in board]
        snippet = gengame(self, htmlboards[0], urls, win or lose)
        if win:
            snippet += '<div>{}</div>'.format(playerside.upper()+' Wins!')
        elif lose:
            snippet += '<div>{}</div>'.format(turn.upper() + ' Wins!')
        elif draw:
            snippet += '<div>Draw!</div>'
        self.response.write(snippet)


class OneTurnHandler(webapp2.RequestHandler):
    def get(self, board, turn):
        playerside = 'x' if turn == 'o' else 'o'
        win = didwin(board, playerside)
        game = list(board)
        draw = True if ((not ('.' in game)) and (not win)) else False
        if (not win) and (not draw):
            board, nextturn = domove(board, turn, oneturnai2())
            urls = createurls('/oneturn', board, nextturn)
        else:
            urls = []
        lose = didwin(board, turn)
        if lose:
            urls = []
        board = [list(board)]
        htmlboards = [[[a, b, c], [d, e, f], [g, h, i]] for a, b, c, d, e, f, g, h, i in board]
        snippet = gengame(self, htmlboards[0], urls, win or lose)
        if win:
            snippet += '<div>{}</div>'.format(playerside.upper() + ' Wins!')
        elif lose:
            snippet += '<div>{}</div>'.format(turn.upper() + ' Wins!')
        elif draw:
            snippet += '<div>Draw!</div>'
        self.response.write(snippet)


class HiddenHandler(webapp2.RequestHandler):
    neuralnet = None

    def get(self, board, turn):
        if HiddenHandler.neuralnet is None:
            HiddenHandler.neuralnet = XOHidden([Perceptron(backmatrix, backoffset, activation=rectified_linear), Perceptron(frontmatrix, frontoffset, activation=linear)])
        playerside = 'x' if turn == 'o' else 'o'
        win = didwin(board, playerside)
        game = list(board)
        draw = True if ((not ('.' in game)) and (not win)) else False
        if (not win) and (not draw):
            board, nextturn = domove(board, turn, HiddenHandler.neuralnet)
            urls = createurls('/hidden', board, nextturn)
        else:
            urls = []
        lose = didwin(board, turn)
        if lose:
            urls = []
        board = [list(board)]
        htmlboards = [[[a, b, c], [d, e, f], [g, h, i]] for a, b, c, d, e, f, g, h, i in board]
        snippet = gengame(self, htmlboards[0], urls, win or lose)
        if win:
            snippet += '<div>{}</div>'.format(playerside.upper() + ' Wins!')
        elif lose:
            snippet += '<div>{}</div>'.format(turn.upper() + ' Wins!')
        elif draw:
            snippet += '<div>Draw!</div>'
        self.response.write(snippet)


class PerceptronHandler(webapp2.RequestHandler):
    neuralnet = None

    def get(self, board, turn):
        if PerceptronHandler.neuralnet is None:
            PerceptronHandler.neuralnet = XOPerceptron(percepmatrix, percepoffset, activation=rectified_linear)
        playerside = 'x' if turn == 'o' else 'o'
        win = didwin(board, playerside)
        game = list(board)
        draw = True if ((not ('.' in game)) and (not win)) else False
        if (not win) and (not draw):
            board, nextturn = domove(board, turn, PerceptronHandler.neuralnet)
            urls = createurls('/percep', board, nextturn)
        else:
            urls = []
        lose = didwin(board, turn)
        if lose:
            urls = []
        board = [list(board)]
        htmlboards = [[[a, b, c], [d, e, f], [g, h, i]] for a, b, c, d, e, f, g, h, i in board]
        snippet = gengame(self, htmlboards[0], urls, win or lose)
        if win:
            snippet += '<div>{}</div>'.format(playerside.upper() + ' Wins!')
        elif lose:
            snippet += '<div>{}</div>'.format(turn.upper() + ' Wins!')
        elif draw:
            snippet += '<div>Draw!</div>'
        self.response.write(snippet)


class PullTestHandler(webapp2.RequestHandler):
    def get(self):
        d = Qlearn('online')
        probs = d['x........']
        self.response.write(probs)


class PutTestHandler(webapp2.RequestHandler):
    def get(self):
        d = Qlearn('online')
        nprobs = d['....x....']
        nprobs[1][0] += 1
        nprobs[1][1] += 1
        d['....x....'] = nprobs
        probs = d['....x....']
        self.response.write(probs)


class OfflineQHandler(webapp2.RequestHandler):
    def get(self, board, turn):
        qtype = 'offline'
        aikey = ndb.Key(QLearnAI, qtype)
        if aikey.get() is None:
            data = Qlearn(qtype)
            for state, probs in database.items():
                data[state] = probs

        playerside = 'x' if turn == 'o' else 'o'
        win = didwin(board, playerside)
        game = list(board)
        draw = True if ((not ('.' in game)) and (not win)) else False
        if (not win) and (not draw):
            board, nextturn = domove(board, turn, XOQLearning(qtype))
            urls = createurls('/qoffline', board, nextturn)
        else:
            urls = []
        lose = didwin(board, turn)
        if lose:
            urls = []
        board = [list(board)]
        htmlboards = [[[a, b, c], [d, e, f], [g, h, i]] for a, b, c, d, e, f, g, h, i in board]
        snippet = gengame(self, htmlboards[0], urls, win or lose)
        if win:
            snippet += '<div>{}</div>'.format(playerside.upper() + ' Wins!')
        elif lose:
            snippet += '<div>{}</div>'.format(turn.upper() + ' Wins!')
        elif draw:
            snippet += '<div>Draw!</div>'
        self.response.write(snippet)


class OnlineQHandler(webapp2.RequestHandler):
    def get(self, board, turn):
        qtype = 'online'
        try:
            pickledmemory = unquote(self.request.get('memory'))
            print 'memory', pickledmemory
            memory = cp.loads(str(pickledmemory))

        except:
            memory = {}
        print 'memory', memory
        playerside = 'x' if turn == 'o' else 'o'
        win = didwin(board, playerside)
        game = list(board)
        draw = True if ((not ('.' in game)) and (not win)) else False
        if (not win) and (not draw):
            game = list(board)
            move = XOQLearning(qtype)(board, turn)
            memory[board] = move
            game[move] = turn
            game = "".join(game)
            nextturn = 'o' if turn == 'x' else 'x'
            urls = createurls('/qonline', game, nextturn)
            pickledmemory = cp.dumps(memory)
            print 'memory', pickledmemory
            urls = [url+'?memory='+quote(pickledmemory) for url in urls]
        else:
            urls = []
        lose = didwin(game, turn)
        if lose:
            urls = []
        game = [list(game)]
        htmlboards = [[[a, b, c], [d, e, f], [g, h, i]] for a, b, c, d, e, f, g, h, i in game]
        snippet = gengame(self, htmlboards[0], urls, win or lose)
        if win:
            score = -1
            XOQLearning(qtype).learn(memory, score)
            snippet += '<div>{}</div>'.format(playerside.upper() + ' Wins!')
        elif lose:
            score = 1
            XOQLearning(qtype).learn(memory, score)
            snippet += '<div>{}</div>'.format(turn.upper() + ' Wins!')
        elif draw:
            snippet += '<div>Draw!</div>'
        self.response.write(snippet)


app = webapp2.WSGIApplication([
    ('/start/(.*)', MainHandler),
    ('/stupid/([.ox]*)/([ox])', StupidHandler),
    ('/oneturn/([.ox]*)/([ox])', OneTurnHandler),
    ('/hidden/([.ox]*)/([ox])', HiddenHandler),
    ('/percep/([.ox]*)/([ox])', PerceptronHandler),
    ('/qonline/([.ox]*)/([ox])', OnlineQHandler),
    ('/pulltest', PullTestHandler)
], debug=True)
