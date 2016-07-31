from random import randint
board = """
...
...
...
"""
#123
#456
#789
#123
#567
#91011
#translater
phonetoboard = [0,1,2,3,5,6,7,9,10,11]
boardtophone = [0,1,2,3,0,4,5,6,0,7,8,9]
# def boardtophone(number):
#     return number + (number-1)//3
#
# #translater
# def phonetoboard(number):
#     return number - (number-1)//3

print [boardtophone[phonetoboard[i]]for i in range(1,10)]
#phone
def stupidai(board,side):
    solution = False
    while not solution:
        play = randint(1,9)
        if board[phonetoboard[play]] == ".":
            solution = True
    return play #phone

def checkmove(board,side):
    positions = [(1,5,9),(2,6,10),(3,7,11),(1,2,3),(5,6,7),(9,10,11),(1,6,11),(3,6,9)]
    for i in range(len(positions)):
        p1,p2,p3 = positions[i]
        if board[p1] == side and board[p2] == side and board[p3] == '.':
            return boardtophone[p3] #phone
        if board[p1] == side and board[p3] == side and board[p2] == '.':
            return boardtophone[p2] #phone
        if board[p2] == side and board[p3] == side and board[p1] == '.':
            return boardtophone[p1] #phone
    return 0 #error

def oneturnai(board,side):
    if side == 'x':
        otherside = 'o'
    else:
        otherside = 'x'
    prediction = checkmove(board,otherside)
    move = checkmove(board,side)
    if move != 0:
        return move #phone
    elif prediction != 0:
        return prediction #phone
    else:
        return stupidai(board,side) #phone

#boolean
def check(a,b,c,side,board):
    return board[a] == side and board[b] == side and board[c] == side

#boolean
def anyonewon(board,side):
    positions = [(1,5,9),(2,6,10),(3,7,11),(1,2,3),(5,6,7),(9,10,11),(1,6,11),(3,6,9)]
    return any([check(p1,p2,p3,side,board) for p1,p2,p3 in positions])

#winner
def play_game(xai,oai,board):
    board = list(board)
    turn = 0
    win = 0
    while '.' in board:
        print turn
        if turn%2 == 0:
            x_move = xai(board,'x')
            if board[phonetoboard[x_move]] == '.':
                board[phonetoboard[x_move]] = 'x'
            else:
                raise Exception("WARNING: ILLEGAL MOVE DUE TO CROSSES")
        if turn%2 == 1:
            o_move = oai(board,'o')
            if board[phonetoboard[o_move]] == '.':
                board[phonetoboard[o_move]] = 'o'
            else:
                raise Exception("WARNING: ILLEGAL MOVE DUE TO NAUGHTS")
        turn += 1
        xwin = anyonewon(board,'x')
        owin = anyonewon(board,'o')
        print "".join(board)
        if xwin:
            print "x wins!"
            break
        if owin:
            print "o wins!"
            break
        print turn

play_game(stupidai,stupidai,board)