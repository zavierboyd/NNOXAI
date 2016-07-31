from random import randint


phonetoboard = [0,1,2,3,5,6,7,9,10,11]
boardtophone = [0,1,2,3,0,4,5,6,0,7,8,9]


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