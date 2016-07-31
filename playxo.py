def check(a,b,c,side,board):
    return board[a] == side and board[b] == side and board[c] == side

def anyonewon(board,side):
    positions = [(1,5,9),(2,6,10),(3,7,11),(1,2,3),(5,6,7),(9,10,11),(1,6,11),(3,6,9)]
    return any([check(p1,p2,p3,side,board) for p1,p2,p3 in positions])

phonetoboard = [0,1,2,3,5,6,7,9,10,11]

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
            return "x wins!"
        if owin:
            return "o wins!"
        print turn
    return "draw"
