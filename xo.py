from random import randint
board = """
...
...
...
"""
#123
#456
#789

x_moves = [1,9,7,4,6]

o_moves = [5,3,8,2]

for i in range(1,10):
    print i, (i-1) // 3

def stupidai(board):
    solution = 0
    while solution != 1:
        play = randint(0,9)
        if board[play+(play-1)//3] == '.':
            solution = 1
    return play


def check(a,b,c,side,board):
    return board[a] == side and board[b] == side and board[c] == side


def anyonewon(board,side):
    print "checking", side
    positions = [(1,5,9),(2,6,10),(3,7,11),(1,2,3),(5,6,7),(9,10,11),(1,6,11),(3,6,9)]
    return any([check(p1,p2,p3,side,board) for p1,p2,p3 in positions])

def play_game(x_moves,o_moves,board):
    turn = 0
    win = 0
    moves = (len(x_moves)+len(o_moves))
    while turn < moves :
        if turn%2 == 0:
            print x_moves
            x_move = x_moves[0]
            del x_moves[0]
            print x_moves
            b = list(board)
            if b[x_move+(x_move-1)//3] == '.':
                b[x_move+(x_move-1)//3] = 'x'
            else:
                raise Exception("WARNING: ILLEGAL MOVE DUE TO CROSSES")
            b = "".join(b)
            print b
        if turn%2 == 1:
            print o_moves
            o_move = o_moves[0]
            del o_moves[0]
            print o_moves
            b = list(board)
            if b[o_move+(o_move-1)//3] == '.':
                b[o_move+(o_move-1)//3] = 'o'
            else:
                raise Exception("WARNING: ILLEGAL MOVE DUE TO NAUGHTS")
            b = "".join(b)
            print b
        turn += 1
        board = b
        xwin = anyonewon(board,'x')
        owin = anyonewon(board,'o')
        if xwin:
            print "x wins!"
            break
        if owin:
            print "o wins!"
            break
        print turn

print play_game(x_moves,o_moves,board)