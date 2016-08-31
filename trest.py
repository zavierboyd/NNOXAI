def stupidai(board):
    solution = 0
    while solution != 1:
        play = randint(0,9)
        if board[play+(play-1)//3] == '.':
            solution = 1
    return play