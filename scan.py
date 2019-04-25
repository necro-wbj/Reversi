import numpy as np


def update(chessboard, x, y, player):
    for px, py in [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]:
        for i in range(1, 8):
            if x+px*i < 0 or y+py*i < 0 or x+px*i >= 8 or y+py*i >= 8:
                break
            elif chessboard[x+px*i][y+py*i] == player:
                for j in range(1, i):
                    chessboard[x+px*j][y+py*j] = player
                break
            elif chessboard[x+px*i][y+py*i] == 0:
                break
    return chessboard


def check(chessboard, x, y):
    check = False
    for px, py in [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]:
        try:
            i = 1
            while chessboard[x+px*i, y+py*i] == chessboard[x, y] * -1:
                i += 1
            if i > 1:
                nx = x+px*i
                ny = y+py*i
                if chessboard[x+px*i, y+py*i] == 0:
                    if check:
                        result = np.vstack((result, np.array((nx, ny), ndmin=2)))
                    else:
                        result = np.array((nx, ny), ndmin=2)
                        check = True
        except IndexError:
            pass
    if check:
        return result
    return
