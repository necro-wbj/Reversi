import numpy as np


def update(chessboard, x, y, player):
    for px, py in [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]:
        for i in range(1, 8):
            if x+px*i < 0 or y+py*i < 0 or x+px*i > 7 or y+py*i > 7:
                break
            if chessboard[x+px*i,y+py*i] == 0:
                break
            if chessboard[x+px*i,y+py*i] == player:
                for j in range(1, i):
                    chessboard[x+px*j,y+py*j] = player
                break
    return chessboard


def check(chessboard, x, y):
    check = False
    result = np.empty((0,0))
    for px, py in [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]:
        try:
            i = 1
            while chessboard[x+px*i, y+py*i] == chessboard[x, y] * -1:
                i += 1
            if i > 1:
                nx = x+px*i
                ny = y+py*i
                #小心 numpy索引方式與list一樣可以有負值(不會觸發IndexError)
                if chessboard[nx, ny] == 0 and 0 <= nx <= 7 and 0 <= ny <= 7:
                    if check:
                        result = np.vstack((result, np.array((nx, ny), ndmin=2)))
                    else:
                        result = np.array((nx, ny), ndmin=2)
                        check = True
        except IndexError:
            pass
    return result
