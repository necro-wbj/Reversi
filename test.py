import numpy as np
from time import clock
from numpy import array
import os
import functools
test = np.ones(60)
t2 = True
time = clock()
test.size>0
print(clock() - time)
time = clock()
t2 == True
print(clock() - time)


# canDown = np.zeros((8, 8), dtype=bool)
# down = False

# print(chessboard)
# def check(chessboard, x, y):
#     check = False
#     for px, py in [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]:
#         i = 1
#         while chessboard[x+px*i, y+py*i] == chessboard[x, y] * -1:
#             i += 1
#         if i > 1:
#             nx = x+px*i
#             ny = y+py*i
#             if chessboard[x+px*i, y+py*i] == 0:
#                 if check:
#                     result = np.vstack((result, np.array((nx, ny), ndmin=2)))
#                 else:
#                     result = np.array((nx, ny), ndmin=2)
#                     check = True
#     return result



# for i in range(8):
#     for j in range(8):
#         if chessboard[i, j] == player:
#             if down:
#                 result = np.vstack((result, check(chessboard, i, j)))
#             else:
#                 result = check(chessboard, i, j)
#                 down = True
# if down:
#     print(result.tolist())
#     for i, j in result:
#         if chessboard[i, j] != player:
#             canDown[i, j] = True
# print(canDown)
