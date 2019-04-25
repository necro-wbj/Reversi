import Main_TrainingReversi as mtr
import numpy as np
import random
import scan

# import TrainingReversi as tr
from flask import Flask


player = -1
p1 = list()
p2 = list()
Record = True


def start():
    global player, p1, p2, Record, chessboard
    Record = True
    chessboard = np.zeros((8, 8))
    player = -1
    p1 = list()
    p2 = list()
    chessboard[3][4] = -1
    chessboard[3][3] = 1
    chessboard[4][3] = -1
    chessboard[4][4] = 1
    dissplay(chessboard)
    print('..................................................')
    return chessboard, player


def check(chessboard, player):
    canDown = np.zeros((8, 8), dtype=bool)
    down = False

    for i in range(8):
        for j in range(8):
            if chessboard[i, j] == player:
                check = scan.check(chessboard, i, j)
                if np.any(check):
                    if down:
                        result = np.vstack((result, check))
                    else:
                        result = check
                        down = True
    if down:
        for i, j in result:
            if chessboard[i, j] != player:
                canDown[i, j] = True
    return canDown



def dissplay(chessboard):
    print(f'  | Ａ | Ｂ | Ｃ | Ｄ | Ｅ | Ｆ | Ｇ | Ｈ | ')
    print(f'{"":-<45}')
    for i in range(8):
        result = np.empty(8, dtype=str)
        for j in range(8):
            if chessboard[i][j] == 0:
                result[j] = '　'
            elif chessboard[i][j] == -1:
                result[j] = '○'
            else:
                result[j] = '●'
        print(
            f' {i}| {result[0]} | {result[1]} | {result[2]} | {result[3]} | {result[4]} | {result[5]} | {result[6]} | {result[7]} | ')
        print('{:-<45}'.format(''))



start()
app = Flask(__name__)


@app.route('/start')
def reset():
    chessboard, player = start()  # , input_temp
    response = ''
    for i in chessboard.reshape(-1):
        response = response + str(int(i)) + ','
    return response[:-1]


@app.route('/')
def dsp():
    st = ''
    for i in chessboard.reshape(-1):
        st = st + str(int(i)) + ','
    return '%s' % st[:len(st)-1]


predict_opt = mtr.predict_opt(chessboard)


@app.route('/<data>')
def down(data):
    data = np.fromstring(data, count=65, sep=',')
    position = int(data[0])
    chessboard = data[1:].reshape(8,8)
    player = -1
    if np.any(chessboard == 0):
        cnaDown = check(player,chessboard)
        if np.any(cnaDown):
            predict_opt = mtr.predict_opt(chessboard)
            predict_opt[cnaDown == False] = np.nan
            dissplay(predict_opt)
            playY = position // 8
            playX = position % 8
            if predict_opt[playY,playX] != np.nan:
                chessboard[playY,playX] = -1
                predict_opt[playY][playX] = 1
                chessboard = scan.update(chessboard, playX, playY, player)
            else:
                print('不能下')
                return ''.join(str(n) for n in chessboard.reshape(64,))
            if np.any(check(player * -1,chessboard)):
                player = player * -1
        while player == 1 and np.any(chessboard == 0):
            cnaDown = check(player,chessboard)
            predict_opt = mtr.predict_opt(chessboard)
            predict_opt[cnaDown == False] = np.nan
            max_filter = predict_opt == np.nanmax(predict_opt)
            predict_opt = np.nan_to_num(predict_opt)
            predict_opt[max_filter] = np.inf
            chessboard[max_filter] = player
            px, py = np.nonzero(max_filter)
            px = px.astype(int)[0]
            py = py.astype(int)[0]
            chessboard = scan.update(chessboard, px, py, player)
            if np.any(check(player * -1,chessboard)):
                player = player * -1
            elif np.any(check(player,chessboard)):
                break
    dissplay(chessboard)
    return ''.join(str(n) for n in chessboard.reshape(64,))




# if batch >= 60 * memory:
#     player_black, player_white = gamming(Probability=0)
#     if player_black < player_white:
#         print(f"主要網路獲勝 => 重新訓練")
#         tr.update_weights(input_data, expect_data, batch)
#     elif player_black > player_white:
#         print("主要網路敗北 => 更新強度為敵對網路")
#         mtr.update_weights(input_data, expect_data, batch)
#     else:
#         print("平手!! => 重新訓練敵對網路")
#         tr.update_weights(input_data, expect_data, batch)
#     batch = 0
#     input_data = list()
#     expect_data = list()
# else:
#     player_black, player_white = gamming(manual=manual, Probability=0)
#     if player_black > player_white:
#         print(
#             f"黑棋獲勝\t黑棋數量: {player_black}\t白旗數量: {player_white}", end='\t')
#         for dataset in p1:
#             input_data.append(dataset[0])
#             dataset[1][dataset[1] == np.amax(dataset[1])] = 1
#             expect_data.append(dataset[1])
#         for dataset in p2:
#             input_data.append(dataset[0])
#             dataset[1][dataset[1] == np.amax(dataset[1])] = 0
#             expect_data.append(dataset[1])
#     elif player_black < player_white:
#         print(
#             f"白棋獲勝\t黑棋數量: {player_black}\t白旗數量: {player_white}", end='\t')
#         for dataset in p1:
#             dataset[1][dataset[1] == np.amax(dataset[1])] = 0
#             input_data.append(dataset[0])
#             expect_data.append(dataset[1])
#         for dataset in p2:
#             input_data.append(dataset[0])
#             dataset[1][dataset[1] == np.amax(dataset[1])] = 1
#             expect_data.append(dataset[1])
#     else:
#         print(
#             f"平手!!\t黑棋數量: {player_black}\t白旗數量: {player_white}", end='\t')
#         for dataset in p1:
#             input_data.append(dataset[0])
#             expect_data.append(dataset[1])
#         for dataset in p2:
#             input_data.append(dataset[0])
#             expect_data.append(dataset[1])
#     batch = len(input_data)
#     print(f'累計棋盤數: {batch}')
#     tr.train(input_data, expect_data, batch)
