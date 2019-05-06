from flask import Flask
import numpy as np
import random
import scan
import os
# import TrainingReversi as tr

import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)
player = -1
p1 = list()
p2 = list()
Record = True
chessboard = np.empty((8, 8))


def create_model():
    # print("正在建立類神經網路模型...")
    activation = keras.activations.softsign
    Mmodel = keras.Sequential()
    Mmodel.add(keras.layers.Dense(
        64, activation=activation, input_shape=(64,)))
    for _ in range(100):
        Mmodel.add(keras.layers.Dense(100, activation=activation))
    Mmodel.add(keras.layers.Dense(
        64, activation=activation))
    # print("正在編譯類神經網路...")
    Mmodel.compile(
        optimizer=keras.optimizers.Nadam(),
        loss=keras.losses.MAE)
    return Mmodel


if os.path.isfile('./Main_Reversi.h5'):
    print('初始化: ')
    keras.backend.clear_session()
    Mmodel = keras.models.load_model('Main_Reversi.h5')
    print("讀取已保存的主要網路模型...")
else:
    Mmodel = create_model()
    print('讀取錯誤!!重新建立主要網路模型...')


def predict(chessboard):
    global Mmodel  # 不加時使用flask會遺失導致錯誤
    chessboard = chessboard.reshape(1, 64)
    tf.Graph.as_default(graph)
    try:
        output = Mmodel.predict(chessboard)
    except Exception as e:
        # print('錯誤!!!')
        print(e)
        keras.backend.clear_session()
        # print('重新讀取....')
        Mmodel = keras.models.load_model('Main_Reversi.h5')
        # print("讀取已保存的主要網路模型...")
        output = Mmodel.predict(chessboard)
    return output.reshape(8, 8)


def update_weights(input_data, expect_data, batch):
    global Mmodel
    import TrainingReversi as tr
    Mmodel.set_weights(tr.weights())
    print("儲存...")
    Mmodel.save('Main_Reversi.h5')
    print("儲存完成!!")


def reload():
    global Mmodel
    try:
        if os.path.isfile('./Main_Reversi.h5'):
            print('重新讀取權重...')
            model = keras.models.load_model('Reversi.h5')
        else:
            print('尚未有權重，執行儲存...')
            model.save('Main_Reversi.h5')
    except:
        print('釋放GPU資源...')
        keras.backend.clear_session()
        model = keras.models.load_model('Reversi.h5')


def start():
    global player, p1, p2, Record, chessboard
    Record = True
    player = -1
    p1 = list()
    p2 = list()
    chessboard = np.zeros((8, 8))
    chessboard[3][4] = -1
    chessboard[3][3] = 1
    chessboard[4][3] = -1
    chessboard[4][4] = 1
    # dissplay(chessboard)
    print('>'*90)
    return chessboard, player


def check(chessboard, player):
    canDown = np.zeros((8, 8), dtype=bool)
    down = False
    for i in range(8):
        for j in range(8):
            if chessboard[i, j] == player:
                check = scan.check(chessboard, i, j)
                if check.size > 0:
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


ptv = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}


@app.route('/<string:data>')
def down(data):
    playY = ptv[data[0]]
    playX = int(data[1])
    data = np.fromstring(data[3:], count=64, sep=',')
    chessboard = data.reshape(8, 8)
    dissplay(chessboard)
    player = -1
    if np.any(chessboard == 0):
        candown = check(chessboard, player)
        if np.any(candown):
            predict_opt = predict(chessboard*-1)
            predict_opt[candown == False] = np.nan
            if candown[playX, playY]:
                chessboard[playX, playY] = -1
                predict_opt[playX, playY] = 1
                chessboard = scan.update(chessboard, playX, playY, player)
            else:
                print('不能下')
                response = ''.join(
                    str(int(x))+',' for x in np.nditer(chessboard))
                return response[:-1]
            if np.any(check(chessboard, player * -1)):
                player = player * -1
        while player == 1 and np.any(chessboard == 0):
            candown = check(chessboard, player)
            if np.any(candown) == False:
                break
            predict_opt = predict(chessboard)
            predict_opt[candown == False] = np.nan
            max_filter = predict_opt == np.nanmax(predict_opt)
            chessboard[max_filter] = player
            px, py = np.nonzero(max_filter)
            px = px.astype(int)[0]
            py = py.astype(int)[0]
            chessboard = scan.update(chessboard, px, py, player)
            if np.any(check(chessboard, player * -1)):
                break
    response = ''.join(str(int(x))+',' for x in np.nditer(chessboard))
    # dissplay(chessboard)
    return response[:-1]


if __name__ == '__main__':
    Mmodel = create_model()
    graph = tf.get_default_graph()
    down("C3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,")
    app.run(host='0.0.0.0', threaded=False)

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
