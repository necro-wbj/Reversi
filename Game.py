import numpy as np
import random
import scan
import Main_TrainingReversi as mtr
import TrainingReversi as tr
# from flask import Flask


player = -1
Record = True
chessboard = np.zeros((8, 8))
p1 = np.array([])
p2 = np.array([])


def start():
    global player, Record, chessboard
    Record = True
    chessboard = np.zeros((8, 8))
    player = -1
    chessboard[3][4] = -1
    chessboard[3][3] = 1
    chessboard[4][3] = -1
    chessboard[4][4] = 1
    dissplay(chessboard)
    print('..................................................')
    return chessboard, player


def check(player):
    result = list()
    canDown = np.zeros((8, 8), dtype=bool)
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == player:
                result.extend(scan.check(chessboard, i, j))
    if len(result) > 0:
        for x, y in result:
            canDown[x][y] = True
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


manual = False  # 是否手動遊玩
row = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}


def gamming(manual=False, Probability=0):
    global memory, batch, input_data, expect_data, Record, p1, p2
    chessboard, player = start()
    while np.any(chessboard == 0):
        cnaDown = check(player)
        if np.any(cnaDown.reshape(-1)):
            if player == -1:
                predict_opt = tr.predict_opt(chessboard * -1)
            else:
                predict_opt = mtr.predict_opt(chessboard)
            end_game = False
            predict_opt[cnaDown == False] = np.nan
            max_filter = predict_opt == np.nanmax(predict_opt)
            predict_opt = np.nan_to_num(predict_opt)
            predict_opt[max_filter] = np.inf
            if manual and player == -1:
                memory = 1
                while True:
                    try:
                        playX, playY = input("請輸入位置：")
                        playX = ord(playX.upper())-65
                        playY = int(playY)
                        if predict_opt[playY][playX] != 0:
                            chessboard[playY][playX] = -1
                            predict_opt[playY][playX] = 1
                            max_value = 1
                            break
                        else:
                            print('錯誤！違反遊戲規則')
                    except:
                        print('錯誤!請重新輸入')
            else:
                Record = True
                if len(predict_opt[cnaDown]) > 1 and random.randint(1, 100) <= Probability:
                    Record = False
                    max_value = np.random.choice(predict_opt[cnaDown])
                chessboard[np.isinf(predict_opt)] = player
            px, py = np.nonzero(np.isinf(predict_opt.reshape(8, 8)))
            px = px.astype(int)[0]
            py = py.astype(int)[0]
            chessboard = scan.update(chessboard, px, py, player)
            # dissplay(chessboard)
            if Record:
                if player == -1:
                    try:
                        p1 = np.append(
                            p1, np.stack(
                                (chessboard.reshape(-1) * -1, predict_opt.reshape(-1))).reshape(1, 2, 64), 0
                        )
                    except:
                        p1 = np.stack(
                            (chessboard.reshape(-1)*-1, predict_opt.reshape(-1))
                        ).reshape(1, 2, 64)
                else:
                    try:
                        p2 = np.append(
                            p2, np.stack(
                                (chessboard.reshape(-1), predict_opt.reshape(-1))).reshape(1, 2, 64), 0
                        )
                    except:
                        p2 = np.stack(
                            (chessboard.reshape(-1), predict_opt.reshape(-1))
                        ).reshape(1, 2, 64)
            print(f'下的位置: {row[py]}{px}    隨機下: {not Record}')
        else:
            if end_game:
                break
            end_game = True
        player = player * -1
    player_black = np.size(chessboard[chessboard == -1])
    player_white = np.size(chessboard[chessboard == 1])
    return player_black, player_white


error_test = ['D2', 'C2', 'E5', 'F3', 'F4', 'E2', 'C3', 'B3', 'G2', 'C5', 'C4', 'B5', 'F2', 'F6', 'E6', 'G1', 'C1', 'E1', 'F1', 'G0', 'H0', 'H3', 'H2', 'B0', 'H4', 'H1', 'F0', 'E7',
              'D0', 'G3', 'G4', 'D1', 'C0', 'E0', 'A0', 'G5', 'H5', 'H6', 'H7', 'G6', 'G7', 'F7', 'D7', 'D6', 'D5', 'A6', 'A5', 'A4', 'A3', 'A2', 'C6', 'C7', 'B4', 'B6', 'A7', 'B1', 'A1', 'B2', 'B7']
batch = 0
memory = 10
while True:
    if batch < 60 * memory:
        player_black, player_white = gamming(manual=manual, Probability=0)
        input_data = np.concatenate((p1[:, 0], p2[:, 0]))
        black = p1[:, 1]
        white = p2[:, 1]
        if player_black == player_white:
            print(
                f"平手!!\t黑棋數量: {player_black}\t白旗數量: {player_white}", end='\t')
            black[np.isinf(black)] = 1
            white[np.isinf(white)] = 1
        else:
            if player_black > player_white:
                print(
                    f"黑棋獲勝\t黑棋數量: {player_black}\t白旗數量: {player_white}", end='\t')
                black[np.isinf(black)] = 1
                white[np.isinf(white)] = -1
            else:
                print(
                    f"白棋獲勝\t黑棋數量: {player_black}\t白旗數量: {player_white}", end='\t')
                black[np.isinf(black)] = -1
                white[np.isinf(white)] = 1
        expect_data = np.concatenate((black, white))
        batch = np.size(input_data, 0)
        print(f'累計棋盤數: {batch}')
        tr.train(input_data, expect_data, batch)
    else:
        player_black, player_white = gamming(Probability=0)
        if player_black < player_white:
            print(f"主要網路獲勝 => 重新訓練")
            tr.update_weights(input_data, expect_data, batch)
        elif player_black > player_white:
            print("主要網路敗北 => 更新強度為敵對網路")
            mtr.update_weights(input_data, expect_data, batch)
        else:
            print("平手!! => 重新訓練敵對網路")
            tr.update_weights(input_data, expect_data, batch)
        batch = 0
        del p1
        del p2
