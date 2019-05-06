import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


def create_model():
    print("正在建立類神經網路模型...")
    activation = keras.activations.softsign
    Mmodel = keras.Sequential()
    Mmodel.add(keras.layers.Dense(
        64, activation=activation, input_shape=(64,)))
    for _ in range(100):
        Mmodel.add(keras.layers.Dense(100, activation=activation))
    Mmodel.add(keras.layers.Dense(
        64, activation=activation))
    print("正在編譯類神經網路...")
    Mmodel.compile(
        optimizer=keras.optimizers.Nadam(),
        loss=keras.losses.MSE)
    return Mmodel


Mmodel = create_model()
# if os.path.isfile('./Main_Reversi.h5'):
#     print('初始化: ')
#     keras.backend.clear_session()
#     Mmodel = keras.models.load_model('Main_Reversi.h5')
#     print("讀取已保存的主要網路模型...")
# else:
#     Mmodel = create_model()
#     print('讀取錯誤!!重新建立主要網路模型...')


def predict_opt(chessboard):
    global Mmodel  # 不加時使用flask會遺失導致錯誤
    chessboard = chessboard.reshape(1, 64)
    try:
        output = Mmodel.predict(chessboard)
    except Exception as e:
        print('錯誤!!!')
        print(e)
        keras.backend.clear_session()
        print('重新讀取....')
        Mmodel = keras.models.load_model('Main_Reversi.h5')
        print("讀取已保存的主要網路模型...")
        output = Mmodel.predict(chessboard)
    return output.reshape(8, 8)


def update_weights(input_data, expect_data, batch):
    global Mmodel
    import TrainingReversi as tr
    Mmodel.set_weights(tr.weights())
    print("儲存...")
    Mmodel.save('Main_Reversi.h5')
    print("儲存完成!!")


# def reload():
#     global model
#     try:
#         if os.path.isfile('./Main_Reversi.h5'):
#             print('重新讀取權重...')
#             model = keras.models.load_model('Reversi.h5')
#         else:
#             print('尚未有權重，執行儲存...')
#             model.save('Main_Reversi.h5')
#     except:
#         print('釋放GPU資源...')
#         keras.backend.clear_session()
#         model = keras.models.load_model('Reversi.h5')

