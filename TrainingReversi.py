import Main_TrainingReversi as mt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

if os.path.isfile('./Main_Reversi.h5'):
    print("讀取已保存的學習網路模型...")
    Tmodel = keras.models.load_model('Main_Reversi.h5')
else:
    print('讀取錯誤!!重新建立學習網路模型...')
    Tmodel = mt.create_model()


def predict_opt(chessboard):
    chessboard = chessboard.reshape(1, 64)
    output = Tmodel.predict(chessboard).reshape(8, 8)
    return output

loss = '未知'
def train(input_data, expect_data, batch):
    global loss
    callback = [tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=100, min_delta=0.000001)]
    print(f'目前誤差: {loss}')
    print("開始訓練...")
    Tmodel.fit(input_data, expect_data, epochs=10000,
                    batch_size=batch, verbose=0, callbacks=callback)
    loss = Tmodel.evaluate(x=input_data, y=expect_data,
                          batch_size=batch, verbose=0, use_multiprocessing=True)
    print(f'訓練後誤差: {loss}')
    if not os.path.isfile('./Main_Reversi.h5'):
        Tmodel.save('Main_Reversi.h5')
    


def update_weights(input_data, expect_data, batch):
    global Tmodel
    import Main_TrainingReversi as mt
    print("更新網路權重為主要網路....")
    Tmodel = keras.models.load_model('Main_Reversi.h5')
    print("更新完成!!")


def weights():
    global Tmodel
    return Tmodel.get_weights()
