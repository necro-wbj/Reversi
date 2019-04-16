import Main_TrainingReversi as mt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

if os.path.isfile('./Reversi.h5'):
    print("讀取已保存的學習網路模型...")
    model = keras.models.load_model('Reversi.h5')
else:
    print('讀取錯誤!!重新建立學習網路模型...')
    model = mt.create_model()


def predict_opt(chessboard):
    chessboard = np.array(chessboard).reshape(1, 64)
    output = model.predict(chessboard).reshape(8, 8)
    return output

loss = '未知'
def train(input_data, expect_data, batch):
    global loss
    input_data = np.array(input_data)
    expect_data = np.array(expect_data)
    callback = [tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=100, min_delta=0.000001)]
    print(f'目前誤差: {loss}')
    print("開始訓練...")
    model.fit(input_data, expect_data, epochs=10000,
              batch_size=batch, verbose=0, callbacks=callback)
    loss = model.evaluate(x=input_data, y=expect_data,
                          batch_size=batch, verbose=0, use_multiprocessing=True)
    print(f'訓練後誤差: {loss}')
    # print(f'訓練後誤差: {loss:.5f}')
    model.save('Reversi.h5')
    if not os.path.isfile('./Main_Reversi.h5'):
        model.save('Main_Reversi.h5')


def update_weights(input_data, expect_data, batch):
    import Main_TrainingReversi as mt
    global model
    mt.reload()
    if os.path.isfile('./Main_Reversi.h5'):
        print('正在讀取主要網路的模型...')
        model = keras.models.load_model('Main_Reversi.h5')
    else:
        print('主網路不存在，將學習網路設為主網路...')
        model.save('Main_Reversi.h5')
    print("更新網路權重為主要網路...")
    model.save('Reversi.h5')


def reload():
    global model
    print('釋放GPU資源...')
    keras.backend.clear_session()
    model = keras.models.load_model('Main_Reversi.h5')
