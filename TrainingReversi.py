import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


def create_model():
    print("正在建立類神經網路模型...")
    activation = keras.activations.softsign
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        64, activation=activation, input_shape=(64,)))
    for _ in range(100):
        model.add(keras.layers.Dense(100, activation=activation))
    model.add(keras.layers.Dense(
        64, activation=activation))
    print("正在編譯類神經網路...")
    model.compile(
        optimizer=keras.optimizers.Nadam(),
        loss=keras.losses.MSE)
    return model


if os.path.isfile('./Main_Reversi.h5'):
    print("讀取已保存的學習網路模型...")
    model = keras.models.load_model('Main_Reversi.h5')
else:
    print('讀取錯誤!!重新建立學習網路模型...')
    model = create_model()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
Mweights = model.get_weights()
Tweights = Mweights

def predict_opt(chessboard,Main = True):
    global model,Mweights,Tweights
    chessboard = chessboard.reshape(1, 64)
    if Main:
        model.set_weights(Mweights)
    else:
        model.set_weights(Tweights)
    try:
        output = model.predict(chessboard).reshape(8, 8)
    except:
        tf.keras.backend.set_session(session)
        output = model.predict(chessboard).reshape(8, 8)
    return output


loss = '未知'
def train(input_data, expect_data, batch):
    global loss,model,Tweights,Mweights
    model.set_weights(Tweights)
    callback = [tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=100, min_delta=0.000001)]
    print(f'目前誤差: {loss}')
    print("開始訓練...")
    model.fit(input_data, expect_data, epochs=1000,
                    batch_size=batch, verbose=0, callbacks=callback)
    loss = model.evaluate(x=input_data, y=expect_data,
                          batch_size=batch, verbose=0, use_multiprocessing=True)
    # print(f'訓練後誤差: {loss}')
    Tweights = model.get_weights()
    if not os.path.isfile('./Main_Reversi.h5'):
        model.save('Main_Reversi.h5')
    


def update_weights(input_data, expect_data, batch,Loss = True):
    global model,Mweights,Tweights
    import Main_TrainingReversi as mt
    if Loss:    #輸主網路
        print("更新網路權重為主要網路....")
        model.set_weights(Mweights)
        Tweights = Mweights
        print("更新完成!!")
    else:
        model.set_weights(Tweights)
        Mweights = Tweights
        print("儲存...")
        model.save('Main_Reversi.h5')
        print("儲存完成!!")
