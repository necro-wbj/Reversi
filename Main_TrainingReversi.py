import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


def create_model():
    print("正在建立類神經網路模型...")
    model = keras.Sequential()
    activation = keras.activations.softsign
    model.add(keras.layers.Dense(100, activation=activation, input_shape=(64,)))
    for _ in range(100):
        model.add(keras.layers.Dense(100, activation=activation))
    model.add(keras.layers.Dense(
        64, activation=activation))
    print("正在編譯類神經網路...")
    model.compile(
        optimizer=keras.optimizers.Nadam(),
                  loss=keras.losses.MAE)
    return model


if os.path.isfile('./Main_Reversi.h5'):
    print('初始化: ')
    model = keras.models.load_model('Main_Reversi.h5')
    print("讀取已保存的主要網路模型...")
else:
    main_checkpoint_path = './Main_Reversi.ckpt'
    checkpoint_path = './Reversi.ckpt'
    checkpoint_dir = os.path.dirname(main_checkpoint_path)
    model = create_model()
    print('讀取錯誤!!重新建立主要網路模型...')


def predict_opt(chessboard):
    chessboard = np.array(chessboard).reshape(1, 64)
    output = model.predict(chessboard).reshape(8, 8)
    return output

def update_weights(input_data, expect_data, batch):
    import TrainingReversi as tr
    global model
    tr.reload()
    print("主要網路更新權重...")
    model = keras.models.load_model('Reversi.h5')
    model.save('Main_Reversi.h5')

def reload():
    global model
    print('釋放GPU資源...')
    keras.backend.clear_session()
    model = keras.models.load_model('Reversi.h5')
