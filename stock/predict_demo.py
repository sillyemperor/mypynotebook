import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from model import LSTM, predict, load_model



if __name__ == '__main__':

    stock = '300127'

    data = [[18.08], [19.03],[17.3],[17.08],[17.05],[16.73],[16.69],[16.58]]
    data.reverse()
    print(stock, 'close', predict(load_model(stock, 3), data))

    data = [[19.13],[19.03],[17.45],[17.24],[17.77],[16.84],[16.95],[17.32]]
    data.reverse()
    print(stock, 'max', predict(load_model(stock, 4), data))

    data = [[18.06],[17.11],[17.01],[16.56],[16.5],[16.39],[16.36],[16.53]]
    data.reverse()
    print(stock, 'min', predict(load_model(stock, 5), data))

    data = [[18.6],[17.25],[17.04],[16.77],[16.52],[16.48],[16.87],[16.92]]
    data.reverse()
    print(stock, 'open', predict(load_model(stock, 6), data))

    data = [[20599565],[24971351],[7078286],[7078251],[11384101],[4807995],[5009700],[7728003]]
    data.reverse()
    print(stock, 'volume', predict(load_model(stock, 11), data))
