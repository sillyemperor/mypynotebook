import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from model import LSTM, predict, load_model



if __name__ == '__main__':

    data = [[3.35],[3.31],[3.31],[3.35],[3.34],[3.36],[3.35],[3.42]]
    data.reverse()
    print('601600', predict(load_model('601600'), data))

    raw = [[5.54],[5.52],[5.5],[5.6]]
    raw.reverse()
    print('601857', predict(load_model('601857', 1, 4), raw))

    data = [[13.67],[13.15],[12.56],[13.38],[13.55],[14.17],[14.2],[12.91]]
    data.reverse()
    print('300472', predict(load_model('300472'), data))

    data = [[10.07],[10.02],[10.38],[9.83],[8.94],[8.86],[9.3],[9.85]]
    data.reverse()
    print('002332', predict(load_model('002332'), data))

    data = [[17.3],[17.08],[17.05],[16.73],[16.69],[16.58],[17.23],[17.25]]
    data.reverse()
    print('300127', predict(load_model('300127'), data))
