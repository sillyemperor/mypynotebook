import numpy as np
import pandas as pd
import torch

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from model import LSTM, sliding_windows, train


def train_model(stock, col):
    data = pd.read_csv(f'../data/ashare/{stock}.csv', converters={
        0: lambda x: datetime.strptime(x, '%Y/%m/%d')
    })
    data = data.sort_index(ascending=False)
    training_set = data.iloc[:, col].values

    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set.reshape(-1, 1))
    # print(training_data)

    num_classes = 2
    seq_length = 8

    x, y = sliding_windows(training_data, seq_length, num_classes)
    print(x.shape)
    print(y.shape)

    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
    # print(trainX)
    # print(trainY)

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    num_epochs = 1500
    learning_rate = 0.01

    input_size = 1
    hidden_size = 2
    num_layers = 1

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

    train(lstm, num_epochs, num_classes, trainX, trainY, learning_rate)

    torch.save(lstm.state_dict(), f'../data/ashare/models/{stock}-col{col}-8-2.pt')

if __name__ == '__main__':
    train_model('300127', 3)  # 收盘价
    train_model('300127', 4)  # 最高价
    train_model('300127', 5)  # 最低价
    train_model('300127', 6)  # 开盘价
    train_model('300127', 11)  # 成交量











