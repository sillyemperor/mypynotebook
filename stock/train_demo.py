import numpy as np
import pandas as pd
import torch

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from model import LSTM, sliding_windows, train



if __name__ == '__main__':
    stock = '300127'

    data = pd.read_csv(f'../data/ashare/{stock}.csv', encoding='GBK', converters={
        0:lambda x:datetime.strptime(x, '%Y-%m-%d')
    })

    data = data.sort_index(ascending=False)

    training_set = data.iloc[:,3].values

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

    torch.save(lstm.state_dict(), f'../data/ashare/models/{stock}-8-2.pt')

