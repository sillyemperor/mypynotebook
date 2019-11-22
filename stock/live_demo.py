import time
import tushare as ts
from model import LSTM
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def local_price(file):
    data = pd.read_csv(file, usecols=[0, 1, 3], converters={
        0: lambda x: datetime.strptime(x, '%H:%M:%S')
    })

    for i in data.iloc[:,1].values:
        yield i


def live_price(stock):
    t1 = time.time()
    while True:
        try:
            df = ts.get_realtime_quotes(stock)
            price = df['price'][0]
            yield price
        except:
            continue
        d = 3 - (time.time() - t1)
        t1 = time.time()
        if d > 0:
            time.sleep(d)


loader = local_price('../data/ashare/30012720191120.csv')
# loader = live_price('300127')


from model import *

num_epochs = 100

num_classes = 10
seq_length = 40

input_size = 1
hidden_size = 2
num_layers = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

sc = MinMaxScaler()

bucket = []
data = []
predict_y = None
aloss_list = []
loss_list = []
x_list = []
y_list = []
plt.ion()
for price in loader:

    bucket.append([float(price)])
    # print(bucket, data)
    if len(bucket) >= seq_length:
        data.append(bucket)

        if len(data) > 1:
            if predict_y is not None:
                x = torch.tensor(predict_y)
                y = torch.tensor(bucket[:num_classes]).view(-1)
                loss = y - x
                aloss = loss.sum() / num_classes

                loss = list(loss.view(-1).numpy())
                x = list(x.view(-1).numpy())
                y = list(y.view(-1).numpy())

                loss_list += loss
                x_list += x
                y_list += y
                aloss_list.append(aloss)

                plt.subplot(2, 2, 1)
                plt.plot(loss)
                plt.subplot(2, 2, 2)
                plt.plot(loss_list)
                plt.subplot(2, 2, 3)
                plt.plot(y, c='g')
                plt.plot(x, c='r')
                plt.subplot(2, 2, 4)
                plt.plot(y_list, c='g')
                plt.plot(x_list, c='r')
                plt.draw()
                plt.pause(0.0001)
                plt.clf()

                # print(x)
            #   print(y)
                print(aloss, elapsed)
                print()



            t1 = time.time()
            training_data = torch.Tensor(data)
            training_data = sc.fit_transform(training_data.view(-1, 1))
            # training_data = torch.Tensor([training_data])

            x, y = sliding_windows(training_data, seq_length, num_classes)
            trainX = torch.Tensor(np.array(x))
            trainY = torch.Tensor(np.array(y))

            loss = train(lstm, num_epochs, num_classes, trainX, trainY)
            elapsed = time.time() - t1

            predict_data = data[-1]
            predict_y = predict(lstm, predict_data)
            print(predict_y)

        bucket = bucket[num_classes:]
