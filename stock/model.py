import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x , (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


def load_model(stock, num_classes=2, seq_length=8):
    input_size = 1
    hidden_size = 2
    num_layers = 1
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
    lstm.load_state_dict(torch.load(f'../data/ashare/models/{stock}-{seq_length}-{num_classes}.pt'))
    return lstm


def predict(model, data):
    sc = MinMaxScaler()
    model.eval()

    data = sc.fit_transform(data)

    dataX = torch.Tensor([data])

    r = model(dataX)
    r = r.detach().numpy()
    r = sc.inverse_transform(r)

    return r


def sliding_windows(data, seq_length, d):
    xx = []
    yy = []
    for i in range(len(data)-seq_length-d):
        _x = data[i:(i+seq_length)]
        _y = data[(i+seq_length):(i+seq_length+d)]
        xx.append(_x)
        yy.append(_y)
    return np.array(xx), np.array(yy)


def train(model, num_epochs, num_classes, trainX, trainY, learning_rate=0.01):
    model.train()
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    final_loss = 1
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(trainX)
        outputs = outputs.view(-1, num_classes, 1)
        #     print(outputs, trainY)
        # obtain the loss function
        loss = criterion(outputs, trainY)

        loss.backward()

        optimizer.step()
        final_loss = loss.item()
        # if epoch % 500 == 0:
        #     print("Epoch: %d, train-loss: %1.5f" % (epoch, final_loss))
    return final_loss
