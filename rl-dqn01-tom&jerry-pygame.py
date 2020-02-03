"""
5X5的方格，有一块奶酪，tom守在奶酪附近，jerry要怎么才能吃到奶酪又不被抓住呢
状态是jerry在方格内的坐标(x,y)，动作是(up,down,left,right)。
吃到奶酪，奖励1分，本轮结束。
落入tom的坐标，奖励-1分，本轮结束。

用NN替代QTable
"""
import numpy as np
import pandas as pd
import pygame
import time
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
import random
from env_tom_jerry import OneCheese, ThreeCheese

EVN = ThreeCheese

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N = 5
M = 64


def mk_state_map(xy):
    map = [0]*(N*N)
    map[xy[0]*N+xy[1]] = .9
    return torch.Tensor(map)


EPSILON = 0.8
status = N*N
actions = EVN.actions

# policy_net = nn.Sequential(nn.Linear(status, len(actions)))
# target_net = nn.Sequential(nn.Linear(status, len(actions)))
# 59 7 [3 3],2,[3 4],0.9 8.2
# 60 6 [2 3],2,[2 4],-0.9 8.0
# 61 8 [3 3],2,[3 4],0.9 8.2
# 62 7 [3 3],2,[3 4],0.9 7.4
# 63 8 [3 3],2,[3 4],0.9 7.2
# 64 7 [3 3],2,[3 4],0.9 7.2
policy_net = nn.Sequential(nn.Linear(status, status//2), nn.Linear(status//2, len(actions)))
target_net = nn.Sequential(nn.Linear(status, status//2), nn.Linear(status//2, len(actions)))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())

# print(policy_net(torch.tensor([1,0])))

optimize_step = 0


def get_action(s, train=True):
    """
    :param s:(x,y)
    :return:
    """
    if train and ((np.random.uniform() > EPSILON) or optimize_step < 100): # 一定比例或者还没有修改以前，返回随机动作
        return np.random.randint(0, 4)
    else:
        with torch.no_grad():
            s = mk_state_map(s)
            return policy_net(s).max(0)[1].item()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = ReplayMemory(10000)
BATCH_SIZE = 4
START = (0, 0)
MAX_EPISODES = 100
GAMMA = 0.9
ALPHA = .1


def optimize_model():
    global optimize_step
    if len(memory) < BATCH_SIZE:
        return 1
    total_loss = 0
    for i in range(BATCH_SIZE):
        trans = random.choice(memory.memory)

        s = trans.state
        a = trans.action
        s_ = trans.next_state
        r = trans.reward

        prediction = policy_net(s)
        targer = torch.zeros(BATCH_SIZE)
        targer[a] = r + target_net(s_).max(0)[0].detach() * GAMMA

        # Compute Huber loss
        loss = F.smooth_l1_loss(prediction, targer)

        total_loss += loss.max(0)[0].detach()

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        optimize_step += 1

    return total_loss


class RollList:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [0]*capacity
        self.position = 0

    def push(self, c):
        """Saves a transition."""
        self.memory[self.position] = c
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return self.memory.__iter__()


env = EVN(N, BASE_DIR)

meter = RollList(5)
for e in range(MAX_EPISODES):
    s = START
    done = False
    counter = []
    env.reset()
    loss = 0
    while not done:
        env.render(s)

        # 预测值
        a = get_action(s)

        # 期望值
        done, s_, r = env.step(s, a)

        memory.push(mk_state_map(s), torch.tensor([a]), mk_state_map(s_), torch.tensor([r]))
        counter.append(f'{s},{a},{s_},{r}')
        s = s_

        loss += optimize_model()
    c = len(counter)
    meter.push(c)
    avsteps = sum(meter.memory)/len(meter)
    print(e, c, counter[-1], loss/c, avsteps)

    target_net.load_state_dict(policy_net.state_dict())

policy_net.train()

counter = []
s = START
env.reset()
done = False
while not done:
    env.render(s)
    a = get_action(s, False)
    done, s_, r = env.step(s, a)
    counter.append(f'{s},{a},{s_},{r}')

c = len(counter)
print('test', c)
