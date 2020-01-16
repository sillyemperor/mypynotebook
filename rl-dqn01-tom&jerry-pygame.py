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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N = 5
M = 64


def mk_state_map(xy):
    map = [0]*(N*N)
    map[xy[0]*N+xy[1]] = .9
    # print(map)
    return torch.Tensor(map)


class Environment:
    def __init__(self):
        self.cheese = [int(N*.6), int(N*.8)]
        self.tom = [int(N*.5), int(N*.8)]
        pygame.init()
        self.cheese_img = pygame.image.load(os.path.join(BASE_DIR, "data/cheese64.png"))
        self.tom_img = pygame.image.load(os.path.join(BASE_DIR, 'data/cat64.png'))
        self.jerry_img = pygame.image.load(os.path.join(BASE_DIR, 'data/mouse64.png'))
        self.screen = pygame.display.set_mode((N * M, N * M))

    def step(self, s, a):
        """
        :param s:(x,y)
        :param a:str
        :return:
        """
        if a == 0:
            d = np.array([-1, 0])
        elif a == 1:
            d = np.array([1, 0])
        elif a == 2:
            d = np.array([0, 1])
        elif a == 3:
            d = np.array([0, -1])
        s_ = s + d
        r = -.1
        done = False
        if s_.min() < 0 or s_.max() > (N - 1):  # 越界留在原地
            s_ = s
            r = -.2
        elif (s_ == self.cheese).all():
            done = True
            r = .9
        elif (s_ == self.tom).all():
            done = True
            r = -0.9
        return done, s_, r

    def render(self, s):
        pygame.event.get()
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.cheese_img, (self.cheese[0]*M, self.cheese[1]*M))
        self.screen.blit(self.tom_img, (self.tom[0]*M, self.tom[1]*M))
        self.screen.blit(self.jerry_img, (s[0]*M, s[1]*M))
        pygame.display.flip()
        time.sleep(.1)


EPSILON = 0.95
status = N*N
actions = 'up,down,left,right'.split(',')

policy_net = nn.Sequential(nn.Linear(status, len(actions)))
target_net = nn.Sequential(nn.Linear(status, len(actions)))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())

# print(policy_net(torch.tensor([1,0])))

optimize_step = 0


def get_action(s):
    """
    :param s:(x,y)
    :return:
    """
    if (np.random.uniform() > EPSILON) or optimize_step < 100: # 一定比例或者还没有修改以前，返回随机动作
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
        return
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

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        optimize_step += 1


env = Environment()
for e in range(MAX_EPISODES):
    s = START
    done = False
    counter = []
    while not done:
        env.render(s)

        # 预测值
        a = get_action(s)

        # 期望值
        done, s_, r = env.step(s, a)

        memory.push(mk_state_map(s), torch.tensor([a]), mk_state_map(s_), torch.tensor([r]))
        counter.append(f'{s},{a},{s_},{r}')
        s = s_

        optimize_model()

    print(e, len(counter), counter[-1])

    target_net.load_state_dict(policy_net.state_dict())
