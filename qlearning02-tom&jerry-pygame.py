"""
5X5的方格，有一块奶酪，tom守在奶酪附近，jerry要怎么才能吃到奶酪又不被抓住呢
状态是jerry在方格内的坐标(x,y)，动作是(up,down,left,right)。
吃到奶酪，奖励1分，本轮结束。
落入tom的坐标，奖励-1分，本轮结束。
"""
import numpy as np
import pandas as pd
import pygame
import time
import os.path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Environment:
    def __init__(self, N):
        self.cheese_img = pygame.image.load(os.path.join(BASE_DIR, "data/cheese64.png"))
        self.tom_img = pygame.image.load(os.path.join(BASE_DIR, 'data/cat64.png'))
        self.jerry_img = pygame.image.load(os.path.join(BASE_DIR, 'data/mouse64.png'))
        self.cheese = [int(N*.6), int(N*.3)]
        self.tom = [int(N*.5), int(N*.3)]
        pygame.init()
        self.screen = pygame.display.set_mode((N * 64, N * 64))

    def step(self, s, a):
        """
        :param s:(x,y)
        :param a:str
        :return:
        """
        if a == 'up':
            d = np.array([-1, 0])
        elif a == 'down':
            d = np.array([1, 0])
        elif a == 'left':
            d = np.array([0, 1])
        elif a == 'right':
            d = np.array([0, -1])
        s_ = s + d
        r = -.1
        done = False
        if s_.min() < 0 or s_.max() > (N - 1):  # 越界留在原地
            s_ = s
            r = -.2
        elif (s_ == self.cheese).all():
            done = True
            r = 1.0
        elif (s_ == self.tom).all():
            done = True
            r = -1.0
        return done, s_, r

    def render(self, s):
        pygame.event.get()
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.cheese_img, (self.cheese[0]*64, self.cheese[1]*64))
        self.screen.blit(self.tom_img, (self.tom[0] * 64, self.tom[1] * 64))
        self.screen.blit(self.jerry_img, (s[0] * 64, s[1] * 64))
        pygame.display.flip()
        time.sleep(.1)


N = 5
EPSILON = 0.9
status = N*N
actions = 'up,down,left,right'.split(',')

q_table = pd.DataFrame(np.zeros((status, len(actions))), columns=actions)


def get_action(s):
    """
    :param s:(x,y)
    :return:
    """
    l = s[0]*N+s[1]
    a = q_table.iloc[l, :]
    if (np.random.uniform() > EPSILON) or ((a == 0).all()): # 一定比例或者还没有修改以前，返回随机动作
        action_name = np.random.choice(actions)
    else: # 返回值最大的那个动作
        action_name = a.idxmax()
    return action_name


env = Environment(N)

START = (0, 0)
MAX_EPISODES = 100
GAMMA = 0.9
ALPHA = .1
for e in range(MAX_EPISODES):
    s = START
    done = False
    counter = []
    while not done:

        l = s[0] * N + s[1]
        # 预测值
        a = get_action(s)
        prediction = q_table.loc[l, a]

        # 实际值
        done, s_, r = env.step(s, a)
        if not done:
            # Base on Bellman equation
            l_ = s_[0] * N + s_[1]
            target = r + q_table.iloc[l_, :].max() * GAMMA
        else:
            target = r

        # 修改QTable
        loss = (target - prediction) * ALPHA
        q_table.loc[l, a] += loss

        counter.append(str(s))
        counter.append(str(a))
        counter.append(str(s_))
        counter.append(str(r))
        counter.append(',')
        # print('\r', e, s, end='')
        # print(e, s, a, s_, r, loss)
        s = s_

        env.render(s)

    print(e, ''.join(counter))
    # print()


print(q_table)

