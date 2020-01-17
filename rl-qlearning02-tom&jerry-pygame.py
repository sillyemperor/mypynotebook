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
from env_tom_jerry import OneCheese, ThreeCheese

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


N = 5
EPSILON = 0.9
status = N*N
actions = OneCheese.actions

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


env = ThreeCheese(N, BASE_DIR)

START = (0, 0)
MAX_EPISODES = 100
GAMMA = 0.9
ALPHA = .1
for e in range(MAX_EPISODES):
    s = START
    done = False
    counter = []
    env.reset()
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

        counter.append(f'{s},{a},{s_},{r}')
        # print('\r', e, s, end='')
        # print(e, s, a, s_, r, loss)
        s = s_

        env.render(s)

    print(e, len(counter), counter[-1])
    # print()


print(q_table)
'''
          up      down      left     right
0  -0.060628  0.447905 -0.131869 -0.087363
1  -0.098020 -0.092701 -0.092640 -0.034588
2  -0.059402 -0.062385 -0.063991 -0.074300
3  -0.059402 -0.056601 -0.056018 -0.049455
4  -0.059402 -0.046607 -0.059402 -0.050505
5  -0.069439  0.616891 -0.085178 -0.043237
6  -0.073021 -0.100000 -0.071535  0.042833
7  -0.050537 -0.049944 -0.046468 -0.054373
8  -0.048269 -0.044354 -0.048517 -0.040198
9  -0.039511 -0.044200 -0.039800 -0.048866
10  0.038648  0.799347 -0.100000  0.048352
11  0.000000  0.000000  0.000000  0.000000
12 -0.019900  0.019241 -0.020791 -0.100000
13 -0.031390 -0.035371 -0.031239 -0.029620
14 -0.038531 -0.037793 -0.039800 -0.030503
15  0.042361 -0.018081  0.999931  0.177741
16  0.000000  0.000000  0.000000  0.000000
17 -0.010000 -0.010000 -0.010000  0.468559
18 -0.030430 -0.019900 -0.019900  0.025146
19 -0.029710 -0.028810 -0.039791 -0.028081
20  0.194787 -0.020000 -0.010000 -0.020000
21  0.000000  0.000000  0.000000 -0.010000
22 -0.010000  0.000000 -0.010000  0.000000
23 -0.019900 -0.038000 -0.010000 -0.010000
24 -0.029701 -0.020000 -0.020000 -0.010000
'''