"""
在0～8之间猜数字。猜中了奖励1，猜错了奖励0。
"""
import numpy as np
import pandas as pd
import time

START = 4
END = 1
EPSILON = 0.5
status = 9
actions = ('+', '-')

# Q表，保存状态和动作已经相关的价值，初始化为0
q_table = pd.DataFrame(np.zeros((status, len(actions))), columns=actions)
# print(q_table)

# print(q_table.iloc[0, :].idxmax())


def env_step(s, a):
    """
    获得环境反馈，通过当前状态和动作获得下一个状态和奖励
    :param s:
    :param a:
    :return:
    """
    if a == '+':
        s_ = s + 1
    else:
        s_ = s - 1

    r = 0
    if s_ > 8 or s_ < 0: # 走出边界本轮结束
        s_ = None
    elif s_ == END: # 目标达成本轮结束
        s_ = None
        r = 1
    return s_, r


def get_action(s):
    """
    根据状态获得动作
    :param s: 当前数字
    :return: 0-加一，1-减一
    """
    a = q_table.iloc[s, :]
    if (np.random.uniform() > EPSILON) or ((a == 0).all()): # 一定比例或者还没有修改以前，返回随机动作
        action_name = np.random.choice(actions)
    else: # 返回值最大的那个动作
        action_name = a.idxmax()
    return action_name


MAX_EPISODES = 100
GAMMA = 0.9
ALPHA = .1
for e in range(MAX_EPISODES):
    s = START
    done = False
    counter = []
    while not done:
        # 预测值
        a = get_action(s)
        prediction = q_table.loc[s, a]

        # 实际值
        s_, r = env_step(s, a)
        if s_ is not None:
            # Base on Bellman equation
            target = r + q_table.iloc[s_, :].max() * GAMMA
        else:
            target = r
            done = True

        # 修改QTable
        loss = (target - prediction) * ALPHA
        q_table.loc[s, a] += loss

        counter.append(str(s))
        counter.append(str(a))
        counter.append(str(s_))
        counter.append(str(r))
        counter.append(',')
        # print('\r', e, s, end='')
        # print(e, s, a, s_, r, loss)
        s = s_
    print(e, ''.join(counter))
    # print()


print(q_table)






