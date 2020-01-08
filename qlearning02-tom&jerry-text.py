"""
5X5的方格，有一块奶酪，tom守在奶酪附近，jerry要怎么才能吃到奶酪又不被抓住呢
状态是jerry在方格内的坐标(x,y)，动作是(up,down,left,right)。
吃到奶酪，奖励一分，本轮结束。
超出边界，奖励0分，本轮结束。
落入tom的坐标，奖励-1分，本轮结束。
"""
import numpy as np
import pandas as pd


N = 5
# 奶酪
cheese = [3, 2]
# tom
tom = [2, 3]


def env_step(s, a):
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
    r = .0
    if s_.min() < 0 or s_.max() > (N-1):
        s_ = None
    elif (s_ == cheese).all():
        s_ = None
        r = 1.0
    elif (s_ == tom).all():
        s_ = None
        r = -1.0
    return s_, r


EPSILON = 0.5
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
        s_, r = env_step(s, a)
        if s_ is not None:
            # Base on Bellman equation
            l_ = s_[0] * N + s_[1]
            target = r + q_table.iloc[l_, :].max() * GAMMA
        else:
            target = r
            done = True

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
    print(e, ''.join(counter))
    # print()


print(q_table)

