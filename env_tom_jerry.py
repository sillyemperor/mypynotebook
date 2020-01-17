import pygame
import os.path
import numpy as np
import time


class OneCheese:
    actions = range(4)
    M = 64

    def __init__(self, N, BASE_DIR):
        self.N = N
        self.cheese = [int(N*.6), int(N*.8)]
        self.tom = [int(N*.5), int(N*.8)]
        pygame.init()
        self.cheese_img = pygame.image.load(os.path.join(BASE_DIR, "data/cheese64.png"))
        self.tom_img = pygame.image.load(os.path.join(BASE_DIR, 'data/cat64.png'))
        self.jerry_img = pygame.image.load(os.path.join(BASE_DIR, 'data/mouse64.png'))
        self.screen = pygame.display.set_mode((N * self.M, N * self.M))

    def reset(self): pass

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
        if s_.min() < 0 or s_.max() > (self.N - 1):  # 越界留在原地
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
        self.screen.blit(self.cheese_img, (self.cheese[0]*self.M, self.cheese[1]*self.M))
        self.screen.blit(self.tom_img, (self.tom[0]*self.M, self.tom[1]*self.M))
        self.screen.blit(self.jerry_img, (s[0]*self.M, s[1]*self.M))
        pygame.display.flip()
        time.sleep(.1)


class ThreeCheese:
    actions = range(4)
    M = 64

    def __init__(self, N, BASE_DIR):
        self.N = N
        pygame.init()
        self.cheese_img = pygame.image.load(os.path.join(BASE_DIR, "data/cheese64.png"))
        self.tom_img = pygame.image.load(os.path.join(BASE_DIR, 'data/cat64.png'))
        self.jerry_img = pygame.image.load(os.path.join(BASE_DIR, 'data/mouse64.png'))
        self.screen = pygame.display.set_mode((N * self.M, N * self.M))

    def reset(self):
        self.cheese = (
            [int(self.N * .6), int(self.N * .8), 1],
            [int(self.N * .6), int(self.N * .6), 1],
            [int(self.N * .5), int(self.N * .6), 1],
        )
        self.num_cheese = len(self.cheese)
        self.tom = [int(self.N * .5), int(self.N * .8)]

    def render(self, s):
        pygame.event.get()
        self.screen.fill((255, 255, 255))
        for i in self.cheese:
            if i[-1]:
                self.screen.blit(self.cheese_img, (i[0]*self.M, i[1]*self.M))
        self.screen.blit(self.tom_img, (self.tom[0]*self.M, self.tom[1]*self.M))
        self.screen.blit(self.jerry_img, (s[0]*self.M, s[1]*self.M))
        pygame.display.flip()
        time.sleep(.1)

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
        if s_.min() < 0 or s_.max() > (self.N - 1):  # 越界留在原地
            s_ = s
            r = -.2
        elif (s_ == self.tom).all():
            done = True
            r = -0.9
        else:
            for i in self.cheese:
                if i[-1]:
                    if (s_ == i[:2]).all():
                        i[-1] = 0
                        r = .9
                        self.num_cheese -= 1
                        done = not self.num_cheese
                # else:
                #     r = -.1
        return done, s_, r

