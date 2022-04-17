from dis import dis
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)


class MultiBernoulli:
    def __init__(self, pi=[], p=[], size=100):
        self.pi = pi
        self.p = p
        self.K = len(pi)
        self.size = size

    def get_discrete01(self):
        pi = [0.5, 0.3, 0.2] # sum one!
        p = [0.7, 0.9, 0.4] # probability equal to 1, coin up
        n = 10000
        discrete01 = []
        for _ in range(n):
            rand = random.random()
            if rand <= pi[0]:
                number = np.random.binomial(1, p[0], self.size)
            elif rand > pi[0] and rand <= pi[0]+pi[1]:
                number = np.random.binomial(1, p[1], self.size)
            else:
                number = np.random.binomial(1, p[2], self.size)
            discrete01.append(number)
        self.data = discrete01
        self.N = len(self.data)
        
    def f(self, x, p):
        pro = 1
        for xi in x:
            pro *= ( p if xi == 1 else (1-p) )
        return pro

    def Gamma(self):
        gm_n_k = []
        for i in range(self.N):
            aa = self.f(self.data[i], self.p[0])
            bb = self.f(self.data[i], self.p[1])
            sum_k = sum( [self.pi[j] * self.f(self.data[i], self.p[j]) for j in range(self.K)] )
            gm_i_k = [((self.pi[j] * self.f(self.data[i], self.p[j])) / sum_k) for j in range(self.K)]
            gm_n_k.append(gm_i_k)
        return gm_n_k

    def P(self, gm_n_k):
        gm_k = [sum([gm_n_k[j][i]*self.size for j in range(self.N)]) for i in range(self.K)]
        p = [sum([(gm_n_k[j][i]*sum(self.data[j])) for j in range(self.N)]) for i in range(self.K)]
        self.p = [np.divide(p[i], gm_k[i]) for i in range(self.K)]

    def Pi(self, gm_n_k):
        gm_k = [sum([gm_n_k[j][i] for j in range(self.N)]) for i in range(self.K)]
        self.pi = np.divide(gm_k, self.N)

    def EM(self):
        itera = 0
        self.get_discrete01()
        while itera < 100:
            # self.get_discrete01()
            pi_old = self.pi
            gm_n_k = self.Gamma()
            self.P(gm_n_k)
            self.Pi(gm_n_k)
            pi_new = self.pi
            itera += 1
            if itera % 1 == 0:
                logging.info('iteration:'+str(itera)+'times.\n')
                logging.info('pi:')
                logging.info(self.pi)
                logging.info('p:')
                logging.info(self.p)

        return self.pi, self.p


if __name__ == "__main__":
    pi_init = [0.3, 0.3, 0.4]
    p_init = [0.8, 0.1, 0.5]
    
    b = MultiBernoulli(pi_init, p_init, 50)
    pi, p = b.EM()

    print("pi is {}".format(pi))
    print("p is {}".format(p))


