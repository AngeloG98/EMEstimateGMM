import logging
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)


class GMM:
    def __init__(self, pi=[], mean=[], cov=[], data=[]):
        self.pi = pi
        self.mean = mean
        self.cov = cov
        self.data = data
        self.K = len(pi)
        self.D = len(mean[0])
        self.N = len(data)

    def f(self, x, mean, cov):
        return (1 / np.sqrt(2 * np.pi * np.linalg.det(cov))) * np.exp(
            -0.5 *
            (np.transpose(x - mean)).dot(np.linalg.inv(cov)).dot(x - mean))

    def Gamma(self):
        gm_n_k = []
        for i in range(self.N):
            sum_k = (sum(self.pi[j] *
                         self.f(self.data[i], self.mean[j], self.cov[j])
                         for j in range(self.K)))
            gm_i_k = [
                ((self.pi[j] * self.f(self.data[i], self.mean[j], self.cov[j]))
                 / sum_k) for j in range(self.K)
            ]
            gm_n_k.append(gm_i_k)
        return gm_n_k

    def Mu(self, gm_n_k):
        gm_k = [sum([gm_n_k[j][i] for j in range(self.N)])
                for i in range(self.K)]
        mu = [sum([(gm_n_k[j][i]*self.data[j]) for j in range(self.N)])
              for i in range(self.K)]
        self.mean = [np.divide(mu[i], gm_k[i]) for i in range(self.K)]

    def Sigma(self, gm_n_k):
        gm_k = [sum([gm_n_k[j][i] for j in range(self.N)])
                for i in range(self.K)]
        sigma = [sum([(gm_n_k[j][i]*np.outer(self.data[j] - self.mean[i], self.data[j] - self.mean[i])) for j in range(self.N)])
                 for i in range(self.K)]
        self.cov = [np.divide(sigma[i], gm_k[i]) for i in range(self.K)]

    def Pi(self, gm_n_k):
        gm_k = [sum([gm_n_k[j][i] for j in range(self.N)])
                for i in range(self.K)]
        self.pi = np.divide(gm_k, self.N)

    def EM(self):
        delta = 1
        itera = 0
        while delta > 10e-7:
            pi_old = self.pi
            gm_n_k = self.Gamma()
            self.Sigma(gm_n_k)
            self.Mu(gm_n_k)
            self.Pi(gm_n_k)
            pi_new = self.pi
            delta = sum(abs(pi_new-pi_old))
            itera += 1
            if itera % 100 == 0:
                logging.info('iteration:'+str(itera)+'times.\n')
        logging.info('pi:\n')
        logging.info(self.pi)
        logging.info('mu:\n')
        logging.info(self.mean)
        logging.info('sigma:\n')
        logging.info(self.cov)
        logging.info('Done!')

        return self.pi, self.mean, self.cov


def get_input():
    N = 200
    plt.figure('Input Points')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title('add points = MouseLEFT \n pop points = MouseRIGHT \n  finish = MouseMIDDLE') 
    pos = plt.ginput(N, timeout=100)
    plt.close()
    poslist = [np.array(list(pos[i])) for i in range(len(pos))]
    return poslist


def plot_gmm(pi, mean, cov, pos):
    plt.figure('Output GMM')
    step = 0.1
    x = np.arange(0, 10, step)
    y = np.arange(0, 10, step)
    X, Y = np.meshgrid(x, y)
    points_x = [pos[i][0] for i in range(len(pos))]
    points_y = [pos[i][1] for i in range(len(pos))]
    plt.scatter(points_x, points_y)
    for w in range(len(pi)):
        Z = []
        for i in X[0]:
            k = []
            for j in Y:
                # allpos = np.array([i, j[0]])
                allpos = np.array([j[0], i])
                k.append((1 / np.sqrt(2 * np.pi * np.linalg.det(cov[w]))) * np.exp(-0.5 *
                                                                                   (np.transpose(allpos - mean[w])).dot(np.linalg.inv(cov[w])).dot(allpos - mean[w])))
                k_arr = np.array(k)
            Z.append(k_arr)
        plt.contour(X, Y, Z)
    plt.show()


if __name__ == "__main__":
    pos = get_input()

    pi_init = [0.8, 0.2]
    mean_init = [np.array([5.0, 0.0]), np.array([0.0, 5.0])]
    cov_init = [np.array([[1.0, 0.5], [0.5, 1.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0]])]

    gmm = GMM(pi_init, mean_init, cov_init, pos)
    pi, mean, cov = gmm.EM()

    plot_gmm(pi, mean, cov, pos)


