# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: vmf.py
@Time: 2019-11-06 10:06
@Desc: vmf.py
"""
try:
    import numpy as np
    import scipy.io as scio
    import argparse
    import time

    from scipy.special import digamma, iv
    from sklearn.cluster import KMeans
    from numpy.matlib import repmat
    izip = zip

    from vmfmm.utils import cluster_acc, predict, calculate_mix, d_besseli, d_besseli_low, caculate_pi, log_normalize
except ImportError as e:
    print(e)
    raise ImportError


class VMFMixture:
    """
    Variational Inference Dirichlet process Mixture Models of datas Distributions
    """
    def __init__(self, n_cluster, max_iter):

        self.T = n_cluster
        self.max_k = 0
        self.max_iter = max_iter
        self.N = 300
        self.D = 3
        self.prior = dict()
        self.pi = None

        self.gamma = None
        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None

        self.rho = None
        self.g = None
        self.h = None

        self.temp_zeta = None
        self.det = 1e-10

    def init_params(self, data):

        (self.N, self.D) = data.shape

        self.prior = {
            'mu': np.sum(data, 0) / np.linalg.norm(np.sum(data, 0)),
            'zeta': 0.05,
            'u': 1,
            'v': 0.01,
            'gamma': 1,
        }

        while np.isfinite(iv(self.D / 2, self.max_k + 10)):
            self.max_k = self.max_k + 10

        self.u = np.ones(self.T)
        self.v = np.ones(self.T) * 0.01
        self.zeta = np.ones(self.T)
        self.xi = np.ones((self.T, self.D))

        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.rho = np.ones((self.N, self.T)) * (1 / self.T)
        self.g = np.zeros(self.T)
        self.h = np.zeros(self.T)

        self.update_zeta_xi(data, self.rho)
        self.update_u_v(self.rho)
        self.update_g_h(self.rho)

    def var_inf(self, x):

        D = self.D
        begin = time.time()
        for ite in range(self.max_iter):
            # compute rho
            E_log_1_pi = np.roll(np.cumsum(digamma(self.h) - digamma(self.g + self.h)), 1)
            E_log_1_pi[0] = 0
            self.rho = x.dot((self.xi * (self.u / self.v)[:, np.newaxis]).T) + (D / 2 - 1) * \
                       (digamma(self.u) - np.log(self.v)) - \
                       (D / 2 * np.log(2 * np.pi)) - \
                       (d_besseli(D / 2 - 1, self.k)) * (self.u / self.v - self.k) - \
                       np.log(iv((D / 2 - 1), self.k) + np.exp(-700)) + \
                       digamma(self.g) - digamma(self.g + self.h) + E_log_1_pi
            if np.any(np.isnan(self.rho)):
                print(1)
            self.rho = np.exp(self.rho) / np.expand_dims(np.sum(np.exp(self.rho), 1), 1)
            if np.any(np.isnan(self.rho)):
                print(1)

            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi(x, self.rho)
            self.update_u_v(self.rho)
            self.update_g_h(self.rho)

            if ite == self.max_iter-1:
                times = time.time() - begin
                self.pi = calculate_mix(self.g, self.h, self.T)
                print('times: {}'.format(times))

    def update_u_v(self, rho):

        D = self.D
        # compute u, v
        self.u = self.prior['u'] + (D / 2 - 1) * np.sum(rho, 0) + \
                 self.zeta * self.k * (d_besseli_low(D / 2 - 1, self.zeta * self.k))
        if np.any(np.isnan(self.u)):
            print(1)
        self.v = self.prior['v'] + (d_besseli(D / 2 - 1, self.k)) * np.sum(rho, 0) \
                 + self.prior['zeta'] * (d_besseli(D / 2 - 1, self.prior['zeta'] * self.k))
        if np.any(np.isnan(self.v)):
            print(1)

    def update_zeta_xi(self, x, rho):

        # compute zeta, xi
        temp = np.expand_dims(self.prior['zeta'] * self.prior['mu'], 0) + rho.T.dot(x)
        self.zeta = np.linalg.norm(temp, axis=1)
        if np.any(np.isnan(self.zeta)):
            print(1)
        self.xi = temp / self.zeta[:, np.newaxis]
        if np.any(np.isnan(self.xi)):
            print(1)

    def update_g_h(self, rho):
        # compute g, h
        N_k = np.sum(rho, 0)
        self.g = 1 + N_k
        for i in range(self.T):
            if i == self.T - 1:
                self.h[i] = self.prior['gamma']
            else:
                temp = rho[:, i + 1:self.T]
                self.h[i] = self.prior['gamma'] + np.sum(np.sum(temp, 1), 0)

    def fit(self, data):

        self.init_params(data)
        self.var_inf(data)

        return self

    def predict(self, data):
        # predict
        pred = predict(data, mu=self.xi, k=self.k, pi=self.pi, n_cluster=self.T)
        return pred

    def fit_predict(self, data):

        self.fit(data)
        return self.predict(data)


class CVMFMixture:
    """
    Collapsed Variational Inference Dirichlet process Mixture Models of datas Distributions
    """
    def __init__(self, n_cluster, max_iter=100):

        self.T = n_cluster
        self.max_k = 0
        self.max_iter = max_iter
        self.N = 300
        self.D = 3
        self.prior = dict()
        self.pi = None

        self.gamma = None
        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None

        self.rho = None

        self.temp_zeta = None
        self.det = 1e-10

    def init_params(self, data):

        (self.N, self.D) = data.shape

        self.prior = {
            'mu': np.sum(data, 0) / np.linalg.norm(np.sum(data, 0)),
            'zeta': 0.01,
            'u': 0.5,
            'v': 0.01,
            'gamma': 1,
        }

        while np.isfinite(iv(self.D / 2, self.max_k + 10)):
            self.max_k = self.max_k + 10

        self.u = np.ones(self.T) * self.prior['u']
        self.v = np.ones(self.T) * self.prior['v']
        self.zeta = np.ones(self.T)
        self.xi = np.ones((self.T, self.D))

        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.rho = np.ones((self.N, self.T)) * (1 / self.T)

        self.update_zeta_xi(data, self.rho)
        self.update_u_v(self.rho)

    def caclulate_log_lik_x(self, x):

        D = self.D
        log_like_x = x.dot((self.xi * (self.u / self.v)[:, np.newaxis]).T) + (D / 2 - 1) * \
        (digamma(self.u) - np.log(self.v)) - \
        (D / 2 * np.log(2 * np.pi)) - \
        (d_besseli(D / 2 - 1, self.k)) * (self.u / self.v - self.k) - \
        np.log(iv((D / 2 - 1), self.k) + np.exp(-700))

        return log_like_x

    def caclulate_rho(self, x):

        gamma = self.prior['gamma']
        log_like_x = self.caclulate_log_lik_x(x)

        E_Nc_minus_n = np.sum(self.rho, 0, keepdims=True) - self.rho
        E_Nc_minus_n_cumsum_geq = np.fliplr(np.cumsum(np.fliplr(E_Nc_minus_n), axis=1))
        E_Nc_minus_n_cumsum = E_Nc_minus_n_cumsum_geq - E_Nc_minus_n

        first_tem = np.log(1 + E_Nc_minus_n) - np.log(1 + gamma + E_Nc_minus_n_cumsum_geq)
        first_tem[:, self.T - 1] = 0
        dummy = np.log(gamma + E_Nc_minus_n_cumsum) - np.log(1 + gamma + E_Nc_minus_n_cumsum_geq)
        second_term = np.cumsum(dummy, axis=1) - dummy

        rho = log_like_x + (first_tem + second_term)
        log_rho, log_n = log_normalize(rho)
        rho = np.exp(log_rho)

        return rho

    def var_inf(self, x):

        begin = time.time()
        for ite in range(self.max_iter):
            # compute rho
            self.rho = self.caclulate_rho(x)

            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi(x, self.rho)
            self.update_u_v(self.rho)

            if ite == self.max_iter - 1:
                times = time.time() - begin
                self.k = self.u / self.v
                self.k[self.k > self.max_k] = self.max_k

                print('times: {}'.format(times))

    def update_u_v(self, rho):

        D = self.D
        # compute u, v
        self.u = self.prior['u'] + (D / 2 - 1) * np.sum(rho, 0) + \
                 self.zeta * self.k * (d_besseli_low(D / 2 - 1, self.zeta * self.k))
        if np.any(np.isnan(self.u)):
            print(1)
        self.v = self.prior['v'] + (d_besseli(D / 2 - 1, self.k)) * np.sum(rho, 0) \
                 + self.prior['zeta'] * (d_besseli(D / 2 - 1, self.prior['zeta'] * self.k))
        if np.any(np.isnan(self.v)):
            print(1)

    def update_zeta_xi(self, x, rho):

        # compute zeta, xi
        temp = np.expand_dims(self.prior['zeta'] * self.prior['mu'], 0) + rho.T.dot(x)
        self.zeta = np.linalg.norm(temp, axis=1)
        if np.any(np.isnan(self.zeta)):
            print(1)
        self.xi = temp / self.zeta[:, np.newaxis]
        if np.any(np.isnan(self.xi)):
            print(1)

    def fit(self, data):

        self.init_params(data)
        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        rho = self.caclulate_rho(data)
        return np.argmax(rho, axis=1), rho

    def fit_predict(self, data):

        self.fit(data)
        return self.predict(data)


if __name__ == "__main__":

    data = scio.loadmat('./3d_data.mat')
    labels = data['z'].reshape(-1)
    data = data['data']

    vm = VMFMixture(9, 100)
    vm.fit(data)
    pred = vm.predict(data)
    score = cluster_acc(pred, labels)
    print("acc: {}".format(score[0]))
