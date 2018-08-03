# -*- coding:utf-8 -*-

import math, collections
import numpy as np
from conf import *
import random
import copy

class DYNDNS(object):
    def if_has_content(self, f, j):
        res = 0.0
        load = sum(self.flows[f].values())
        for i, rate in self.flows[f].items():
            site = i % N_SITE
            for region in range(N_REGION):
                ni = region*N_SITE + site
                if self.previous[j][ni]:
                    res += rate/load
                    break
        return res

    def gain(self, loads, i, j):
        qual = self.rate0*self.q[i][j]
        bal = self.capacity[j]-loads[j]
        differ = self.if_has_content(i,j)
        # print qual, bal, differ
        res = OMEGA[0]*qual + OMEGA[1]*bal + OMEGA[2]*differ
        return res

    def calculation(self, q, capacity, flows, rate0, previous):
        # print sum(rate.values())
        # print rate
        # self.rate0 = max(rate.values())/16
        # # 分流
        # sorted_rate = collections.OrderedDict(sorted(rate.items(), key=lambda x:(x[0]/N_REGION, -x[1])))
        # self.flows = []
        # for region in range(N_REGION):
        #     flow = {}
        #     count = 0
        #     for no in range(N_SITE):
        #         loc = region*N_SITE + no
        #         i = sorted_rate.items()[loc][0]
        #         rest = rate[i]
        #         while count + rest > self.rate0:
        #             flow[i] = self.rate0 - count
        #             rest -= self.rate0-count
        #             count = 0
        #             self.flows.append(flow)
        #             flow = {}
        #         if rest>0:
        #             flow[i] = rest
        #             count += rest
        #     if len(flow):
        #         self.flows.append(flow)
        # 计算q
        self.q = q
        self.capacity = capacity
        self.flows = flows
        self.rate0 = rate0
        self.previous = previous

        n_flow = len(self.flows)
        # self.q = []
        # for f, flow in enumerate(self.flows):
        #     load = sum(flow.values())
        #     self.q.append([0]*N_SERVER)
        #     for i in flow:
        #         for u in range(N_SERVER):
        #             self.q[-1][u] += flow[i]*q[i][u]/load
        # self.capacity = np.floor(1.0* np.array(capacity)/self.rate0)

        if n_flow  > self.capacity.sum():
            print n_flow, self.capacity.sum()

        E = {i: range(N_SERVER) for i in range(n_flow)}
        loads = collections.defaultdict(int)
        d = [-1] * n_flow

        while len(E)>0:
            maxgain = 0
            dicision = -1
            for i in E:
                for j in E[i]:
                    tmp = self.gain(loads, i, j)
                    if tmp > maxgain:
                        maxgain = tmp
                        dicision = (i,j)
            if maxgain == 0:
                break
            else:
                ii, jj = dicision
                d[ii] = jj
                E.pop(ii)
                loads[jj] += 1
                if loads[jj]>=self.capacity[jj]:
                    for i in E:
                        E[i].remove(jj)

        solution = [[ 0 for i in range(N_UNIT)]for j in range(N_SERVER)]
        for f in range(n_flow):
            j = d[f]
            if j==-1:
                continue
            for i, rate in self.flows[f].items():
                solution[j][i] += rate
        return copy.deepcopy(solution)

class MARRIAGE(object):

    def calculation(self, q, capacity, flows):
        # self.rate0 = max(rates.values()) / 16
        # # 分流
        # sorted_rate = collections.OrderedDict(sorted(rates.items(), key=lambda x: (x[0] / N_REGION, -x[1])))
        # self.flows = []
        # for region in range(N_REGION):
        #     flow = {}
        #     count = 0
        #     for no in range(N_SITE):
        #         loc = region * N_SITE + no
        #         i = sorted_rate.items()[loc][0]
        #         rest = rates[i]
        #         while count + rest > self.rate0:
        #             flow[i] = self.rate0 - count
        #             rest -= self.rate0 - count
        #             count = 0
        #             self.flows.append(flow)
        #             flow = {}
        #         if rest > 0:
        #             flow[i] = rest
        #             count += rest
        #     if len(flow):
        #         self.flows.append(flow)
        # # 计算q
        # n_flow = len(self.flows)
        # self.q = []
        # for f, flow in enumerate(self.flows):
        #     load = sum(flow.values())
        #     self.q.append([0] * N_SERVER)
        #     for i in flow:
        #         for u in range(N_SERVER):
        #             self.q[-1][u] += flow[i] * q[i][u] / load
        # self.capacity = np.floor(1.0 * np.array(capacity) / self.rate0)

        n_flow = len(flows)

        m2w = {i: collections.deque(list(sorted(range(N_SERVER), key=lambda x: q[i][x], reverse=True))) for i in range(n_flow)}
        w2m = {j: list(sorted(range(n_flow), key=lambda x: q[x][j], reverse=True)) for j in range(N_SERVER)}

        # E = {i: rates[i/16]/16 for i in range(n_flow)}
        #
        # loads = [[ 0 for i in range(n_flow)]for j in range(N_SERVER)]
        #
        # while len(E)>0:
        #     man, rate = E.popitem()
        #     if len(m2w[man]):
        #         j = m2w[man].popleft()
        #     else:
        #         continue
        #
        #     loads[j][man] += rate
        #     cha = sum(loads[j]) - self.capacity[j]
        #     while cha > 0:
        #         drop = max(range(n_flow), key= lambda x: w2m[j].index(x) if loads[j][x]>0 else -1)
        #         tmp = max(min(cha, loads[j][drop]), rates[drop/16]/16)
        #         loads[j][drop] -= tmp
        #         if drop in E:
        #             E[drop] += tmp
        #         else:
        #             E[drop] = tmp
        #         cha -= tmp
        # self.loads = [[sum(loads[j][i * 16:(i + 1) * 16]) for i in range(N_UNIT)] for j in range(N_SERVER)]

        E = {i: 1 for i in range(n_flow)}

        loads = [[0 for i in range(n_flow)] for j in range(N_SERVER)]

        while len(E) > 0:
            man, rate = E.popitem()
            if len(m2w[man]):
                j = m2w[man].popleft()
            else:
                continue

            loads[j][man] += 1
            if sum(loads[j])>capacity[j]:
                drop = max(range(n_flow), key=lambda x: w2m[j].index(x) if loads[j][x] > 0 else -1)
                loads[j][drop] -= 1
                if drop in E:
                    E[drop] += 1
                else:
                    E[drop] = 1

        solution = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
        for j in range(N_SERVER):
            for f, flow in enumerate(flows):
                if loads[j][f]==0:
                    continue
                for i, rate in flow.items():
                    solution[j][i] += rate
        return copy.deepcopy(solution)

class NONDYNDNS(object):
    def if_has_content(self, f, j):
        res = 0.0
        load = sum(self.flows[f].values())
        for i, rate in self.flows[f].items():
            site = i % N_SITE
            for region in range(N_REGION):
                ni = region*N_SITE + site
                if self.previous[j][ni]:
                    res += rate/load
                    break
        return res

    def gain(self, loads, i, j):
        qual = self.rate0*self.q[i][j]
        bal = self.capacity[j]-loads[j]
        res = OMEGA[0]*qual + OMEGA[1]*bal
        return res

    def calculation(self, q, capacity, flows, rate0, previous):
        # print sum(rate.values())
        # print rate
        # self.rate0 = max(rate.values())/16
        # # 分流
        # sorted_rate = collections.OrderedDict(sorted(rate.items(), key=lambda x:(x[0]/N_REGION, -x[1])))
        # self.flows = []
        # for region in range(N_REGION):
        #     flow = {}
        #     count = 0
        #     for no in range(N_SITE):
        #         loc = region*N_SITE + no
        #         i = sorted_rate.items()[loc][0]
        #         rest = rate[i]
        #         while count + rest > self.rate0:
        #             flow[i] = self.rate0 - count
        #             rest -= self.rate0-count
        #             count = 0
        #             self.flows.append(flow)
        #             flow = {}
        #         if rest>0:
        #             flow[i] = rest
        #             count += rest
        #     if len(flow):
        #         self.flows.append(flow)
        # 计算q
        self.q = q
        self.capacity = capacity
        self.flows = flows
        self.rate0 = rate0
        self.previous = previous

        n_flow = len(self.flows)
        # self.q = []
        # for f, flow in enumerate(self.flows):
        #     load = sum(flow.values())
        #     self.q.append([0]*N_SERVER)
        #     for i in flow:
        #         for u in range(N_SERVER):
        #             self.q[-1][u] += flow[i]*q[i][u]/load
        # self.capacity = np.floor(1.0* np.array(capacity)/self.rate0)

        if n_flow  > self.capacity.sum():
            print n_flow, self.capacity.sum()

        E = {i: range(N_SERVER) for i in range(n_flow)}
        loads = collections.defaultdict(int)
        d = [-1] * n_flow

        while len(E)>0:
            maxgain = 0
            dicision = -1
            for i in E:
                for j in E[i]:
                    tmp = self.gain(loads, i, j)
                    if tmp > maxgain:
                        maxgain = tmp
                        dicision = (i,j)
            if maxgain == 0:
                break
            else:
                ii, jj = dicision
                d[ii] = jj
                E.pop(ii)
                loads[jj] += 1
                if loads[jj]>=self.capacity[jj]:
                    for i in E:
                        E[i].remove(jj)

        solution = [[ 0 for i in range(N_UNIT)]for j in range(N_SERVER)]
        for f in range(n_flow):
            j = d[f]
            if j==-1:
                continue
            for i, rate in self.flows[f].items():
                solution[j][i] += rate
        return copy.deepcopy(solution)

def if_has_content(i, j, loads):
    site = i % N_SITE
    for region in range(N_REGION):
        ni = region*N_SITE + site
        if loads[j][ni]:
            return True
    return False


if __name__ == '__main__':
    prior = [np.random.permutation(range(N_SITE))] * N_SERVER
    print prior