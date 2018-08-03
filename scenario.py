# -*- coding: utf-8 -*-

from strategies import *
from conf import *
from util import *
import random
from mystatus import myplot
import time, collections
import numpy as np
from cache import LruCache
from cold_miss import cache_workload, che_cache_hit_ratio_simplified
import matplotlib.pyplot as pyt

class Evaluation(object):
    def conf(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        # 计算距离
        paths = parse_ashiip(FILEPATH)
        self.servers = random.sample(paths.keys(), N_SERVER)
        distance = [[paths[self.servers[u]][self.servers[v]] for v in range(N_SERVER)] for u in range(N_SERVER)]
        # 聚类
        data = whiten(distance)
        centroid = kmeans(data, N_REGION)[0]
        self.server2region = vq(data, centroid)[0]
        # 聚类结果
        self.region2server = [[] for i in range(N_REGION)]
        for u in range(N_SERVER):
            self.region2server[self.server2region[u]].append(u)
        # mapunit
        self.unit = [(region, site) for region in range(N_REGION) for site in range(N_SITE)]
        # 距离
        self.ddis = [[np.mean([distance[u][v] for u in range(N_SERVER) if self.server2region[u] == i]) for v in range(N_SERVER)] for i in range(N_REGION)]
        self.distance = {i: {j: self.ddis[self.unit[i][0]][j] for j in range(N_SERVER)} for i in range(N_UNIT)}
        # # 关于内容的偏好
        # self.dcont_prefer = [np.random.permutation(range(1, N_SITE+1)) for u in range(N_SERVER)]
        # self.cont_prefer = [[self.dcont_prefer[u][self.unit[i][1]] for i in range((N_UNIT))] for u in range(N_SERVER)]
        # 质量
        self.q = [[self.distance[i][j]*(3*random.random()) for j in range(N_SERVER)] for i in range(N_UNIT)]
        # 地区热度
        self.region_dist = np.array([len(self.region2server[region]) for region in range(N_REGION)])
        self.region_dist = self.region_dist*1.0/self.region_dist.sum()
        self.region_dist = DiscreteDist(self.region_dist, seed=seed)
        # server容量
        region_capa = [ CAPACITY*self.region_dist.pdf[region]/len(self.region2server[region]) for region in range(N_REGION)]
        self.capacity = [region_capa[self.server2region[u]] for u in range(N_SERVER)]

    def cal_load(self, policy):
        # policy = list(N_SITE)
        # for j in range(N_SERVER):
        #     if sum(policy[j]) > self.capacity[j]:
        #         print sum(policy[j]), self.capacity[j]
        loads = [100*sum(policy[j])/self.capacity[j] for j in range(N_SERVER)]
        # print loads
        return np.std(loads, ddof=1)

    # def cal_dynload(self, rate, policy):
    #     real = copy.deepcopy(policy)
    #     i2j = [ sum([ policy[j][i] for j in range(N_SERVER)]) for i in range(N_UNIT)]
    #     for j in range(N_SERVER):
    #         for i in range(N_UNIT):
    #             real[j][i] = policy[j][i]/i2j[i]*rate[i]
    #     loads = [100 * sum(real[j]) / self.capacity[j] for j in range(N_SERVER)]
    #     return max(loads)

    def cal_quality(self, policy):
        res = 0.0
        # print sum([sum(policy[j]) for j in range(N_SERVER)])
        for j in range(N_SERVER):
            for i in range(N_UNIT):
                res += self.q[i][j]*policy[j][i]
        return res

    def cal_difference(self, policy, prev_policy):
        total = sum([sum(policy[j]) for j in range(N_SERVER)])
        res = 0
        for j in range(N_SERVER):
            for i in range(N_UNIT):
                if policy[j][i]>0 and (not if_has_content(i,j, prev_policy)):
                    res += policy[j][i]/total
        return res

    def no_log_cache(self, dist, policys, pdf_label=range(N_SITE), seed=SEED):
        unit_pdf = []
        for no in range(3):
            unit_pdf.append([[policys[no][j][i] for j in range(N_SERVER)] for i in range(N_UNIT)])
            for i in range(N_UNIT):
                su = sum(unit_pdf[no][i])
                # print unit_pdf[i]
                unit_pdf[no][i] = np.array(unit_pdf[no][i]) / su
                # print unit_pdf[i]
                unit_pdf[no][i] = DiscreteDist(unit_pdf[no][i], seed)

        T = 30
        # T = N_MEASURE / RATE
        n_measure, work = cache_workload(RATE, T)
        cont_dist = TruncatedZipfDist(CONTENT_ALPHA, N_CONTENT, seed=seed)

        for t, event in work:
            site = pdf_label[dist.rv() - 1]
            region = self.region_dist.rv() - 1
            i = region * N_SITE + site
            content = cont_dist.rv() - 1 + N_CONTENT * site
            if event == 1:
                for no in range(3):
                    j = unit_pdf[no][i].rv() - 1
                    self.site_category[no][j].update([site])
                    self.category[no][j].update([content])
                    if not self.cache[no][j].get(content):
                        self.cache[no][j].put(content)

    def cal_cache(self, dist, policys, pdf_label = range(N_SITE), seed=SEED):
        unit_pdf = []
        for no in range(3):
            unit_pdf.append([ [policys[no][j][i] for j in range(N_SERVER)] for i in range(N_UNIT)])
            for i in range(N_UNIT):
                su = sum(unit_pdf[no][i])
                # print unit_pdf[i]
                unit_pdf[no][i] = np.array(unit_pdf[no][i])/su
                # print unit_pdf[i]
                unit_pdf[no][i] = DiscreteDist(unit_pdf[no][i], seed)
                # print unit_pdf[i]
                # print len(unit_pdf[i])

        T = 30
        # T = N_MEASURE / RATE
        n_measure, work = cache_workload(RATE, T, self.time_base)

        cache_hits = [0,0,0]
        count = 0
        count_base = 0
        hit_base = [0,0,0]
        hit_rate = [[],[],[]]
        cont_dist = TruncatedZipfDist(CONTENT_ALPHA, N_CONTENT, seed=seed)

        for t, event in work:
            site = pdf_label[dist.rv()-1]
            region = self.region_dist.rv()-1
            i = region*N_SITE + site
            content = cont_dist.rv() - 1 + N_CONTENT * site
            if event==0 and count-count_base > 0:
                for no in range(3):
                    hit_rate[no].append((t, 1.0 * (cache_hits[no] - hit_base[no]) / (count - count_base)))
                    hit_base[no] = cache_hits[no]
                # print count, count_base
                count_base = count
                self.time_base = t
            elif event == 1:
                count += 1
                for no in range(3):
                    j = unit_pdf[no][i].rv() - 1
                    if self.cache[no][j].get(content):
                        cache_hits[no] += 1
                    else:
                        self.cache[no][j].put(content)
        return count-np.array(cache_hits), hit_rate

    def divide_flow(self, rates):
        rate0 = max(rates.values()) / 16
        # 分流
        sorted_rate = collections.OrderedDict(sorted(rates.items(), key=lambda x: (x[0] / N_REGION, -x[1])))
        flows = []
        for region in range(N_REGION):
            flow = {}
            count = 0
            for no in range(N_SITE):
                loc = region * N_SITE + no
                i = sorted_rate.items()[loc][0]
                rest = rates[i]
                while count + rest > rate0:
                    flow[i] = rate0 - count
                    rest -= rate0 - count
                    count = 0
                    flows.append(flow)
                    flow = {}
                if rest > 0:
                    flow[i] = rest
                    count += rest
            if len(flow):
                flows.append(flow)
        # 计算q
        n_flow = len(flows)
        q = []
        # [[self.distance[i][j] * (3 * random.random()) for j in range(N_SERVER)] for i in range(N_UNIT)]
        for f, flow in enumerate(flows):
            load = sum(flow.values())
            q.append([0] * N_SERVER)
            for i in flow:
                for u in range(N_SERVER):
                    q[-1][u] += flow[i] * self.q[i][u] / load
        capacity = np.floor(1.0 * np.array(self.capacity) / rate0)
        return (rate0, flows, q, capacity)

    def static(self):
        print "[%s] Begin Static" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.dyndns = DYNDNS()
        self.marriage = MARRIAGE()
        self.results = Tree()

        for ratio in RATE_RATIO:
            self.results['STATIC']['RATIO']['Q']['DYNDNS'][ratio] = []
            self.results['STATIC']['RATIO']['Q']['AKAMAI'][ratio] = []
            self.results['STATIC']['RATIO']['LOAD']['DYNDNS'][ratio] = []
            self.results['STATIC']['RATIO']['LOAD']['AKAMAI'][ratio] = []

        for alpha in POPS:
            self.results['STATIC']['POP']['Q']['DYNDNS'][alpha] = []
            self.results['STATIC']['POP']['Q']['AKAMAI'][alpha] = []
            self.results['STATIC']['POP']['LOAD']['DYNDNS'][alpha] = []
            self.results['STATIC']['POP']['LOAD']['AKAMAI'][alpha] = []

        for _ in range(N_TIME):
            self.conf(SEED)

            # change traffic load
            print "[%s] Static Round %d,Change traffic load" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _)
            dist = TruncatedZipfDist(SITE_ALPHA, N_SITE, seed=SEED)
            for ratio in RATE_RATIO:
                totalrate = ratio*RATE
                rates = {}
                for i in range(N_UNIT):
                    region, site = self.unit[i]
                    rates[i] = dist.pdf[site]*self.region_dist.pdf[region]*totalrate
                rate0, flows, q, capacity = self.divide_flow(rates)

                res_dyn = [[ 0 for i in range(N_UNIT)]for j in range(N_SERVER)]
                # res_marriage = [[ 0 for i in range(N_UNIT)]for j in range(N_SERVER)]

                print "[%s] Static Round %d,Caculating DYNDNS: load=%s" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _, str(ratio))
                res_dyn = self.dyndns.calculation(q, capacity, flows, rate0, res_dyn)
                # print res_dyn
                print "[%s] Static Round %d,Caculating MARRIAGE: load=%s" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _, str(ratio))
                res_marriage = self.marriage.calculation(q, capacity, flows)
                # print res_marriage
                print "[%s] Static Round %d,Caculating quality" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _)
                self.results['STATIC']['RATIO']['Q']['DYNDNS'][ratio].append(self.cal_quality(res_dyn))
                self.results['STATIC']['RATIO']['Q']['AKAMAI'][ratio].append(self.cal_quality(res_marriage))
                print "[%s] Static Round %d,Caculating load" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _)
                self.results['STATIC']['RATIO']['LOAD']['DYNDNS'][ratio].append(self.cal_load(res_dyn))
                self.results['STATIC']['RATIO']['LOAD']['AKAMAI'][ratio].append(self.cal_load(res_marriage))
                np.save("static", np.array(self.results.dict()))

            # # change POP
            # print "[%s] Static Round %d,Change popularity" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _)
            # for alpha in POPS:
            #     rates = {}
            #     for i in range(N_UNIT):
            #         region, site = self.unit[i]
            #         rates[i] = dist.pdf[site] * self.region_dist.pdf[region] * RATE
            #     rate0, flows, q, capacity = self.divide_flow(rates)
            #
            #     res_dyn = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            #     print "[%s] Static Round %d,Caculating DYNDNS: alpha=%s" % (
            #         time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _, str(alpha))
            #     res_dyn = self.dyndns.calculation(q, capacity, flows, rate0, res_dyn)
            #     print "[%s] Static Round %d,Caculating MARRIAGE: alpha=%s" % (
            #         time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _, str(alpha))
            #     res_marriage = self.marriage.calculation(q, capacity, flows)
            #     print "[%s] Static Round %d,Caculating quality" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _)
            #     self.results['STATIC']['POP']['Q']['DYNDNS'][alpha].append(self.cal_quality(res_dyn))
            #     self.results['STATIC']['POP']['Q']['AKAMAI'][alpha].append(self.cal_quality(res_marriage))
            #     print "[%s] Static Round %d,Caculating load" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), _)
            #     self.results['STATIC']['POP']['LOAD']['DYNDNS'][alpha].append(self.cal_load(res_dyn))
            #     self.results['STATIC']['POP']['LOAD']['AKAMAI'][alpha].append(self.cal_load(res_marriage))
            #     np.save("static", np.array(self.results.dict()))

    def dynamic1(self):
        print "[%s] Begin Dynamic Scenario 1" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.conf(SEED+1 if SEED else SEED)

        self.dyndns = DYNDNS()
        self.marriage = MARRIAGE()
        self.nondyn = NONDYNDNS()
        self.results = Tree()

        print "[%s] Begin Set Capacity" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.category = [[set() for j in range(N_SERVER)] for _ in range(3)]
        self.site_category = [[set() for j in range(N_SERVER)] for _ in range(3)]
        self.cache = [[LruCache(CACHE_SIZE) for j in range(N_SERVER)] for _ in range(3)]
        dist = TruncatedZipfDist(SITE_ALPHA, N_SITE, seed=SEED + 1 if SEED else SEED)
        # random.shuffle(pdf_label)
        # print pdf_label
        rates = {}
        for i in range(N_UNIT):
            region, site = self.unit[i]
            rates[i] = dist.pdf[site] * self.region_dist.pdf[region] * RATE
            # print rates[i]
        rate0, flows, q, capacity = self.divide_flow(rates)

        prev_dyn = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
        prev_none = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
        prev_dyn = self.dyndns.calculation(q, capacity, flows, rate0, prev_dyn)
        prev_akamai = self.marriage.calculation(q, capacity, flows)
        prev_none = self.nondyn.calculation(q, capacity, flows, rate0, prev_none)
        self.no_log_cache(dist, [prev_dyn, prev_akamai, prev_none])
        cache_sizes = [int(np.mean([len(self.category[0][j]),len(self.category[1][j]), len(self.category[2][j])])) for j in range(N_SERVER)]

        for ratio in POP_INTERVAL:
            interval = int(N_SITE*ratio)
            n_interval = int(1.0/ratio)

            self.results['DYNAMIC1']['Q']['DYNDNS'][ratio]=[]
            self.results['DYNAMIC1']['Q']['AKAMAI'][ratio]=[]
            self.results['DYNAMIC1']['Q']['NONE'][ratio]=[]
            self.results['DYNAMIC1']['DIFFER']['DYNDNS'][ratio]=[]
            self.results['DYNAMIC1']['DIFFER']['AKAMAI'][ratio]=[]
            self.results['DYNAMIC1']['DIFFER']['NONE'][ratio]=[]
            self.results['DYNAMIC1']['MISS']['DYNDNS'][ratio] = []
            self.results['DYNAMIC1']['MISS']['AKAMAI'][ratio] = []
            self.results['DYNAMIC1']['MISS']['NONE'][ratio] = []
            self.results['DYNAMIC1']['HIT']['DYNDNS'][ratio] = []
            self.results['DYNAMIC1']['HIT']['AKAMAI'][ratio] = []
            self.results['DYNAMIC1']['HIT']['NONE'][ratio] = []

            print "[%s] Dynamic ratio=%s: Warm Up" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ratio)
            dist = TruncatedZipfDist(SITE_ALPHA, N_SITE, seed=SEED+1 if SEED else SEED)
            self.cache = [[LruCache(cache_sizes[j]) for j in range(N_SERVER)] for _ in range(3)]
            # print self.cache[0][0].maxlen
            prev_dyn = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            prev_none = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            prev_akamai = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            pdf_label = range(N_SITE)
            # print pdf_label
            for _ in range(N_WARMUP):
                tmp_label = []
                for ino in range(n_interval):
                    tmp = pdf_label[ino*interval:(ino+1)*interval]
                    random.shuffle(tmp)
                    tmp_label += tmp
                pdf_label = tmp_label
                # print pdf_label
                rates = {}
                for i in range(N_UNIT):
                    region, site = self.unit[i]
                    # print site, pdf_label[site], region
                    rates[i] = dist.pdf[pdf_label[site]] * self.region_dist.pdf[region] * RATE
                    # print rates[i]
                rate0, flows, q, capacity = self.divide_flow(rates)

                prev_dyn = self.dyndns.calculation(q, capacity, flows, rate0, prev_dyn)
                prev_akamai = self.marriage.calculation(q, capacity, flows)
                prev_none = self.nondyn.calculation(q, capacity, flows, rate0, prev_none)
                self.no_log_cache(dist, [prev_dyn, prev_akamai, prev_none], pdf_label)

            self.time_base = 0.0

            for no in range(N_TIME):
                print "[%s] Dynamic ratio=%s: Round %d" % (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ratio, no+1)
                tmp_label = []
                for ino in range(n_interval):
                    tmp = pdf_label[ino * interval:(ino + 1) * interval]
                    random.shuffle(tmp)
                    tmp_label += tmp
                pdf_label = tmp_label
                rates = {}
                for i in range(N_UNIT):
                    region, site = self.unit[i]
                    rates[i] = dist.pdf[pdf_label[site]] * self.region_dist.pdf[region] * RATE
                rate0, flows, q, capacity = self.divide_flow(rates)

                print "[%s] Dynamic ratio=%s: Round %d Caculating DYNDNS" % (
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ratio, no + 1)
                res_dyn = self.dyndns.calculation(q, capacity, flows, rate0, prev_dyn)
                print "[%s] Dynamic ratio=%s: Round %d Caculating AKAMAI" % (
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ratio, no + 1)
                res_akamai = self.marriage.calculation(q, capacity, flows)
                print "[%s] Dynamic ratio=%s: Round %d Caculating NONE" % (
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ratio, no + 1)
                res_none = self.nondyn.calculation(q, capacity, flows, rate0, prev_none)

                print "[%s] Dynamic ratio=%s: Round %d Caculating quality" % (
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ratio, no + 1)
                self.results['DYNAMIC1']['Q']['DYNDNS'][ratio].append(self.cal_quality(res_dyn))
                self.results['DYNAMIC1']['Q']['AKAMAI'][ratio].append(self.cal_quality(res_akamai))
                self.results['DYNAMIC1']['Q']['NONE'][ratio].append(self.cal_quality(res_none))
                # print "[%s] Caculating load" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                # self.results['DYNAMIC'][no]['DYNDNS']['LOAD'] = self.cal_load(res_dyn)
                # self.results['DYNAMIC'][no]['AKAMAI']['LOAD'] = self.cal_load(res_akamai)
                # self.results['DYNAMIC'][no]['NONE']['LOAD'] = self.cal_load(res_none)
                print "[%s] Dynamic ratio=%s: Round %d Caculating difference" % (
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ratio, no + 1)
                self.results['DYNAMIC1']['DIFFER']['DYNDNS'][ratio].append(self.cal_difference(res_dyn, prev_dyn))
                self.results['DYNAMIC1']['DIFFER']['AKAMAI'][ratio].append(self.cal_difference(res_akamai, prev_akamai))
                self.results['DYNAMIC1']['DIFFER']['NONE'][ratio].append(self.cal_difference(res_none, prev_none))

                print "[%s] Dynamic ratio=%s: Round %d Caculating cache" % (
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ratio, no + 1)
                miss, hitrates = self.cal_cache(dist,[res_dyn, res_akamai, res_none], pdf_label)
                # self.results['DYNAMIC'][no]['DYNDNS']['HIT'] = hit
                # self.results['DYNAMIC'][no]['DYNDNS']['MISS'] = miss
                self.results['DYNAMIC1']['MISS']['DYNDNS'][ratio].append(miss[0])
                self.results['DYNAMIC1']['HIT']['DYNDNS'][ratio] += hitrates[0]
                # self.results['DYNAMIC'][no]['AKAMAI']['HIT'] = hit
                # self.results['DYNAMIC'][no]['AKAMAI']['MISS'] = miss
                self.results['DYNAMIC1']['MISS']['AKAMAI'][ratio].append(miss[1])
                self.results['DYNAMIC1']['HIT']['AKAMAI'][ratio] += hitrates[1]
                # self.results['DYNAMIC'][no]['NONE']['HIT'] = hit
                # self.results['DYNAMIC'][no]['NONE']['MISS'] = miss
                self.results['DYNAMIC1']['MISS']['NONE'][ratio].append(miss[2])
                self.results['DYNAMIC1']['HIT']['NONE'][ratio] += hitrates[2]
                prev_none = res_none
                prev_dyn = res_dyn
                prev_akamai = res_akamai
                np.save("dynamic-1", np.array(self.results.dict()))

    def dynamic2(self):
        self.dyndns = DYNDNS()
        self.marriage = MARRIAGE()
        self.nondyn = NONDYNDNS()
        self.results = Tree()

        # print "[%s] Round %d Begin Dynamic Scenario 2" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no)
        self.results['DYNAMIC2']['Q']['DYNDNS'] = []
        self.results['DYNAMIC2']['Q']['AKAMAI'] = []
        self.results['DYNAMIC2']['Q']['NONE'] = []
        self.results['DYNAMIC2']['DIFFER']['DYNDNS'] = []
        self.results['DYNAMIC2']['DIFFER']['AKAMAI'] = []
        self.results['DYNAMIC2']['DIFFER']['NONE'] = []
        self.results['DYNAMIC2']['MISS']['DYNDNS'] = []
        self.results['DYNAMIC2']['MISS']['AKAMAI'] = []
        self.results['DYNAMIC2']['MISS']['NONE'] = []
        self.results['DYNAMIC2']['HIT']['DYNDNS'] = []
        self.results['DYNAMIC2']['HIT']['AKAMAI'] = []
        self.results['DYNAMIC2']['HIT']['NONE'] = []

        for no in range(N_TIME):
            self.conf(SEED + 2 if SEED else SEED)

            print "[%s] Round %d Begin Set Capacity" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no+1)
            self.category = [[set() for j in range(N_SERVER)] for _ in range(3)]
            self.site_category = [[set() for j in range(N_SERVER)] for _ in range(3)]
            self.cache = [[LruCache(CACHE_SIZE) for j in range(N_SERVER)] for _ in range(3)]
            dist = TruncatedZipfDist(SITE_ALPHA, N_SITE, seed=SEED + 2 if SEED else SEED)
            # random.shuffle(pdf_label)
            # print pdf_label
            rates = {}
            for i in range(N_UNIT):
                region, site = self.unit[i]
                rates[i] = dist.pdf[site] * self.region_dist.pdf[region] * RATE
                # print rates[i]
            rate0, flows, q, capacity = self.divide_flow(rates)

            prev_dyn = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            prev_none = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            prev_dyn = self.dyndns.calculation(q, capacity, flows, rate0, prev_dyn)
            prev_akamai = self.marriage.calculation(q, capacity, flows)
            prev_none = self.nondyn.calculation(q, capacity, flows, rate0, prev_none)
            self.no_log_cache(dist, [prev_dyn, prev_akamai, prev_none])
            cache_sizes = [int(np.mean([len(self.category[0][j]), len(self.category[1][j]), len(self.category[2][j])])) for
                           j in range(N_SERVER)]

            print "[%s] Dynamic 2: Round %d: Warm Up" % (
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no+1)
            dist = TruncatedZipfDist(SITE_ALPHA, N_SITE, seed=SEED + 2 if SEED else SEED)
            self.cache = [[LruCache(cache_sizes[j]) for j in range(N_SERVER)] for _ in range(3)]
            # print self.cache[0][0].maxlen
            prev_dyn = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            prev_none = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            prev_akamai = [[0 for i in range(N_UNIT)] for j in range(N_SERVER)]
            rates = {}
            for i in range(N_UNIT):
                region, site = self.unit[i]
                rates[i] = dist.pdf[site] * self.region_dist.pdf[region] * RATE
                # print rates[i]
            rate0, flows, q, capacity = self.divide_flow(rates)

            prev_dyn = self.dyndns.calculation(q, capacity, flows, rate0, prev_dyn)
            prev_akamai = self.marriage.calculation(q, capacity, flows)
            prev_none = self.nondyn.calculation(q, capacity, flows, rate0, prev_none)
            self.no_log_cache(dist, [prev_dyn, prev_akamai, prev_none])

            self.time_base = 0.0
            pdf = dist.pdf*RATE
            random.seed(None)
            incre_site = random.choice(range(int(N_SITE*FORMER)))
            increase = pdf[incre_site]*(BAI-1)
            others = np.delete(copy.deepcopy(pdf), incre_site, axis=0)
            others /= others.sum()
            for si in range(N_SITE):
                if si < incre_site:
                    pdf[si] -= others[si]*increase
                elif si == incre_site:
                    pdf[si] *= BAI
                else:
                    pdf[si] -= others[si-1]*increase
            pdf = pdf/pdf.sum()
            pdf = list(sorted(pdf,reverse=True))
            dist = DiscreteDist(pdf, seed=SEED+2 if SEED else SEED)
            rates = {}
            for i in range(N_UNIT):
                region, site = self.unit[i]
                rates[i] = dist.pdf[site] * self.region_dist.pdf[region] * RATE
            rate0, flows, q, capacity = self.divide_flow(rates)

            print "[%s] Dynamic 2: Round %d Caculating DYNDNS" % (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no + 1)
            res_dyn = self.dyndns.calculation(q, capacity, flows, rate0, prev_dyn)
            print "[%s] Dynamic 2: Round %d Caculating AKAMAI" % (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no + 1)
            res_akamai = self.marriage.calculation(q, capacity, flows)
            print "[%s] Dynamic 2: Round %d Caculating NONE" % (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no + 1)
            res_none = self.nondyn.calculation(q, capacity, flows, rate0, prev_none)

            print "[%s] Dynamic 2: Round %d Caculating quality" % (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no + 1)
            self.results['DYNAMIC2']['Q']['DYNDNS'].append(self.cal_quality(res_dyn))
            self.results['DYNAMIC2']['Q']['AKAMAI'].append(self.cal_quality(res_akamai))
            self.results['DYNAMIC2']['Q']['NONE'].append(self.cal_quality(res_none))
            # print "[%s] Caculating load" % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            # self.results['DYNAMIC'][no]['DYNDNS']['LOAD'] = self.cal_load(res_dyn)
            # self.results['DYNAMIC'][no]['AKAMAI']['LOAD'] = self.cal_load(res_akamai)
            # self.results['DYNAMIC'][no]['NONE']['LOAD'] = self.cal_load(res_none)
            print "[%s] Dynamic 2: Round %d Caculating difference" % (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no + 1)
            self.results['DYNAMIC2']['DIFFER']['DYNDNS'].append(self.cal_difference(res_dyn, prev_dyn))
            self.results['DYNAMIC2']['DIFFER']['AKAMAI'].append(self.cal_difference(res_akamai, prev_akamai))
            self.results['DYNAMIC2']['DIFFER']['NONE'].append(self.cal_difference(res_none, prev_none))

            print "[%s] Dynamic 2: Round %d Caculating cache" % (
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), no + 1)
            miss, hitrates = self.cal_cache(dist, [res_dyn, res_akamai, res_none])
            # self.results['DYNAMIC'][no]['DYNDNS']['HIT'] = hit
            # self.results['DYNAMIC'][no]['DYNDNS']['MISS'] = miss
            self.results['DYNAMIC2']['MISS']['DYNDNS'].append(miss[0])
            self.results['DYNAMIC2']['HIT']['DYNDNS'] += hitrates[0]
            # self.results['DYNAMIC'][no]['AKAMAI']['HIT'] = hit
            # self.results['DYNAMIC'][no]['AKAMAI']['MISS'] = miss
            self.results['DYNAMIC2']['MISS']['AKAMAI'].append(miss[1])
            self.results['DYNAMIC2']['HIT']['AKAMAI'] += hitrates[1]
            # self.results['DYNAMIC'][no]['NONE']['HIT'] = hit
            # self.results['DYNAMIC'][no]['NONE']['MISS'] = miss
            self.results['DYNAMIC2']['MISS']['NONE'].append(miss[2])
            self.results['DYNAMIC2']['HIT']['NONE'] += hitrates[2]
            np.save("dynamic-2", np.array(self.results.dict()))

if __name__ == '__main__':
    eva = Evaluation()
    # eva.static()
    # eva.dynamic1()
    eva.dynamic2()
    myplot('plot')

