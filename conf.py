# -*- coding: utf-8 -*-
from util import parse_ashiip, TruncatedZipfDist
import random, math
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import numpy as np

# 基本参数
N_SITE = 100
N_SERVER = 20
N_REGION = 5
N_UNIT = N_REGION*N_SITE

SEED = None

# 拓扑路径
FILEPATH = 'out1.txt'

CAPACITY= 4200
RATE = CAPACITY *0.9
RATE_RATIO = [0.5, 0.55, 0.6, 0.65 ,0.7, 0.75 ,0.8, 0.85 ,0.9]
# RATE_RATIO = [1]
POPS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
POP_INTERVAL = [0.05, 0.1, 0.2, 0.5, 1.0]
# POP_INTERVAL = [ 0.4, 0.5, 1.0]

FORMER = 0.1

# 命中率测试
N_MEASURE = 1000000
N_CONTENT = 100
N_FLOW = 50
CACHE_SIZE = N_CONTENT*N_FLOW

OMEGA = [1, 1, 30]
LOAD_OMEGA = [0,0.5,0]

# dynamic测试场景
N_WARMUP = 10
N_TIME = 100

# LOAD_WARMUP = 0
# LOAD_TIME = 10

# Zipf们的参数
CONTENT_ALPHA = 1.5
SITE_ALPHA = 0.8
REGION_ALPHA = 0.5

BAI = 1.5

if __name__ == '__main__':
    RATE_RATIO = np.linspace(0.5, 1.0, 6)