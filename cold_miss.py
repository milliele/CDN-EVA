from util import DiscreteDist, TruncatedZipfDist
from cache import LruCache
import random
import matplotlib.pyplot as pyt
import math
from scipy.optimize import fsolve
import numpy as np
from conf import *

def numeric_cache_hit_ratio(pdf, cache, rate, measure, hit_ratio, seed=None):
    """Numerically compute the cache hit ratio of a cache under IRM
    stationary demand with a given pdf.

    Parameters
    ----------
    pdf : array-like
        The probability density function of an item being requested
    cache : Cache
        The cache object (i.e. the instance of a class subclassing
        icarus.Cache)
    warmup : int, optional
        The number of warmup requests to generate. If not specified, it is set
        to 10 times the content population
    measure : int, optional
        The number of measured requests to generate. If not specified, it is
        set to 30 times the content population
    seed : int, optional
        The seed used to generate random numbers

    Returns
    -------
    cache_hit_ratio : float
        The cache hit ratio
    """
    z = DiscreteDist(pdf, seed)
    cache_hits = 0
    t_event = 0.0
    n_interval = 0
    INTERVAL = 400 / RATE
    base = 0
    x = []
    y = []
    for _ in range(measure):
        t_event += (random.expovariate(rate))
        content = z.rv()
        now_interval = int(t_event/INTERVAL)
        if  now_interval > n_interval:
            x.append(_)
            y.append(1.0 * cache_hits / (_ - base + 1))
            base = _
            cache_hits = 0
            for i in range(n_interval+1, now_interval):
                x.append(_)
                y.append(0)
            n_interval = now_interval
        if cache.get(content):
            cache_hits += 1
        else:
            cache.put(content)
    # pyt.figure()
    # pyt.plot(x, y, label="hit rate")
    # pyt.xlabel("Number of Request")
    # pyt.ylabel("Cache Hit Ratio")
    # # pyt.hlines(hit_ratio, 0, len(y))
    # pyt.show()
    threshold = min(y[-int(len(y)*0.6):])
    print threshold
    for i, v in enumerate(y):
        if v > threshold:
            return 1.0*x[i]/RATE
    # print x[B : CUT]
    # print y[B : CUT]

def che_characteristic_time(pdf, cache_size, target=None):
    """Return the characteristic time of an item or of all items, as defined by
    Che et al.

    Parameters
    ----------
    pdf : array-like
        The probability density function of an item being requested
    cache_size : int
        The size of the cache (in number of items)
    target : int, optional
        The item index [1,N] for which characteristic time is requested. If not
        specified, the function calculates the characteristic time of all the
        items in the population.

    Returns
    -------
    r : array of float or float
        If target is None, returns an array with the characteristic times of
        all items in the population. If a target is specified, then it returns
        the characteristic time of only the specified item.
    """
    def func_r(r, i):
        return sum(math.exp(-pdf[j] * r) for j in range(len(pdf)) if j != i) \
               - len(pdf) + 1 + cache_size
    items = range(len(pdf)) if target is None else [target - 1]
    r = [fsolve(func_r, x0=cache_size, args=(i)) for i in items]
    return r if target is None else r[0]

def che_per_content_cache_hit_ratio(pdf, cache_size, target=None):
    """Estimate the cache hit ratio of an item or of all items using the Che's
    approximation.

    Parameters
    ----------
    pdf : array-like
        The probability density function of an item being requested
    cache_size : int
        The size of the cache (in number of items)
    target : int, optional
        The item index for which cache hit ratio is requested. If not
        specified, the function calculates the cache hit ratio of all the items
        in the population.

    Returns
    -------
    cache_hit_ratio : array of float or float
        If target is None, returns an array with the cache hit ratios of all
        items in the population. If a target is specified, then it returns
        the cache hit ratio of only the specified item.
    """
    items = range(len(pdf)) if target is None else [target]
    r = che_characteristic_time(pdf, cache_size)
    hit_ratio = [1 - math.exp(-pdf[i] * r[i]) for i in items]
    return hit_ratio if target is None else hit_ratio[0]

def che_cache_hit_ratio(pdf, cache_size):
    """Estimate the overall cache hit ratio of an LRU cache under generic IRM
    demand using the Che's approximation.

    Parameters
    ----------
    pdf : array-like
        The probability density function of an item being requested
    cache_size : int
        The size of the cache (in number of items)

    Returns
    -------
    cache_hit_ratio : float
        The overall cache hit ratio
    """
    ch = che_per_content_cache_hit_ratio(pdf, cache_size)
    return sum(pdf[i] * ch[i] for i in range(len(pdf)))

def che_characteristic_time_simplified(pdf, cache_size):
    """Return the characteristic time of an LRU cache under a given IRM
    workload, as defined by Che et al.
    This function computes one single characteristic time for all contents.
    This further approximation is normally accurate for workloads with
    reduced skewness in their popularity distribution.

    Parameters
    ----------
    pdf : array-like
        The probability density function of an item being requested
    cache_size : int
        The size of the cache (in number of items)

    Returns
    -------
    r : float
        The characteristic time.
    """
    def func_r(r):
        return sum(math.exp(-pdf[j] * r) for j in range(len(pdf))) \
               - len(pdf) + cache_size
    return fsolve(func_r, x0=cache_size)[0]

def che_per_content_cache_hit_ratio_simplified(pdf, cache_size, target=None):
    """Estimate the cache hit ratio of an item or of all items using the Che's
    approximation. This version uses a single characteristic time for all
    contents.

    Parameters
    ----------
    pdf : array-like
        The probability density function of an item being requested
    cache_size : int
        The size of the cache (in number of items)
    target : int, optional
        The item index for which cache hit ratio is requested. If not
        specified, the function calculates the cache hit ratio of all the items
        in the population.

    Returns
    -------
    cache_hit_ratio : array of float or float
        If target is None, returns an array with the cache hit ratios of all
        items in the population. If a target is specified, then it returns
        the cache hit ratio of only the specified item.
    """
    items = range(len(pdf)) if target is None else [target]
    r = che_characteristic_time_simplified(pdf, cache_size)
    hit_ratio = [1 - math.exp(-pdf[i] * r) for i in items]
    return hit_ratio if target is None else hit_ratio[0]

def che_cache_hit_ratio_simplified(pdf, cache_size):
    """Estimate the overall cache hit ratio of an LRU cache under generic IRM
    demand using the Che's approximation. This version uses a single
    characteristic time for all contents.

    Parameters
    ----------
    pdf : array-like
        The probability density function of an item being requested
    cache_size : int
        The size of the cache (in number of items)

    Returns
    -------
    cache_hit_ratio : float
        The overall cache hit ratio
    """
    ch = che_per_content_cache_hit_ratio_simplified(pdf, cache_size)
    return sum(pdf[i] * ch[i] for i in range(len(pdf)))

def cache_workload(rate, T, t_event = 0.0):
    INTERVAL = 2000.0 / rate
    res = []
    t_begin = t_event
    while True:
        t_event += random.expovariate(rate)
        if t_event - t_begin > T:
            break
        res.append((t_event, 1))
    n_res = len(res)
    n_interval = int(math.ceil(res[-1][0]/INTERVAL))
    # print rate, n_interval
    for i in range(n_interval):
        res.append(((i+1)*INTERVAL, 0))
    res.sort()
    return (n_res, res)

def law(crossrate, step, length = 10):
    fullset = set(range(step))
    before = set(random.sample(fullset, length))
    cross = length*crossrate
    after = set(random.sample(before, cross))
    after = after | random.sample(fullset-after, length-cross)
    return (before,after)

def newpdf(totalpdf, ):
    pass
if __name__ == '__main__':
    pass
    # pdf = TruncatedZipfDist(0.8, N_CONTENT).pdf
    #
    # y1 = []
    # y2 = []
    #
    # for ratio in CAHCE_SIZE:
    #     cache_size = N_CONTENT*ratio
    #     cache = LruCache(cache_size)
    #     theo = che_cache_hit_ratio_simplified(pdf, cache_size)
    #     print theo
    #     y1.append(theo)
    #     thres = numeric_cache_hit_ratio(pdf, cache, RATE, N_MEASURE, theo)
    #     y2.append(thres)
    #
    # pyt.figure()
    # pyt.plot(CAHCE_SIZE, y1)
    # pyt.xlabel("Cache to Population Ratio")
    # pyt.ylabel("Expected Cache Hit Ratio")
    # # pyt.hlines(hit_ratio, 0, len(y))
    # pyt.show()
    #
    # pyt.figure()
    # pyt.plot(CAHCE_SIZE, y2)
    # pyt.xlabel("Cache to Population Ratio")
    # pyt.ylabel("Time to Convergence(s)")
    # # pyt.hlines(hit_ratio, 0, len(y))
    # pyt.show()


    # pyt.figure()
    # content = [100, 500, 1000, 5000, 10000]
    # rate = np.linspace(0.1, 0.9, 9)
    # for n in content:
    #     y = []
    #     pdf = TruncatedZipfDist(0.8, n).pdf
    #     for i in rate:
    #         theo = che_cache_hit_ratio_simplified(pdf, n*i)
    #         y.append(theo)
    #     pyt.plot(rate, y, label="content = %d" %n)
    #
    # pyt.legend()
    # pyt.show()

    STEP = 20
    group = np.empty(20)
    for i in range(20):
        group[i] = np.arange(i, N_CONTENT, 20)
    print group

    pdf = TruncatedZipfDist(0.8, N_CONTENT).pdf








