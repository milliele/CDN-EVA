# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as pyt
import os.path
import numpy as np
import collections
from util import cdf, confidence_interval
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from conf import *

METRIC = [
    'LOAD',
    'Q',
    'DIFFER',
    'MISS',
    'HIT',
    ]
METRIC_LABEL = {
    'LOAD':      'Standard Derivation of Load',
    'Q': 'Quality per Request',
    'MISS': 'Number of Misses',
    'HIT' : 'Cache Hit Ratio (%)',
    'DIFFER': 'Fraction of Inconsistency (%)',
    'HITS': 'Cache Hit Ratio (%)',
}

METRIC_TITLE = {
    'LOAD':      'Load blancing',
    'Q': 'Quality',
    'MISS': 'Cache Miss',
    'HIT' : 'Real-time Cache Hit Ratio',
    'DIFFER': 'Persistency',
    'HITS': 'Cache Hit Ratio',
}


STRATEGY = [
    'AKAMAI',
    'DYNDNS',
    'NONE'
]
STRATEGY_LABEL = {
    'DYNDNS':       'DynDNS',
    'AKAMAI':     'Stable Matching',
    'NONE':          'NonDynDNS'
}

STYLE = {
    'DYNDNS':   'r-o',
    'AKAMAI':   'c-^',
    'NONE':     'y-s'
}

BORDER = {
    'AKAMAI': (88, 80),
    'DYNDNS': (90, 87),
    'NONE': (89, 80)
}

pyt.rcParams['font.size'] = 20
pyt.rcParams['lines.markersize'] = 10
pyt.rcParams['lines.linewidth'] = 4
pyt.rcParams['lines.markeredgewidth'] = 1.5
pyt.rcParams['figure.subplot.wspace'] = 0.25
pyt.rcParams['figure.subplot.top'] = 0.8

def static(plotdir):
    import ast
    # static
    results = ast.literal_eval(str(np.load("static.npy")))
    # print results
    fig = pyt.figure(figsize=(16, 6))
    for j, metric in enumerate(METRIC[:-3]):
        ax = pyt.subplot(1,2,j+1)
        res = results['STATIC']['RATIO']
        for i, method in enumerate(STRATEGY[:-1]):
            all_data = res[metric][method]
            x = list(sorted(all_data.keys()))
            y = []
            # yerr = [[], []]
            for k in x:
                data = all_data[k]
                if metric == 'Q':
                    data = np.array(data) / RATE/k
                # be, en = confidence_interval(data, 0.05, 0.95)
                mean = np.mean(data)
                y.append(mean)
                # yerr[0].append(mean - be)
                # yerr[1].append(en - mean)
            # pyt.errorbar(np.array(x) * 100, y, yerr=yerr, fmt=STYLE[method], label=STRATEGY_LABEL[method])
            pyt.plot(np.array(x) * 100, y, STYLE[method], label=STRATEGY_LABEL[method])
        if metric == 'Q':
            pyt.ylim(13, 17)
        pyt.xlabel('Demand to Capacity Ratio(%)')
        pyt.ylabel(METRIC_LABEL[metric])
        if j == 0:
            pyt.legend(loc="upper left", ncol=4, bbox_to_anchor=(0,1.2, 2.2, 0.05), mode='expand')
    # pyt.show()
    pyt.savefig(os.path.join(plotdir, "static-1.pdf"), bbox_inches='tight')
    pyt.savefig(os.path.join(plotdir, "static-1.png"), bbox_inches='tight')
    pyt.close()

    # fig = pyt.figure(figsize=(16, 6))
    # for j, metric in enumerate(METRIC[:-3]):
    #     ax = pyt.subplot(1,2,j+1)
    #     res = results['STATIC']['POP']
    #     for i, method in enumerate(STRATEGY[:-1]):
    #         all_data = res[metric][method]
    #         x = list(sorted(all_data.keys()))
    #         y = []
    #         # yerr = [[], []]
    #         for k in x:
    #             data = all_data[k]
    #             if metric == 'Q':
    #                 data = np.array(data) / 3500.0
    #             # be, en = confidence_interval(data, 0.05, 0.95)
    #             mean = np.mean(data)
    #             y.append(mean)
    #             # yerr[0].append(mean - be)
    #             # yerr[1].append(en - mean)
    #         # pyt.errorbar(np.array(x) * 100, y, yerr=yerr, fmt=STYLE[method], label=STRATEGY_LABEL[method])
    #         pyt.plot(np.array(x) * 100, y, STYLE[method], label=STRATEGY_LABEL[method])
    #     if j ==0:
    #         pyt.ylim(15,25)
    #     elif j==1:
    #         pyt.ylim(14.5, 15.5)
    #     pyt.xlabel('Zipf Exponent Alpha')
    #     pyt.ylabel(METRIC_LABEL[metric])
    #     if j == 0:
    #         pyt.legend(loc="upper left", ncol=4, bbox_to_anchor=(0,1.2, 2.2, 0.05), mode='expand')
    #
    # # pyt.show()
    # pyt.savefig(os.path.join(plotdir, "static-2.pdf"), bbox_inches='tight')
    # pyt.savefig(os.path.join(plotdir, "static-2.png"), bbox_inches='tight')
    # pyt.close()

def dynamic1(plotdir):
    import ast
    # dyamic
    results = ast.literal_eval(str(np.load("dynamic-1.npy")))
    fig = pyt.figure(figsize=(24, 6))
    for j, metric in enumerate(METRIC[1:-1]):
        ax = pyt.subplot(1,3,j+1)
        for i, method in enumerate(STRATEGY):
            all_data = results['DYNAMIC1'][metric][method]
            x = list(sorted(all_data.keys()))
            y = []
            yerr = [[],[]]
            for k in x:
                data = all_data[k]
                if metric == 'Q':
                    data = np.array(data)/RATE
                if metric == 'DIFFER':
                    data = np.array(data) * 100
                be,en = confidence_interval(data, 0.05, 0.95)
                mean = np.mean(data)
                y.append(mean)
                yerr[0].append(mean-be)
                yerr[1].append(en-mean)
            pyt.errorbar(np.array(x)*100, y, yerr=yerr, fmt=STYLE[method], label=STRATEGY_LABEL[method], capsize=5, elinewidth=2)
        if metric=="MISS":
            ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
            pyt.ylim(8000, 16000)
        if metric == 'DIFFER':
            pyt.ylim(0, 60)
        # if metric == 'Q':
        #     pyt.ylim(13, 17)
        pyt.title("(%s)  %s" % (chr(ord('a') + j), METRIC_TITLE[metric]), y=-0.31)
        pyt.xlabel('Shuffle Section (%)')
        pyt.ylabel(METRIC_LABEL[metric])
        if j==0:
            pyt.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.2, 3.5, 0.05), mode='expand')
    # pyt.show()
    pyt.savefig(os.path.join(plotdir, "dynamic-1.pdf"), bbox_inches='tight')
    pyt.savefig(os.path.join(plotdir, "dynamic-1.png" ), bbox_inches='tight')
    pyt.close()

    fig = pyt.figure(figsize=(12, 12))
    for i, method in enumerate(STRATEGY):
        ax = pyt.subplot(3,1,i+1)
        data = results['DYNAMIC1']['HIT'][method][0.2]
        data = data[:int(len(data)*0.1)]
        data = zip(*data)
        data[1] = np.array(data[1])*100
        pyt.plot(data[0], data[1], STYLE[method][:-1], label=STRATEGY_LABEL[method])
        for j, v in enumerate(data[1]):
            # print "now", v
            if v>BORDER[method][0] and j+1<len(data[1]) and data[1][j+1]<BORDER[method][1]:
                xx = data[0][j]-1
                yy = (v+data[1][j+1])/2
                sub = data[1][j+1]-v
                pyt.text(xx, yy, "%.1f%%" % sub, fontsize=12, horizontalalignment='right', verticalalignment='center')
        pyt.ylim(60,100)
        ax.xaxis.set_major_locator(MultipleLocator(30))
        if i==2:
            pyt.xlabel("Time (s)")
        if i==1:
            pyt.ylabel(METRIC_LABEL['HIT'])
        if i==0:
            pyt.plot([],[], STYLE[STRATEGY[i+1]][:-1], label=STRATEGY_LABEL[STRATEGY[i+1]])
            pyt.plot([],[], STYLE[STRATEGY[i+2]][:-1], label=STRATEGY_LABEL[STRATEGY[i+2]])
            pyt.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.25, 1.05, 0.05), mode='expand')
    # pyt.show()
    pyt.savefig(os.path.join(plotdir, "dynamic-1-hit.pdf"), bbox_inches='tight')
    pyt.savefig(os.path.join(plotdir, "dynamic-1-hit.png"), bbox_inches='tight')
    pyt.close()

def dynamic2(plotdir):
    import ast
    # dyamic
    results = ast.literal_eval(str(np.load("dynamic-2.npy")))
    fig = pyt.figure(figsize=(25, 6))
    for j, metric in enumerate(METRIC[1:-1]):
        ax = pyt.subplot(1,3,j+1)
        for i, method in enumerate(STRATEGY):
            data = results['DYNAMIC2'][metric][method]
            # print data
            y, x = cdf(data)
            if metric == 'Q':
                y /= RATE
            if metric == 'DIFFER':
                y = y*100
            pyt.plot(y, x, STYLE[method][:-1], label=STRATEGY_LABEL[method])
        if metric=="MISS":
            ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        pyt.title("(%s)  %s" % (chr(ord('a')+j), METRIC_TITLE[metric]), y=-0.31)
        pyt.ylim(0,1)
        if metric == 'DIFFER':
            pyt.xlim(0, 60)
        if metric=="MISS":
            pyt.xlim(8000, 16000)
        if metric == 'Q':
            pyt.xlim(13, 17)
        pyt.ylabel('CDF')
        pyt.xlabel(METRIC_LABEL[metric])
        if j==0:
            pyt.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, 1.2, 3.5, 0.05), mode='expand')
    # pyt.show()
    pyt.savefig(os.path.join(plotdir, "dynamic-2.pdf"), bbox_inches='tight')
    pyt.savefig(os.path.join(plotdir, "dynamic-2.png"), bbox_inches='tight')
    pyt.close()

def myplot(plotdir):
    static(plotdir)
    dynamic1(plotdir)
    dynamic2(plotdir)

if __name__ == '__main__':
    myplot('plot')
