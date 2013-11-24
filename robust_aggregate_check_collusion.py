#!/usr/bin/python

import sys
import math
import random

from robust_aggregate import estimate, reciprocal, exponential

import matplotlib.pyplot as pp
import scipy.stats as stats

TRUTH = 0
VARIANCE = 2
COLLUSION_VALUE = 7

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

def test_iterfilter(num_honest, num_skewing, num_avg, num_times, repetitions, randseed=None):
    if randseed:
        random.seed(randseed)

    num_pre_avg = num_honest + num_skewing
    recip_rms = [] 
    expo_rms = [] 

    for r in range(repetitions):
        print (r)
        honest_data = [[random.gauss(TRUTH, VARIANCE) for j in range(num_times)] for i in range(num_honest)]
        skewed_data = [[random.gauss(COLLUSION_VALUE, VARIANCE) for j in range(num_times)] for i in range(num_skewing)]
        uncolluded_data = honest_data + skewed_data
        sneaky_data = [[sum([uncolluded_data[j][i] for j in range(num_pre_avg)])/num_pre_avg for i in range(num_times)]]
        final_data = uncolluded_data + sneaky_data

        recip_rms += [rms_error(estimate(final_data, reciprocal), [TRUTH] * num_times)]
        expo_rms += [rms_error(estimate(final_data, exponential), [TRUTH] * num_times)]

    recip_rms_bayes = stats.bayes_mvs(recip_rms)
    expo_rms_bayes = stats.bayes_mvs(expo_rms)

    return (recip_rms_bayes[0], expo_rms_bayes[0])

if __name__ == '__main__':
    sensors = 10
    colluders = 3
    repeats = 100
    max_averagers = 3
    times = 3
    seed = round(random.random(), 6)

    values = []
    recip_results = [] 
    expo_results = []
    for i in range(1, max_averagers + 1):
        values += [i]
        recip_res, expo_res = test_iterfilter(sensors - colluders, colluders - i, i, times, repeats, randseed=seed)
        recip_results += [recip_res]
        expo_results += [expo_res]
  
    recip_means = [x[0] for x in recip_results]
    recip_errors = [x[0] - x[1][0] for x in recip_results]

    expo_means = [x[0] for x in expo_results]
    expo_errors = [x[0] - x[1][0] for x in expo_results]

    pp.errorbar(values, recip_means, yerr=recip_errors, label='reciprocal')
    pp.errorbar(values, expo_means, yerr=expo_errors, label='exponential')
    pp.xlabel('Number of Averaging Colluders')
    pp.ylabel('RMS Error')
    pp.suptitle('RMS Error by Number of Averaging Colluders')
    pp.title('sensors={}, colluders={}, colluder bias={}, times={}, repeats={}, randseed={}'.format(sensors, colluders, COLLUSION_VALUE, times, repeats, seed))
    pp.legend(loc='upper left')
    pp.show()
