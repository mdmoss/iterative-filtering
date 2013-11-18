#!/usr/bin/python

import math
import random
import sys

import matplotlib.pyplot as pp
import scipy.stats as stats

import iterative_filter

TRUTH = 0
VARIANCE = 1

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

assert (rms_error([0], [0]) == 0)
assert (rms_error([2, 2], [0, 0]) == 2)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ('Usage: timeslice_test num_sensors num_times num_repetitions')
        sys.exit()

    num_sensors = int(sys.argv[1])
    num_times = int(sys.argv[2])
    num_repetitions = int(sys.argv[3])

    print ('{} sensors at {} times, {} repetitions'.format(num_sensors, num_times, num_repetitions))
    recip_error_calcs = [[] for i in range(num_times)]
    exp_error_calcs = [[] for i in range(num_times)]
    truth = [TRUTH * num_sensors for i in range(num_times)]

    for i in range(num_repetitions):
        readings = [[random.gauss(TRUTH, VARIANCE) for i in range(num_sensors)] for j in range(num_times)]

        for j in range(num_times):
            recip_iterative_result = iterative_filter.by_time(readings[0:j+1], iterative_filter.reciprocal)
            exp_iterative_result = iterative_filter.by_time(readings[0:j+1], iterative_filter.exponential)

            recip_avg_error = rms_error(recip_iterative_result[0:1], [TRUTH] * len(recip_iterative_result))
            exp_avg_error = rms_error(exp_iterative_result[0:1], [TRUTH] * len(exp_iterative_result))

            recip_error_calcs[j].append(recip_avg_error)
            exp_error_calcs[j].append(exp_avg_error)

    recip_error_bayes = [stats.bayes_mvs(x) for x in recip_error_calcs]
    exp_error_bayes = [stats.bayes_mvs(x) for x in exp_error_calcs]

    slice_length = range(1, num_times + 1)    
  
    recip_mids = [t[0][0] for t in recip_error_bayes]
    recip_errors = [e[0][1][0]-m for (e, m) in zip(recip_error_bayes, recip_mids)]
  
    exp_mids = [t[0][0] for t in exp_error_bayes]
    exp_errors = [e[0][1][0]-m for (e, m) in zip(exp_error_bayes, exp_mids)]

    pp.errorbar(slice_length, recip_mids, yerr=recip_errors, label='reciprocal discriminant')
    pp.errorbar(slice_length, exp_mids, yerr=exp_errors, label='exponential discriminant')
    pp.xlabel('Instants')
    pp.ylabel('RMS Error')
    pp.title('RMS Error using discriminant (n={})'.format(num_repetitions))
    pp.show()
