#!/usr/bin/env python

'''Testing to see if our RobustAggregete implementation is biased'''

import math

from scipy.stats import bayes_mvs
import matplotlib.pyplot as pp

import mle
from iterative_filter import iterfilter, reciprocal, exponential
import robust_aggregate
import readings_generator

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

if __name__ == '__main__':
    repeats = 1000
    variance = 1
    bias = 0
    truth = 0
    times = 10
    num_sensors = 10

    variances = [variance] * num_sensors
    biases = [bias] * num_sensors

    iter_rms_errors = []
    mle_rms_errors = []

    def truth_fn(t):
        return truth

    for i in range(repeats):
        print ('{}/{}'.format(i, repeats))

        readings = readings_generator.readings(biases, variances, times, truth_fn)
        estimate = robust_aggregate.estimate(readings, exponential)
        iter_rms_errors += [rms_error(estimate, [0]*num_sensors)]
   
        mle_estiamte = mle.estimate(readings, variances, biases) 
        mle_rms_errors += [rms_error(mle_estiamte, [0]*num_sensors)]

    iter_mvs = bayes_mvs(iter_rms_errors)
    mle_mvs = bayes_mvs(mle_rms_errors)


    pp.bar([0, 1], [iter_mvs[0][0], mle_mvs[0][0]], yerr=[iter_mvs[0][0]-iter_mvs[1][0], mle_mvs[0][0]-mle_mvs[1][0]])
    pp.show()
