#!/usr/bin/env python

'''Testing to see if our RobustAggregete implementation is biased'''

from scipy.stats import bayes_mvs
import matplotlib.pyplot as pp

from iterative_filter import iterfilter, reciprocal, exponential
import robust_aggregate
import readings_generator

if __name__ == '__main__':
    repeats = 1000
    variance = 1
    bias = 0
    truth = 0
    times = 10
    num_sensors = 10

    variances = [variance] * num_sensors
    biases = [bias] * num_sensors

    time_errors = [[] for t in range(times)]

    def truth_fn(t):
        return truth

    for i in range(repeats):
        print ('{}/{}'.format(i, repeats))

        readings = readings_generator.readings(biases, variances, times, truth_fn)
        estimate = robust_aggregate.estimate(readings, exponential)

        for t in range(times):
            time_errors[t] += [estimate[t]] 

    mvs = [bayes_mvs(t) for t in time_errors]

    pp.errorbar(range(times), [m[0][0] for m in mvs], yerr=[m[0][0]-m[1][0] for m in mvs])
    pp.show()
