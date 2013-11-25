#!/usr/bin/env python

'''Collusion detection and revocation for the RobustAggregate algorithm'''

import random

from scipy.stats import kstest
import numpy as np


def find_colluders(readings, estimates, alpha=0.05):
    num_sensors = len(readings)

    def sensor_errors(sensor_index):
        return [s - e for s, e in zip(readings[sensor_index], estimates)]

    errors = [sensor_errors(s) for s in range(num_sensors)]
    means = [np.mean(e) for e in errors]

    def dezero(x):
        if x == 0:
            return 0.0000000001
        else:
            return x

    std_devs = [dezero(np.std(e, ddof=1)) for e in errors]

    def sensor_normalized_errors(sensor_index):
        m = means[sensor_index]
        s = std_devs[sensor_index]
        return [(e - m)/s for e in errors[sensor_index]]

    normalized_errors = [sensor_normalized_errors(s) for s in range(num_sensors)]
    ks_results = [(index, kstest(x, 'norm')[1]) for index, x in zip(range(num_sensors), normalized_errors)]
    colluders = [index for index, test in ks_results if test < alpha]
    return colluders, ks_results, normalized_errors


def partition(readings, estimates):
    errors = [[x - r for (x, r) in zip(s, estimates)] for s in readings]
    means = [np.mean(e) for e in errors]
    std_devs = [s if s != 0 else 0.01 for s in [np.std(e, ddof=1) for e in errors]]
    normalized = [[(x - means[i]) / std_devs[i] for x in errors[i]] for i in range(len(errors))]
    ks_results = [kstest(x, 'norm') for x in normalized]

    return [r for (r, ks) in zip(readings, ks_results) if ks[1] > 0.5]


if __name__ == '__main__':
    assert (partition([[1, 2, 4, 4], [3, 4, 4, 1]], [1, 2, 3, 6]) == [[1, 2, 4, 4], [3, 4, 4, 1]])

    for i in range(1000):
        means = [0] * 10
        distributions = [[random.gauss(m, 1) for i in range(100)] for m in means]
        dodgy = [[-1] * 100]
        assert (dodgy not in partition(distributions + dodgy, means + [0]))
