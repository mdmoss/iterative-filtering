#!/usr/bin/env python

'''Collusion detection and revocation for the RobustAggregate algorithm'''

import random

from scipy.stats import kstest
import numpy as np

def partition(readings, estimates):
    errors = [[x - r for (x, r) in zip(s, estimates)] for s in readings]
    means = [np.mean(e) for e in errors]
    std_devs = [s if s != 0 else 0.01 for s in [np.std(e, ddof=1) for e in errors]]
    normalized = [[(x - means[i]) / std_devs[i] for x in errors[i]] for i in range(len(errors))]
    ks_results = [kstest(x, 'norm') for x in normalized]

    return [r for (r, ks) in zip(readings, ks_results) if ks[1] > 0.5]

if __name__ == '__main__':
    assert(partition([[1, 2, 4, 4],[3, 4, 4, 1]], [1, 2, 3, 6]) == [[1, 2, 4, 4],[3, 4, 4, 1]])

    for i in range(1000):
        means = [0] * 10
        distributions = [[random.gauss(m, 1) for i in range(100)] for m in means]
        dodgy = [[-1] * 100]
        assert(dodgy not in partition(distributions + dodgy, means + [0]))
