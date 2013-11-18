#!/usr/bin/python

import sys
import math
import random

import matplotlib.pyplot as pp
import scipy.stats as stats

"""
Implementation of Algorithm 1 from [RIBJ 2013]

Input:
    X: Readings of each sensor
    N: Number of sensors
    T: Number of readings for each sensor

Output: Reputation vector r

Note:
    x(a,b): reading of sensor a at time b
    X contains readings of each sensor, indexed by time interval
"""

# Test data
intel_X = [[19.3612, 19.3612, 19.3612],
           [19.42, 19.4102, 19.42],
           [19.0084, 19.0084, 19.0084],
           [18.5674, 18.5478, 17.117],
           [17.95, 21.282, 21.3408],
           [22.153, 21.347, 20.813],
           [18.0088, 18.0088, 21.625],
           [20.4, 20.4098, 19.7924]]
intel_N = 8
intel_T = 3

def aggregate(instant_readings, weights):
    # dot product
    top = sum([r * w for r, w in zip(instant_readings, weights)])
    bottom = sum(weights)
    return top / bottom

def compute_next_r(readings, weights):
    # matrix rotation
    instant_readings = [[x[i] for x in readings] for i in range(len(readings[0]))]
    return [aggregate(r, weights) for r in instant_readings]

def sensor_distance(sensor_readings, next_r):
    return sum([(x - r)**2 for x, r in zip(sensor_readings, next_r)])

assert(sensor_distance([1,2,3], [1,2,3]) == 0)
assert(sensor_distance([1,1,1,1], [0,0,0,0]) == 4)

def compute_d(readings, next_r):
    return [sensor_distance(x, next_r) / len(readings[0]) for x in readings]

assert(compute_d([[1,1,1,1]], [0,0,0,0]) == [1])
assert(compute_d([[1,1,1,1]], [1,1,1,1]) == [0])

def compute_next_w(distances, readings, next_r, g):
    return [g(distances[i]) for i in range(len(distances))]

def iterative_filter(x, n, t, g):
    l = 0
    w = [[1] * n]
    r = [[]]
    converged = False
    while not converged:
        r.append(compute_next_r(x, w[l]))
        d = compute_d(x, r[l+1])
        w.append(compute_next_w(d, x, r[l+1], g))

        if [round(x, 4) for x in r[l]] == [round(y, 4) for y in r[l-1]]:
            converged = True
        l += 1;
    return r[l]

def reciprocal(distance):
    if distance:
        return distance**-1
    else:
        return sys.maxsize

def exponential(distance):
    return math.exp(distance * -1)

assert(iterative_filter(intel_X, intel_N, intel_T, reciprocal) == [19.42, 19.4102, 19.42])

def iterfilter(readings, discriminant):
    return iterative_filter(readings, len(readings), len(readings[0]), discriminant)

TRUTH = 0
VARIANCE = 1
COLLUSION_VALUE = 3

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

def test_iterfilter(num_honest, num_skewing, num_avg, num_times, repetitions, randseed=None):
    if randseed:
        random.seed(randseed)

    num_pre_avg = num_honest + num_skewing
    recip_rms = [] 
    expo_rms = [] 

    for r in range(repetitions):
        honest_data = [[random.gauss(TRUTH, VARIANCE) for j in range(num_times)] for i in range(num_honest)]
        skewed_data = [[random.gauss(COLLUSION_VALUE, VARIANCE) for j in range(num_times)] for i in range(num_skewing)]
        uncolluded_data = honest_data + skewed_data
        sneaky_data = [[sum([uncolluded_data[j][i] for j in range(num_pre_avg)])/num_pre_avg for i in range(num_times)]]
        final_data = uncolluded_data + sneaky_data

        recip_rms += [rms_error(iterfilter(final_data, reciprocal), [TRUTH] * num_times)]
        expo_rms += [rms_error(iterfilter(final_data, exponential), [TRUTH] * num_times)]

    recip_rms_bayes = stats.bayes_mvs(recip_rms)
    expo_rms_bayes = stats.bayes_mvs(expo_rms)

    return (recip_rms_bayes[0], expo_rms_bayes[0])

if __name__ == '__main__':
    sensors = 30
    max_colluders = 30
    seed = random.random()

    values = []
    recip_results = [] 
    expo_results = []
    for i in range(1, max_colluders + 1):
        values += [i]
        recip_res, expo_res = test_iterfilter(sensors - i, i - 1, 1, 1, 100, randseed=seed)
        recip_results += [recip_res]
        expo_results += [expo_res]
  
    recip_means = [x[0] for x in recip_results]
    recip_errors = [x[0] - x[1][0] for x in recip_results]

    expo_means = [x[0] for x in expo_results]
    expo_errors = [x[0] - x[1][0] for x in expo_results]

    pp.errorbar(values, recip_means, yerr=recip_errors)
    pp.errorbar(values, expo_means, yerr=expo_errors)
    pp.xlabel('Number of Colluding Sensors')
    pp.ylabel('RMS Error')
    pp.title('RMS Error by Number of Colluding Sensors (randseed={})'.format(seed))
    pp.show()
