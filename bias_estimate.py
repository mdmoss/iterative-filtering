#!/usr/bin/python

import random
import math

from scipy.optimize import minimize
import matplotlib.pyplot as pp
import scipy.stats as stats

# Readings format: Array of sensors containing array of times
# [[S1T1, S1T2], [S2T1, S2T2]]

# Danger Will Robinson: Lack of thread safety ahead!
_readings = []

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

def delta(i, j):
    assert(len(_readings[i]) == len(_readings[j]))
    return sum([a - b for (a, b) in zip(_readings[i], _readings[j])]) / len(_readings[i])

def per_sensor_bias(i, bi, j, bj):
    return (bi - bj - delta(i, j))**2

def bias_estimator(x):
    total = 0
    for i in range(len(x)):
        for j in range(i):
            total += per_sensor_bias(i, x[i], j, x[j]) 
    return total

def bias_constraint(x):
    return sum(x)

bias_constraint_dict = {
    'type': 'eq',
    'fun': bias_constraint,
}

def estimate(readings):
    global _readings
    _readings = readings 
    res = minimize(bias_estimator, [0]*len(readings), method='SLSQP', constraints=bias_constraint_dict)
    return res.x

TRUTH = 0

if __name__ == '__main__':
    honest_sensors = 20
    colluding_sensors = 10
    colluder_target = 100
    repeats = 1
    bias_std_dev = 1
    times = 10

    for i in range(repeats):
        honest_bias = [random.gauss(0, bias_std_dev) for x in range(honest_sensors)]
        colluder_bias = [colluder_target] * colluding_sensors
        honest_data = [[random.gauss(TRUTH, x) for t in range(times)] for x in honest_bias]
        colluder_data = [[colluder_target for t in range(times)] for x in colluder_bias]
        bias = honest_bias + colluder_bias
        data = honest_data + colluder_data
        estimate_bias = estimate(data)
        print ([round(x, 4) for x in bias])
        print ([round(x, 4) for x in estimate_bias])
        print (rms_error(bias, estimate_bias))

