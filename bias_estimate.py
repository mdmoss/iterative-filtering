#!/usr/bin/python

import random
from numpy import array, mean, hstack, vstack, linalg

import matplotlib.pyplot as pp
import scipy.stats as stats

# Readings format: Array of sensors containing array of times
# [[S1T1, S1T2], [S2T1, S2T2]]

# Danger Will Robinson: Lack of thread safety ahead!
_readings = []


#def rms_error(estimates, truths):
#    return math.sqrt(sum([(e - t) ** 2 for (e, t) in zip(estimates, truths)]) / len(estimates))


def delta(readings):
    return array([
        [mean(array(readings[i]) - array(readings[j])) for j in range(0, len(readings[i]))]
        for i in range(0, len(readings))
    ])


#def delta(i, j, readings):
#    assert (len(_readings[i]) == len(_readings[j]))
#    return sum([a - b for (a, b) in zip(_readings[i], _readings[j])]) / len(_readings[i])


def bias_estimate(delta_matrix):
    num_sensors = len(delta_matrix)

    def d(i, j):
        if i < j:
            return -delta_matrix[i, j]
        else:
            return delta_matrix[i, j]

    def c(k):
        return 2 * sum([1 / d(i, k) for i in range(num_sensors) if i != k])

    c_vector = array([c(k) for k in range(num_sensors)] + [0])

    def f(i, k):
        if i != k:
            return 2 / (d(i, k)**2)
        else:
            return sum([f(j, k) for j in range(num_sensors) if j != k])

    def f_vector(k):
        return [f(i, k) for i in range(num_sensors)]

    f_matrix = array([f_vector(k) for k in range(num_sensors)])

    a_matrix = vstack((
        hstack((
            f_matrix,
            array([-1] * num_sensors)
        )),
        [1] * num_sensors + [0]
    ))

    return linalg.solve(a_matrix, c_vector)[range(num_sensors)]


"""
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
"""

if __name__ == "__main__":
    num_sensors = 20
    num_t1_sensors = 5
    num_t2_sensors = num_sensors - num_t1_sensors
    t1_bias = 0
    t2_bias = 2
    bias_compensator = -(num_t1_sensors * t1_bias + num_t2_sensors * t2_bias) / num_sensors
    biases = array([t1_bias + bias_compensator] * num_t1_sensors +
                   [t2_bias + bias_compensator] * num_t2_sensors)
    variances = array([1 + 0.1*t for t in range(num_sensors)])
    true_value = lambda t: 1 + 3 * t
    num_times = 10
    readings = [
        [random.gauss(true_value(t) + biases[sensor], variances[sensor]) for t in range(num_times)]
        for sensor in range(num_sensors)
    ]
    delta_matrix = delta(readings)
    estimate = bias_estimate(delta_matrix)
    print(estimate)