#!/usr/bin/python

import random
import matplotlib.pyplot as pyplot
from numpy import array, mean, hstack, vstack, linalg, dot
from scipy import optimize, stats

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
        [mean(array(readings[i]) - array(readings[j])) for j in range(0, len(readings))]
        for i in range(0, len(readings))
    ])


#def delta(i, j, readings):
#    assert (len(_readings[i]) == len(_readings[j]))
#    return sum([a - b for (a, b) in zip(_readings[i], _readings[j])]) / len(_readings[i])


#def bias_estimate(delta_matrix):
#    num_sensors = len(delta_matrix)
#
#    def d(i, j):
#        if i < j:
#            return -delta_matrix[i, j]
#        else:
#            return delta_matrix[i, j]
#
#    def c(k):
#        return 2 * sum([1 / d(i, k) for i in range(num_sensors) if i != k])
#
#    c_vector = array([c(k) for k in range(num_sensors)] + [0])
#
#    def f(i, k):
#        if i != k:
#            return 2 / (d(i, k) ** 2)
#        else:
#            return sum([f(j, k) for j in range(num_sensors) if j != k])
#
#    def f_vector(k):
#        return [f(i, k) for i in range(num_sensors)]
#
#    f_matrix = array([f_vector(k) for k in range(num_sensors)])
#    #print("the f_matrix has shape {}".format(f_matrix.shape))
#
#    a_matrix = vstack((
#        hstack((
#            f_matrix,
#            array([[-1]] * num_sensors)
#        )),
#        [1] * num_sensors + [0]
#    ))
#
#    return linalg.solve(a_matrix, c_vector)[0:-1]


def bias_estimate_2(readings):
    """
        `readings` is a sensor-major matrix of sensor/time readings. `readings[s,t]` is the reading of
        sensor `s` at time `t`.
    """
    delta_matrix = delta(readings)
    num_sensors = len(delta_matrix)

    def target_function(b):
        return sum([
            sum([((b[sensor1] - b[sensor2]) / delta_matrix[sensor1, sensor2] - 1) ** 2 for sensor2 in range(sensor1)])
            for sensor1 in range(len(readings))
        ])

    def constraint_function(b):
        return array(sum(b))

    solution = optimize.fmin_slsqp(func=target_function,
                                   x0=[0] * num_sensors,
                                   f_eqcons=constraint_function,
                                   iprint=0)

    #solution = optimize.minimize(method='SLSQP',
    #                             fun=target_function,
    #                             x0=[0] * num_sensors,
    #                             constraints={'type': 'eq',
    #                                          'fun': constraint_function})

    return solution


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


def readings(biases, variances, num_times, true_value):
    num_sensors = len(biases)
    bias_compensator = -mean(biases)
    compensated_biases = biases + bias_compensator * array([1] * len(biases))
    return array([
        [random.gauss(true_value(t) + compensated_biases[sensor],
                      variances[sensor])
         for t in range(num_times)]
        for sensor in range(num_sensors)
    ])


if __name__ == "__main__":
    num_sensors = 20
    num_t1_sensors = 15
    num_t2_sensors = num_sensors - num_t1_sensors
    t1_bias = 0
    t2_bias = 2
    biases = array([t1_bias] * num_t1_sensors +
                   [t2_bias] * num_t2_sensors)
    bias_compensator = -mean(biases)
    compensated_biases = biases + bias_compensator * array([1] * len(biases))
    variances = array([1 + 0.1 * t for t in range(num_sensors)])
    true_value = lambda t: 1 + 3 * t
    num_times = 10
    num_readings_samples = 100
    readings_samples = [readings(biases, variances, num_times, true_value) for i in range(num_readings_samples)]
    bias_estimates = array([v for v in map(bias_estimate_2, readings_samples)])
    alpha = 0.95
    confidence_intervals = [stats.bayes_mvs(v, alpha) for v in bias_estimates.transpose()]

    means = array([ci[0][0] for ci in confidence_intervals])
    mean_lower_bounds = means - array([ci[0][1][0] for ci in confidence_intervals])
    mean_upper_bounds = array([ci[0][1][1] for ci in confidence_intervals]) - means

    stddev = array([ci[2][0] for ci in confidence_intervals])
    stddev_lower_bounds = stddev - array([ci[2][1][0] for ci in confidence_intervals])
    stddev_upper_bounds = array([ci[2][1][1] for ci in confidence_intervals]) - stddev

    fig = pyplot.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_title('Bias Estimation')
    axes.plot(compensated_biases)

    axes.errorbar(range(num_sensors), means, vstack((mean_lower_bounds, mean_upper_bounds)))
    axes.errorbar(range(num_sensors), stddev, vstack((stddev_lower_bounds, stddev_upper_bounds)))

    axes.legend(['True', 'Estimated - Mean', 'Estimated - Standard Deviation'])
    axes.set_ylim(ymax=3)
    axes.set_xlabel('Sensor ID')
    axes.set_ylabel('Bias')

    pyplot.show()