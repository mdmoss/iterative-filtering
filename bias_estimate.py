#!/usr/bin/python

import matplotlib.pyplot as pyplot
from numpy import array, mean, vstack
from scipy import optimize
import scipy.stats as stats

from readings_generator import readings


def delta(readings):
    """
        `readings` is a sensor-major array of sensor/time readings. `readings[s,t]` is the reading of
        sensor `s` at time `t`.
    """
    return array([
        [mean(array(readings[i]) - array(readings[j])) for j in range(0, len(readings))]
        for i in range(0, len(readings))
    ])


def bias_estimate(readings):
    """
        `readings` is a sensor-major array of sensor/time readings. `readings[s,t]` is the reading of
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
    return solution


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
    num_readings_samples = 1000
    readings_samples = [readings(biases, variances, num_times, true_value) for i in range(num_readings_samples)]
    bias_estimates = array([v for v in map(bias_estimate, readings_samples)])
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