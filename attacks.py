#!/usr/bin/env python

import random
from numpy import array, vstack, mean, concatenate
from bias_estimate import bias_estimate
from iterative_filter import iterfilter, exponential
import mle
from readings_generator import readings
import robust_aggregate
from variance_estimate import variance_estimate

__author__ = 'Edward'


def transpose(l):
    return map(list, zip(*l))


_colluder_noise_variance = 0.1


def _noise():
    return random.gauss(0, _colluder_noise_variance ** 2)


def readings_simple_attack(legitimate_readings,
                           true_value,
                           num_colluders, colluder_bias):
    num_times = len(legitimate_readings[0])
    colluder_values = [[true_value(t) + colluder_bias + _noise() for t in range(num_times)]
                       for i in range(num_colluders)]
    return array(legitimate_readings.tolist() + colluder_values)


def readings_sophisticated_attack(legitimate_readings,
                                  true_value,
                                  num_colluders, colluder_bias):
    num_times = len(legitimate_readings[0])
    simple_colluder_values = [[true_value(t) + colluder_bias + _noise() for t in range(num_times)]
                              for i in range(num_colluders - 1)]
    legitimate_and_simple_readings = legitimate_readings.tolist() + simple_colluder_values
    averaging_colluder_values = [mean(t_readings) + _noise()
                                 for t_readings in transpose(legitimate_and_simple_readings)]
    return array(legitimate_and_simple_readings + [averaging_colluder_values])


def readings_ks_attack(legitimate_readings,
                       true_value,
                       num_colluders, colluder_bias):
    """
        readings[s,t] is reading of sensor s at time t

    """

    def estimate(readings):
        return robust_aggregate.estimate(readings, robust_aggregate.exponential)

    b = colluder_bias #/ (num_colluders - 1)
    colluder_value = [v + b + _noise() for v in estimate(legitimate_readings)]
    readings_with_colluders = vstack((array(legitimate_readings), array(colluder_value)))
    for i in range(num_colluders - 2):
        colluder_value = [v + b + _noise() for v in estimate(readings_with_colluders)]
        readings_with_colluders = vstack((array(readings_with_colluders), array(colluder_value)))
    colluder_bias_estimate = bias_estimate(readings_with_colluders)
    colluder_variance_estimate = variance_estimate(readings_with_colluders, colluder_bias_estimate)
    mle_estimate = mle.estimate(readings_with_colluders, colluder_bias_estimate, colluder_variance_estimate)
    final_colluder_value = [v + _noise() for v in mle_estimate]
    return vstack((array(readings_with_colluders), array(final_colluder_value)))


if __name__ == "__main__":
    num_legitimate_sensors = 15
    num_colluders = 5
    colluder_bias = 6
    biases = [0] * num_legitimate_sensors
    variances = [1] * num_legitimate_sensors
    num_times = 4
    true_value = lambda t: 0
    legitimate_readings = readings(biases, variances, num_times, true_value)
    simple_attack_readings = readings_simple_attack(legitimate_readings, true_value, num_colluders, colluder_bias)
    sophisticated_attack_readings = readings_sophisticated_attack(legitimate_readings, true_value, num_colluders,
                                                                  colluder_bias)
    sk_attack_readings = readings_ks_attack(legitimate_readings, true_value, num_colluders, colluder_bias)
    print(simple_attack_readings)
    print(sophisticated_attack_readings)
    print(sk_attack_readings)
