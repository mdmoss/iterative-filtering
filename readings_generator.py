import random
from numpy import mean, array

__author__ = 'Pierzchalski'


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