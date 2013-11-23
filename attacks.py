from numpy import array, vstack
from readings_generator import readings
from iterative_filter import iterfilter, exponential

__author__ = 'Edward'


def readings_simple_attack(legitimate_readings, num_colluders, colluder_bias):
    aggregate = iterfilter(legitimate_readings, exponential)
    collusion_readings = [[v + colluder_bias for v in aggregate]] * num_colluders
    return vstack((array(legitimate_readings), array(collusion_readings)))


def readings_sophisticated_attack(legitimate_readings, num_colluders, colluder_bias):
    aggregate = iterfilter(legitimate_readings, exponential)
    collusion_readings = [[v + colluder_bias for v in aggregate]] * (num_colluders - 1)
    all_readings = vstack((array(legitimate_readings), array(collusion_readings)))
    final_colluder_readings = iterfilter(all_readings, exponential)
    return vstack((all_readings, array(final_colluder_readings)))


if __name__ == "__main__":
    num_legitimate_sensors = 15
    num_colluders = 5
    colluder_bias = 6
    biases = [0] * num_legitimate_sensors
    variances = [1] * num_legitimate_sensors
    num_times = 8
    true_value = lambda t: 0
    legitimate_readings = readings(biases, variances, num_times, true_value)
    simple_attack_readings = readings_sophisticated_attack(legitimate_readings, num_colluders, colluder_bias)
    print(simple_attack_readings)