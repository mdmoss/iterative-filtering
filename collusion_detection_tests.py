import attacks
from bias_estimate import bias_estimate
from collusion_detection import find_colluders
from iterative_filter import iterfilter, exponential
import mle
import readings_generator
from variance_estimate import variance_estimate

__author__ = 'Pierzchalski'


def robust_first_estimate(readings):
    first_bias_est = bias_estimate(readings)
    first_variance_est = variance_estimate(readings, first_bias_est)

    weights = mle.weight_vector(first_bias_est, first_variance_est)
    return iterfilter(readings, exponential, weights)


if __name__ == "__main__":
    num_legit_sensors = 20
    biases = [0] * num_legit_sensors
    variances = [1] * num_legit_sensors
    num_times = 800
    true_value = lambda t: 0
    num_colluders = 3
    colluder_bias = 5
    legitimate_readings = readings_generator.readings(biases, variances, num_times, true_value)
    sophisticated_attack_readings = attacks.readings_sophisticated_attack(legitimate_readings,
                                                                          true_value,
                                                                          num_colluders,
                                                                          colluder_bias)
    estimates = robust_first_estimate(sophisticated_attack_readings)
    colluders, ks_results = find_colluders(sophisticated_attack_readings, estimates)