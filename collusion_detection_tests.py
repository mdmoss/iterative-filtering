import attacks
from bias_estimate import bias_estimate
from collusion_detection import find_colluders
from iterative_filter import iterfilter, exponential
import mle
import readings_generator
from variance_estimate import variance_estimate
import matplotlib.pyplot as pyplot

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
    colluders, ks_results, regularised_errors = find_colluders(sophisticated_attack_readings, estimates)
    print(ks_results)

    fig = pyplot.figure()

    axes = fig.add_subplot(2, 2, 1)
    axes.set_title('Legitimate Sensor')
    axes.hist(regularised_errors[0], bins=20)
    axes.set_ylim(ymax=150)
    axes.set_xlim(xmin=-4, xmax=4)

    axes = fig.add_subplot(2, 2, 2)
    axes.set_title('Legitimate Sensor')
    axes.hist(regularised_errors[1], bins=20)
    axes.set_ylim(ymax=150)
    axes.set_xlim(xmin=-4, xmax=4)


    axes = fig.add_subplot(2, 2, 3)
    axes.set_title('Simple Colluder')
    axes.hist(regularised_errors[21], bins=20)
    axes.set_ylim(ymax=150)
    axes.set_xlim(xmin=-4, xmax=4)


    axes = fig.add_subplot(2, 2, 4)
    axes.set_title('Averaging Colluder')
    axes.hist(regularised_errors[22], bins=20)
    axes.set_ylim(ymax=150)
    axes.set_xlim(xmin=-4, xmax=4)


    pyplot.savefig('./collusion_detection_regularisation_failure.png')
    pyplot.show()