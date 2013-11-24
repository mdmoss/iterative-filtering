from matplotlib import pyplot
from attacks import readings_ks_attack, readings_simple_attack, readings_sophisticated_attack
from scipy.stats import bayes_mvs
from iterative_filter import exponential
import readings_generator
from robust_aggregate import rms_error
import robust_aggregate

__author__ = 'Edward'

if __name__ == "__main__":
    total_sensors = 20
    max_num_colluders = 10
    num_times = 5
    true_value = lambda t: t
    biases = [0] * total_sensors
    variances = [2] * total_sensors
    colluder_bias = 5


    def attack_rmse(attack, num_colluders):
        num_legit_sensors = total_sensors - num_colluders
        legitimate_readings = readings_generator.readings(biases[:num_legit_sensors], variances[:num_legit_sensors],
                                                          num_times, true_value)
        return rms_error(robust_aggregate.estimate(attack(legitimate_readings,
                                                          true_value,
                                                          num_colluders,
                                                          colluder_bias),
                                                   exponential), [true_value(t) for t in range(num_times)])

    num_iterations_per_attack_size = 100

    def rmse_ci(attack, num_colluders):
        rms_errors = [attack_rmse(attack, num_colluders) for i in range(num_iterations_per_attack_size)]
        mean_ci, variance_ci, stddev_ci = bayes_mvs(rms_errors)
        return mean_ci

    def cis_over_num_colluders(attack):
        return [rmse_ci(attack, num_colluders) for num_colluders in range(2, max_num_colluders + 1)]

    def cis_to_pyplot_bounds(cis):
        centers, bounds = zip(*cis)
        lower_bounds, upper_bounds = zip(*bounds)
        lower_bound_diffs = [c - lb for c, lb in zip(centers, lower_bounds)]
        upper_bound_diffs = [ub - c for c, ub in zip(centers, upper_bounds)]
        return centers, [lower_bound_diffs, upper_bound_diffs]

    simple, sophisticated, ks = (cis_to_pyplot_bounds(cis_over_num_colluders(attack))
                                 for attack in
                                 (readings_simple_attack, readings_sophisticated_attack, readings_ks_attack))

    fig = pyplot.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_title('RMS Error under Attacks')
    for centers, bounds in (simple, sophisticated, ks):
        axes.errorbar(range(2, max_num_colluders), centers, yerr=bounds)
    axes.legend(['Simple', 'Sophisticated', 'SK'])
    axes.set_xlabel('Number of colluding sensors')
    axes.set_ylabel('RMS Error')

    pyplot.show()

    #print(s)
    #print(len(s))