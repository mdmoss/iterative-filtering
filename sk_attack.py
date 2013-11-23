from numpy import vstack, array
import readings_generator
import robust_aggregate

__author__ = 'Edward'


def attack(readings, num_colluders, colluder_target_bias):
    """
        readings[s,t] is reading of sensor s at time t

    """
    def estimate(readings):
        return robust_aggregate.estimate(readings, robust_aggregate.exponential)

    b = colluder_target_bias/(num_colluders - 1)
    colluder_value = [v + b for v in estimate(readings)]
    readings_with_colluders = vstack((array(readings), array(colluder_value)))
    for i in range(num_colluders - 2):
        colluder_value = [v + b for v in estimate(readings_with_colluders)]
        readings_with_colluders = vstack((array(readings_with_colluders), array(colluder_value)))
    final_colluder_value = estimate(readings_with_colluders)
    return vstack((array(readings_with_colluders), array(final_colluder_value)))

if __name__ == "__main__":
    num_legit_sensor = 9
    num_colluders = 4
    num_times = 4
    true_value = lambda t: 0
    biases = [0] * num_legit_sensor
    variances = [1] * num_legit_sensor
    readings = readings_generator.readings(biases, variances, num_times, true_value)
    colluder_target_bias = 10
    s = attack(readings, num_colluders, colluder_target_bias)
    print(s)
    print(len(s))