
from attacks import readings_ks_attack, readings_simple_attack, readings_sophisticated_attack
import readings_generator

__author__ = 'Edward'

if __name__ == "__main__":
    num_legit_sensor = 9
    num_colluders = 4
    num_times = 4
    true_value = lambda t: 0
    biases = [0] * num_legit_sensor
    variances = [1] * num_legit_sensor
    legitimate_readings = readings_generator.readings(biases, variances, num_times, true_value)
    colluder_bias = 10
    simple_attack_readings, sophisticated_attack_readings, ks_attack_readings = [f(legitimate_readings,
                                                                                   true_value,
                                                                                   num_colluders,
                                                                                   colluder_bias)]
    simple_attack_readings = readings_simple_attack(legitimate_readings,
                                                    true_value,
                                                    num_colluders,
                                                    colluder_bias)
    print(s)
    print(len(s))