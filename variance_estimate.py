__author__ = 'Pierzchalski'

from numpy import array, mean
from scipy import optimize, stats
import readings_generator
from bias_estimate import bias_estimate
import matplotlib.pyplot as pyplot


def variance_estimate(readings, biases):
    """
        `readings` is a sensor-major array of sensor/time readings. `readings[s,t]` is the reading of
        sensor `s` at time `t`.
    """
    num_sensors = len(readings)
    num_times = len(readings[0])
    #debiased_readings[s, t] is \hat{x}^t_s
    debiased_readings = array(readings) - array([biases[sensor] * num_times for sensor in range(num_sensors)])

    def beta_matrix_entry(sensor1, sensor2):
        return sum([
            (debiased_readings[sensor1, time] - debiased_readings[sensor2, time]) ** 2
            for time in range(num_times)
        ]) / (num_times - 1)

    beta_matrix = array([
        [beta_matrix_entry(sensor1, sensor2) for sensor2 in range(num_sensors)]
        for sensor1 in range(num_sensors)
    ])

    #mean_readings[t] is the mean of all sensor readings at time t.
    mean_readings = array([mean(r) for r in array(readings).transpose()])

    def target_function(v):
        return sum([
            sum([((v[sensor1] + v[sensor2]) / beta_matrix[sensor1, sensor2] - 1) ** 2
                 for sensor2 in range(sensor1)
            ])
            for sensor1 in range(num_sensors)
        ])

    constraint_value = num_sensors / (num_times * (num_sensors - 1)) * sum([
        sum([
            (debiased_readings[sensor, time] - mean_readings[time]) ** 2
            for time in range(num_times)
        ])
        for sensor in range(num_sensors)
    ])

    def constraint_function(v):
        return sum(v) - constraint_value

    solution = optimize.fmin_slsqp(func=target_function,
                                   x0=[0] * num_sensors,
                                   f_eqcons=constraint_function,
                                   iprint=0)
    return solution


if __name__ == "__main__":
    num_sensors = 20
    biases = array([0.3 * sensor for sensor in range(num_sensors)])
    variances = array([(num_sensors - sensor + 1) / 2 for sensor in range(num_sensors)])
    num_times = 10
    true_value = lambda t: (t - num_times / 2) ** 2
    num_readings = 20
    reading_sampling = [readings_generator.readings(biases, variances, num_times, true_value) for i in
                        range(num_readings)]
    bias_estimates = [bias_estimate(r) for r in reading_sampling]
    variance_estimates = array([variance_estimate(r, b) for r, b in zip(reading_sampling, bias_estimates)])
    alpha = 0.95
    #variance_estimates[i,s] gives the estimate of sensor s in reading i.
    #variance_estimates.transpose()[s, i] gives the same.
    confidence_intervals = [stats.bayes_mvs(v_est, alpha) for v_est in variance_estimates.transpose()]

    compensated_biases = biases - array([mean(biases)] * num_sensors)
    fig = pyplot.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_title('Variance Estimation')
    axes.plot(variances)

    #axes.legend(['True', 'Estimated - Mean', 'Estimated - Standard Deviation'])
    axes.set_ylim(ymax=3)
    axes.set_xlabel('Sensor ID')
    axes.set_ylabel('Bias')

    pyplot.show()