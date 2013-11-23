#!/usr/bin/env python

'''Implementation of RobustAggregate algorithm from [Rezvani, Ignjatovic, Bertino, Jha 2013]'''

import math

from scipy.stats import bayes_mvs
import matplotlib.pyplot as pp

from bias_estimate import bias_estimate
from variance_estimate import variance_estimate
import mle
from iterative_filter import iterfilter, reciprocal, exponential
import collusion_detection
import readings_generator

def estimate(readings, discriminant):
    first_bias_est = bias_estimate(readings)
    first_variance_est = variance_estimate(readings, first_bias_est)
    first_estimate = mle.estimate(readings, first_bias_est, first_variance_est)

    weights = mle.weight_vector(first_bias_est, first_variance_est)
    filtering_result = iterfilter(readings, discriminant, weights)
    
    trustworthy_readings = collusion_detection.partition(readings, filtering_result)

    second_bias_est = bias_estimate(trustworthy_readings)
    second_variance_est = variance_estimate(trustworthy_readings, second_bias_est)
    final_estimate = mle.estimate(trustworthy_readings, second_bias_est, second_variance_est)

    return final_estimate

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

if __name__ == '__main__':

    repeats = 10

    num_sensors = 20
    num_times = 100

    biases = [0] * 20

    variance_max = 5

    x_values = []
    cramer_rao = []
    iter_recip_means = []
    iter_recip_errors = []
    iter_expo_means = []
    iter_expo_errors = []
    robust_agg_recip_means = []
    robust_agg_recip_errors = []
    robust_agg_expo_means = []
    robust_agg_expo_errors = []

    for v in range(variance_max):
        variance = v + 1
        x_values += [variance]
        print ('variance: {}'.format(variance))

        variances = [variance] * 20
        def truth(t):
            return 0

        cramer_rao += [math.sqrt(1 / sum([1 / v for v in variances]))]

        iter_recip = []
        iter_expo = []
        robust_agg_recip = []
        robust_agg_expo = []

        for i in range(repeats):
            print (i)
            readings = readings_generator.readings(biases, variances, num_times, truth)
            iter_recip += [rms_error(iterfilter(readings, reciprocal), [0]*num_sensors)]
            iter_expo += [rms_error(iterfilter(readings, exponential), [0]*num_sensors)]
            robust_agg_recip += [rms_error(estimate(readings, reciprocal), [0]*num_sensors)]
            robust_agg_expo += [rms_error(estimate(readings, exponential), [0]*num_sensors)]

        iter_recip_mean = bayes_mvs(iter_recip)[0]
        iter_expo_mean = bayes_mvs(iter_expo)[0]
        robust_agg_recip_mean = bayes_mvs(robust_agg_recip)[0]
        robust_agg_expo_mean = bayes_mvs(robust_agg_expo)[0]

        iter_recip_means += [iter_recip_mean[0]]
        iter_expo_means += [iter_expo_mean[0]]
        robust_agg_recip_means += [robust_agg_recip_mean[0]]
        robust_agg_expo_means += [robust_agg_expo_mean[0]]

        iter_recip_errors += [iter_recip_mean[0] - iter_recip_mean[1][0]]
        iter_expo_errors += [iter_expo_mean[0] - iter_expo_mean[1][0]]
        robust_agg_recip_errors += [robust_agg_recip_mean[0] - robust_agg_recip_mean[1][0]]
        robust_agg_expo_errors += [robust_agg_expo_mean[0] - robust_agg_expo_mean[1][0]]

    pp.errorbar(x_values, iter_recip_means, yerr=iter_recip_errors, label='Iterative Filtering - reciprocal')
    pp.errorbar(x_values, iter_expo_means, yerr=iter_expo_errors, label='Iterative Filtering - exponential')
    pp.errorbar(x_values, robust_agg_recip_means, yerr=robust_agg_recip_errors, label='Robust Aggregate - reciprocal')
    pp.errorbar(x_values, robust_agg_expo_means, yerr=robust_agg_expo_errors, label='Robust Aggregate - exponential')
    pp.errorbar(x_values, cramer_rao, label='Cramer-Rao Bound')
    
    pp.legend(loc='upper left')
    pp.title('RMS Error against Sensor Variance')
    pp.xlabel('Sensor Variance')
    pp.ylabel('RMS Error')
    pp.show()
