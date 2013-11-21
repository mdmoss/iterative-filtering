#!/usr/bin/env python

'''Implementation of RobustAggregate algorithm from [Rezvani, Ignjatovic, Bertino, Jha 2013]'''

import math

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

    filtering_result = iterfilter(readings, discriminant)
    
    trustworthy_readings = collusion_detection.partition(readings, filtering_result)

    second_bias_est = bias_estimate(trustworthy_readings)
    second_variance_est = variance_estimate(trustworthy_readings, second_bias_est)
    final_estimate = mle.estimate(trustworthy_readings, second_bias_est, second_variance_est)

    return final_estimate

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

if __name__ == '__main__':

    num_sensors = 20
    num_times = 100

    biases = [0] * 20
    variances = [1] * 20
    def truth(t):
        return 0

    readings = readings_generator.readings(biases, variances, num_times, truth)
    print (rms_error(estimate(readings, reciprocal), [0] * num_sensors))
