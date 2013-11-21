#!/usr/bin/env python

'''Implementation of RobustAggregate algorithm from [Rezvani, Ignjatovic, Bertino, Jha 2013]'''

from bias_estimate import bias_estimate
from variance_estimate import variance_estimate
import mle
from iterative_filter import iterfilter
import collusion_detection

reciprocal = iterative_filter.reciprocal
exponential = iterative_filter.exponential

def estimate(readings, discriminant):
    first_bias_est = bias_estimate(readings)
    first_varance_est = variance_estimate(readings, first_bias_est)
    first_estimate = mle.estimate(readings, first_bias_est, first_variance_est)

    filtering_result = iterfilter(readings, discriminant)
    
    trustworthy_readings = collusion_detection.partition(readings, filtering_result)

    second_bias_est = bias_estimate(trustworthy_readings)
    second_varance_est = variance_estimate(trustworthy_readings, second_bias_est)
    final_estimate = mle.estimate(trustworthy_readings, second_bias_est, second_variance_est)

    return final_estimate
