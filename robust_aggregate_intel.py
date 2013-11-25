#!/usr/bin/env python

'''Implementation of RobustAggregate algorithm from [Rezvani, Ignjatovic, Bertino, Jha 2013]'''

import math
import sys

from scipy.stats import bayes_mvs
import matplotlib.pyplot as pp

from bias_estimate import bias_estimate
from variance_estimate import variance_estimate
import mle
from iterative_filter import iterfilter, reciprocal, exponential, iterative_filter
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
    if len(sys.argv) == 1:
        data = 'datasets/intel-temp.csv'
    else:
        data = sys.argv[1]
    with open(data) as f:
        raw = [l.rstrip().split(',') for l in f]
        readings = [[float(r) for r in l] for l in raw]
        
        print ('X:')
        for line in readings:
            print (line)
        print ('N: {}'.format(len(readings)))
        print ('T: {}'.format(len(readings[0])))
        result = iterative_filter(readings, len(readings), len(readings[0]), reciprocal)
        print ('reciprocal: {}'.format(result))
        result = [round(x, 4) for x in iterative_filter(readings, len(readings), len(readings[0]), exponential)]
        print ('exponential: {}'.format(result))
        result = estimate(readings, reciprocal)
        print ('RobustAggregate-reciprocal: {}'.format(result))
        result = estimate(readings, exponential)
        print ('RobustAggregate-exponential: {}'.format(result))
