#!/usr/bin/python

import random
import math

import robust_aggregate
import readings_generator

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

if __name__ == '__main__':
    truth = 0
    num_times = 100

    num_honest = 20
    honest_variance = 1
    honest_bias = 0
    honest_bias_variance = 0.1
    honest_variances = [honest_variance] * num_honest
    honest_biases = [random.gauss(honest_bias, honest_bias_variance) for i in range(num_honest)]
    honest_readings = readings_generator.readings(honest_biases, honest_variances, num_times, lambda t: truth).tolist()

    num_influenced = 5
    influenced_value = 10000

    influenced_readings = num_influenced * [[influenced_value] * num_times]

    readings = honest_readings + influenced_readings

    estimates = robust_aggregate.estimate(readings, robust_aggregate.reciprocal)
    error = rms_error(estimates, [truth] * len(readings))

    print (error)
