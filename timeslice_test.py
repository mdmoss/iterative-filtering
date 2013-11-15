#!/usr/bin/python

import math
import random
import sys

import iterative_filter

TRUTH = 0
VARIANCE = 1

def rms_error(estimates, truths):
    return math.sqrt(sum([(e - t)**2 for (e, t) in zip(estimates, truths)]) / len(estimates))

assert (rms_error([0], [0]) == 0)
assert (rms_error([2, 2], [0, 0]) == 2)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ('Usage: timeslice_test num_sensors num_times num_repetitions')
        sys.exit()

    num_sensors = int(sys.argv[1])
    num_times = int(sys.argv[2])
    num_repetitions = int(sys.argv[3])

    print ('{} sensors at {} times, {} repetitions'.format(num_sensors, num_times, num_repetitions))
    error_calcs = [[] for i in range(num_times)]
    truth = [TRUTH * num_sensors for i in range(num_times)]

    for i in range(num_repetitions):
        readings = [[random.gauss(TRUTH, VARIANCE) for i in range(num_sensors)] for j in range(num_times)]

        for j in range(num_times):
            iterative_result = iterative_filter.by_time(readings[0:j+1], iterative_filter.exponential)
            error = rms_error(iterative_result, [TRUTH] * len(iterative_result))
            error_calcs[j].append(error)

    error_averages = [sum(x)/len(x) for x in error_calcs]
    for i in range(len(error_averages)):
        print ('{}          {}'.format(i + 1, round(error_averages[i], 2)))
