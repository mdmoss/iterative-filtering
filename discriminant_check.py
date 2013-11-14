#!/usr/bin/python

import math
import random

import iterative_filter

'''
Exploration of the accuracy of different discriminant functions
'''

def gen_readings(num_sensors, num_readings, gen_value):
    return [[gen_value() for y in range(num_readings)] for x in range(num_sensors)]

if __name__ == '__main__':
    print("Running simulation with nine good sensors and one poor sensor, t=3")
    

    for i in range(100):
        truth = round(random.uniform(0, 100), 4)
        print("Truth value: {}".format(truth))

        good = gen_readings(9, 10, lambda: random.gauss(truth, 1))
        poor = gen_readings(1, 10, lambda: random.gauss(truth, 10))
        readings = good + poor

        print (iterative_filter.iterfilter(readings, iterative_filter.reciprocal))
