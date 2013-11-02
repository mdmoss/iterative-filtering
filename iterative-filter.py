#!/usr/bin/python

import math

"""
Implementation of Algorithm 1 from [RIBJ 2013]

Input:
    X: Readings of each sensor
    N: Number of sensors
    T: Number of readings for each sensor

Output: Reputation vector r

Note:
    x(a,b): reading of sensor a at time b
    X contains readings of each sensor, indexed by time interval
"""

# Test data
intel_X = [[19.3612, 19.3612, 19.3612],
           [19.42, 19.4102, 19.42],
           [19.0084, 19.0084, 19.0084],
           [18.5674, 18.5478, 17.117],
           [17.95, 21.282, 21.3408],
           [22.153, 21.347, 20.813],
           [18.0088, 18.0088, 21.625],
           [20.4, 20.4098, 19.7924]]
intel_N = 8
intel_T = 3

def aggregate(instant_readings, weights):
    # dot product
    top = sum([r * w for r, w in zip(instant_readings, weights)])
    bottom = sum(weights)
    return round(top / bottom, 4)

assert(aggregate([x[0] for x in intel_X], [1] * intel_N) == 19.3586)
assert(aggregate([x[1] for x in intel_X], [1] * intel_N) == 19.6719)
assert(aggregate([x[2] for x in intel_X], [1] * intel_N) == 19.8097)

def compute_next_r(readings, weights):
    # matrix rotation
    instant_readings = [[x[i] for x in readings] for i in range(len(readings[0]))]
    return [aggregate(r, weights) for r in instant_readings]

assert(compute_next_r(intel_X, [1] * intel_N) == [19.3586, 19.6719, 19.8097])

def sensor_distance(sensor_readings, next_r):
    return sum([(x - r)**2 for x, r in zip(sensor_readings, next_r)])

assert(sensor_distance([1,2,3], [1,2,3]) == 0)
assert(sensor_distance([1,1,1,1], [0,0,0,0]) == 4)

def compute_d(readings, next_r):
    return [sensor_distance(x, next_r) / len(readings[0]) for x in readings]

assert(compute_d([[1,1,1,1]], [0,0,0,0]) == [1])
assert(compute_d([[1,1,1,1]], [1,1,1,1]) == [0])

def compute_next_w(distances, readings, next_r, g):
    return [g(distances[i]) for i in range(len(distances))]

def iterative_filter(x, n, t, g):
    l = 0
    w = [[1] * n]
    r = [[]]
    converged = False
    while not converged:
        r.append(compute_next_r(x, w[l]))
        d = compute_d(x, r[l+1])
        w.append(compute_next_w(d, x, r[l+1], g))

        l += 1;
        if (r[l] == r[l-1]):
            converged = True
    return r[l]

def reciprocal(distance):
    if distance:
        return distance**-1
    else:
        return 1000000000000000000

assert(iterative_filter(intel_X, intel_N, intel_T, reciprocal) == [19.42, 19.4102, 19.42])

if __name__ == '__main__':
    with open('datasets/intel-temp.csv') as f:
        raw = [l.rstrip().split(',') for l in f]
        readings = [[float(r) for r in l] for l in raw]
        
        print ('X:')
        for line in readings:
            print (line)
        print ('N: {}'.format(len(readings)))
        print ('T: {}'.format(len(readings[0])))
        print ()
