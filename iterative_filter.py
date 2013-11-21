#!/usr/bin/python

import sys
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
    return top / bottom

def compute_next_r(readings, weights):
    # matrix rotation
    instant_readings = [[x[i] for x in readings] for i in range(len(readings[0]))]
    return [aggregate(r, weights) for r in instant_readings]

def sensor_distance(sensor_readings, next_r):
    return sum([(x - r)**2 for x, r in zip(sensor_readings, next_r)])

def compute_d(readings, next_r):
    return [sensor_distance(x, next_r) / len(readings[0]) for x in readings]

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

        if [round(x, 4) for x in r[l]] == [round(y, 4) for y in r[l-1]]:
            converged = True
        l += 1;
    print ('Filter completed in', l, 'rounds')
    return r[l]

def reciprocal(distance):
    if distance:
        return distance**-1
    else:
        return sys.maxsize

def exponential(distance):
    return math.exp(distance * -1)


def iterfilter(readings, discriminant):
    return iterative_filter(readings, len(readings), len(readings[0]), discriminant)

if __name__ == '__main__':

    assert(iterative_filter(intel_X, intel_N, intel_T, reciprocal) == [19.42, 19.4102, 19.42])

    assert(sensor_distance([1,2,3], [1,2,3]) == 0)
    assert(sensor_distance([1,1,1,1], [0,0,0,0]) == 4)

    assert(compute_d([[1,1,1,1]], [0,0,0,0]) == [1])
    assert(compute_d([[1,1,1,1]], [1,1,1,1]) == [0])

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
