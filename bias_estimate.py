#!/usr/bin/python

from scipy.optimize import minimize

# Readings format: Array of sensors containing array of times
# [[S1T1, S1T2], [S2T1, S2T2]]

# Danger Will Robinson: Lack of thread safety ahead!
_readings = []

def delta(i, j):
    assert(len(_readings[i]) == len(_readings[j]))
    return sum([a - b for (a, b) in zip(_readings[i], _readings[j])]) / len(_readings[i])

def per_sensor_bias(i, bi, j, bj):
    return (bi - bj - delta(i, j))**2

def bias_estimator(x):
    total = 0
    for i in range(len(x)):
        for j in range(i):
            total += per_sensor_bias(i, x[i], j, x[j]) 
    return total

def bias_constraint(x):
    return (sum(x) == 0)

bias_constraint = {
    'type': 'eq',
    'fun': bias_constraint,
}

def estimate(readings):
    global _readings
    _readings = readings 
    res = minimize(bias_estimator, [0]*len(readings), method='SLSQP', constraints=bias_constraint, options={'disp':True})
    return res.x

print (estimate([[1,0], [0, 1]]))
