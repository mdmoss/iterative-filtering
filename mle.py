#!/usr/bin/python

import math

def chunk(i, variances, biases, lambd=0.01):
    return 1 / (variances[i] + biases[i]**2 + lambd)

def weighting(i, variances, biases, lambd=0.01):
    return chunk(i, variances, biases, lambd) / sum([chunk(j, variances, biases, lambd) for j in range(len(variances))])

assert(weighting(0, [0], [0], 1) == 1)

def weight_vector(variances, biases, lambd=0.01):
    return [weighting(i, variances, biases, lambd) for i in range(len(variances))]

def estimate(readings, variances, biases):
    assert(len(readings) == len(variances) == len(biases))
    return [sum([weighting(i, variances, biases)*readings[i][t] for i in range(len(readings))]) for t in range(len(readings[0]))]
     
if __name__ == '__main__':
    assert(estimate([[1]], [0], [0]) != 0)
    assert(chunk(0, [0], [0], 1) == 1)

