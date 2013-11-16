#!/usr/bin/python

# Readings format: Array of sensors containing array of times
# [[S1T1, S1T2], [S2T1, S2T2]]

def delta(i, j, readings):
    assert(len(readings[i]) == len(readings[j]))
    return sum([a - b for (a, b) in zip(readings[i], readings[j])]) / len(readings[i])

assert(delta(0, 1, [[0, 0], [0, 0]]) == 0)
assert(delta(0, 1, [[1, 1], [0, 0]]) == 1)
