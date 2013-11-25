import random
import matplotlib.pyplot as pyplot
import scipy.stats as stats
from scipy.stats import kstest

__author__ = 'Pierzchalski'


if __name__ == "__main__":
    sample_size = 1000
    gaussian_sample = [random.gauss(0, 1) for i in range(sample_size)]
    cauchy_sample = [stats.cauchy.rvs() for i in range(sample_size)]
    print(kstest(gaussian_sample, 'norm'))
    print(kstest(cauchy_sample, 'norm'))

    fig = pyplot.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.hist(gaussian_sample, bins=100)
    #axes.hist(cauchy_sample, bins=100)
    pyplot.show()