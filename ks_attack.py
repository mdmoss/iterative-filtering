from matplotlib import pyplot
from attacks import readings_ks_attack, readings_simple_attack, readings_sophisticated_attack
from iterative_filter import exponential
import readings_generator
from robust_aggregate import rms_error
import robust_aggregate

__author__ = 'Edward'

if __name__ == "__main__":
    total_sensors = 20
    max_num_colluders = 10
    num_times = 4
    true_value = lambda t: t
    biases = [0] * total_sensors
    variances = [2] * total_sensors
    colluder_bias = 10


    def test_trio(num_colluders):
        num_legit_sensors = total_sensors - num_colluders
        legitimate_readings = readings_generator.readings(biases[:num_legit_sensors], variances[:num_legit_sensors],
                                                          num_times, true_value)

        def log(f, *args):
            print("processing {} with {} colluders".format(f, num_colluders))
            return f(*args)

        return tuple([
            rms_error(robust_aggregate.estimate(log(f, legitimate_readings,
                                                    true_value,
                                                    num_colluders,
                                                    colluder_bias),
                                                exponential), [true_value(t) for t in range(num_times)])
            for f in (readings_simple_attack, readings_sophisticated_attack, readings_ks_attack)])

    simple_RMSE, sophisticated_RMSE, sk_RMSE = zip(*[
        test_trio(num_colluders) for num_colluders in range(2, max_num_colluders)
    ])

    fig = pyplot.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_title('RMS Error under Attacks')
    axes.plot(simple_RMSE)
    axes.plot(sophisticated_RMSE)
    axes.plot(sk_RMSE)
    axes.legend(['Simple', 'Sophisticated', 'SK'])
    axes.set_xlabel('Number of colluding sensors')
    axes.set_ylabel('RMS Error')

    pyplot.show()

    #print(s)
    #print(len(s))