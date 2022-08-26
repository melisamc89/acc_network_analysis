import numpy as np
import matplotlib.pyplot as plt


def plotting_activty(calcium_trace, srate, figure_name):

    calcium_mean = np.sum(calcium_trace, axis=1) * srate / calcium_trace.shape[1]
    calcium_std = np.std(calcium_trace, axis=1)

    neurons_id = np.arange(0, calcium_trace.shape[0])
    figure, axes = plt.subplots(1, 2)
    axes[0].plot(neurons_id, calcium_mean, color='b')
    axes[0].fill_between(neurons_id, calcium_mean - calcium_std, calcium_mean + calcium_std, color='b', alpha=0.3)
    axes[0].set_ylabel('Activity Rate (a.u./s)', fontsize=15)
    axes[0].set_xlabel('Units', fontsize=15)
    axes[1].scatter(calcium_mean, calcium_std)
    axes[1].set_xlabel('Activity Rate (a.u./s)', fontsize=15)
    axes[1].set_ylabel('STD Activity Rate (a.u./s)', fontsize=15)

    figure.set_size_inches([18, 5])
    figure.savefig(figure_name)
    plt.show()

    return

def plotting_mean_deconvolution(calcium_trace, neuron_ID, srate, threshold, figure_name):

    figure, axes = plt.subplots()

    time = np.arange(0,calcium_trace.shape[1])/srate
    axes.plot(time , calcium_trace[neuron_ID,:] , c = 'k')
    axes.hlines(threshold, 0, calcium_trace.shape[1]/srate, color= 'r')
    axes.set_xlabel('Time (s)',fontsize = 15)
    axes.set_ylabel('Activity',fontsize = 15)

    figure.set_size_inches([10,5])
    figure.savefig(figure_name)
    plt.show()
    return