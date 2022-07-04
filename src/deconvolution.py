import numpy as np
import matplotlib.pyplot as plt
import src.plotting_function as plotting

input_directory = '/home/melisamc/Documentos/acc_network_analysis/data/traces/'
output_directory = '/home/melisamc/Documentos/acc_network_analysis/data/binary/'
figure_directory = '/home/melisamc/Documentos/acc_network_analysis/figures/'

file_name = 'nike_calcium_trace.npy'
#file_name = 'havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces_accepted.npy'
srate = 30
calcium_trace = np.load(input_directory + file_name)

plotting.plotting_activty(calcium_trace, srate, figure_directory + 'activity_characterization.png')

calcium_mean = np.mean(calcium_trace, axis=1)
calcium_std = np.std(calcium_trace, axis=1)
n_std = 1
spikes_threshold = calcium_mean + n_std * calcium_std

binary_signal = np.zeros_like(calcium_trace)
for neuron in range(calcium_trace.shape[0]):
    binary_signal[neuron, np.where(calcium_trace[neuron,:] > spikes_threshold[neuron])[0]] = 1

binary_file = 'binary_threshold_1std.np'
np.save(binary_file,output_directory + binary_file)

for neuron_ID in range(0,30):
#neuron_ID = 1
    plotting.plotting_mean_deconvolution(calcium_trace, neuron_ID, srate, spikes_threshold[neuron_ID], figure_directory + 'neuron_threshold_'+str(neuron_ID)+'.png')

plotting.plotting_activty(binary_signal, srate, figure_directory + 'binary_activity_characterization.png')

firing_rate = np.sum(binary_signal,axis = 1)*srate / binary_signal.shape[1]
a,b = np.histogram(firing_rate)

figure, axes = plt.subplots()

axes.hist(firing_rate,20, color = 'b')
axes.set_xlabel('Firing Rate (Hz)', fontsize = 15)
axes.set_ylabel('Number of cells', fontsize = 15)

figure.savefig(figure_directory + 'firing_rate_histogram.png')
plt.show()