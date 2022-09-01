import numpy as np
import matplotlib.pyplot as plt
import pickle
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF as load_CNMF

input_directory = '/home/melisamc/Documentos/acc_network_analysis/data/traces/'
output_directory = '/home/melisamc/Documentos/acc_network_analysis/data/'
figure_directory = '/home/melisamc/Documentos/acc_network_analysis/figures/tunning_curves_calcium/'

filter_path = '/home/melisamc/Documentos/acc_network_analysis/data/filter/'
filter_file = 'mouse_3_year_2022_month_3_day_23_example_1__v.1.0.1_gSig_7.npy'
contour_file = 'mouse_3_year_2022_month_3_day_23_example_0__v.1.1.1.hdf5'

cn_filter = np.load(filter_path + filter_file)
cnm = load_CNMF(filter_path + contour_file)

colors_list = ['b','r','g','orange','magenta','cyan']

file_name = 'nike_calcium_trace.npy'
#file_name = 'havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces_accepted.npy'
srate = 30
calcium_trace = np.load(input_directory + file_name)
calcium_mean = np.mean(calcium_trace, axis=1)
#selected_order = np.argsort(calcium_mean)
#calcium_trace = calcium_trace[selected_order[:50],:]
calcium_mean = np.mean(calcium_trace, axis=1)
calcium_std = np.std(calcium_trace, axis=1)
n_std = 1
spikes_threshold = calcium_mean + n_std * calcium_std

binary_signal = np.zeros_like(calcium_trace)
for neuron in range(calcium_trace.shape[0]):
    binary_signal[neuron, np.where(calcium_trace[neuron,:] > spikes_threshold[neuron])[0]] = 1

timestamps_filename = '/home/melisamc/Documentos/acc_network_analysis/data/20220323-143325_213_output.pickle'
with open(timestamps_filename, 'rb') as f:
    timestamps = pickle.load(f)
sounds_list = []
sounds_list.append(timestamps['sound1'])
sounds_list.append(timestamps['sound2'])
sounds_list.append(timestamps['sound3'])
sounds_list.append(timestamps['sound4'])
sounds_list.append(timestamps['sound5'])
sounds_list.append(timestamps['sound6'])

stim_length = 6
audio_matrix_list = []
iti_matrix_list = []
mean_sound_activity_evolution_list = []
mean_iti_activity_evolution_list = []
C_0 = binary_signal.copy()
n_neurons = C_0.shape[0]
timeline = [0, 0]

for sound_index in range(len(sounds_list)):
    # create matrix to contain mean activity of all neurons over trials
    sound_matrix = np.zeros((n_neurons, int(stim_length * srate*1.5)))
    audio_matrix_list.append(sound_matrix)
    # create matrix to contrain mean activity evolution over trials
    sound_mean = np.zeros((n_neurons, len(sounds_list[sound_index])))
    mean_sound_activity_evolution_list.append(sound_mean)

traces = calcium_trace.copy()
### zscored traces
traces_zscored = traces - traces.mean(axis=1, keepdims=True) / traces.std(axis=1, keepdims=True)
### normed traces
traces_normed = (traces_zscored - traces_zscored.min(axis=1, keepdims=True)) / (
            traces_zscored.max(axis=1, keepdims=True) - traces_zscored.min(axis=1, keepdims=True))
C_final = traces_normed

#C_final = binary_signal.copy()

for n in range(n_neurons):
    print('cell number = ', n)
    # figure, axes = plt.subplots(1,4)
    # create a figure for every neuron
    figure = plt.figure()
    gs = plt.GridSpec(15, 48)
    axes = figure.add_subplot(gs[0:3, 0:12])
    axes.imshow(cn_filter, cmap='gray')
    # coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape,
    #                                                   0.2, 'max')
    # counter = 0
    # for c in coordinates:
    #     if counter == n:
    #         v = c['coordinates']
    #         c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
    #                      np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    #         axes.plot(*v.T, c='r')
    #     counter = counter + 1
    # axes.set_xlabel('Pixel', fontsize=20)
    # axes.set_ylabel('Pixel', fontsize=20)

    axes = figure.add_subplot(gs[0:3, 10:32])
    axes.plot(np.arange(0, len(C_0[n])) / srate, C_final[n], c='k')
    axes.set_xlabel('time [s]', fontsize=25)
    axes.set_yticks([])
    axes.set_ylabel('Actvivity', fontsize=25)

    for sound_index in range(len(sounds_list)):
        for j in range(0, sounds_list[sound_index].shape[0]):
            sound_onset = int(sounds_list[sound_index][j] + timeline[1] - stim_length * srate / 2)
            sound_end = int(sound_onset + stim_length * srate)
            axes.plot(np.arange(sound_onset, sound_end) / srate, C_final[n][sound_onset:sound_end],
                      c=colors_list[sound_index])

    axes_0 = figure.add_subplot(gs[0:3, 37:48])
    # axes_1 = figure.add_subplot(gs[0:3, 40:48])

    for sound_index in range(len(sounds_list)):  # for everysound
        ### create matrix con trials
        aux_matrix_sound = np.zeros((len(sounds_list[sound_index]), int(stim_length * srate *1.5)))
        aux_time = np.arange(0, stim_length, 1 / srate)
        for trial in range(0, len(sounds_list[sound_index])):
            sound_onset = int(sounds_list[sound_index][trial] + timeline[1] - stim_length * srate / 2)
            aux_matrix_sound[trial, :] = C_final[n][sound_onset:sound_onset + int(stim_length * srate *1.5)]
        # mean_activity = np.mean(aux_matrix_sound,axis = 0)

        ### z-score and normalize traces
        # aux_matrix_zs = (aux_matrix_sound - aux_matrix_sound.mean(axis = 1, keepdims = True))/ aux_matrix_sound.std(axis = 1,keepdims=True)
        mean_sound_activity_evolution_list[sound_index][n, :] = np.mean(aux_matrix_sound, axis=1)
        # matrix1_norm = (aux_matrix_zs - aux_matrix_zs.min(axis = 1,keepdims = True)) / (aux_matrix_zs.max(axis=1, keepdims = True) - aux_matrix_zs.min(axis = 1,keepdims= True))
        final_matrix = aux_matrix_sound

        axes = figure.add_subplot(gs[13:15, sound_index * 8: sound_index * 8 + 6])
        mean_activity = np.mean(final_matrix, axis=0)
        std_activity = np.std(final_matrix, axis=0) /np.sqrt(final_matrix.shape[0])
        ###save the mean for population analysis
        audio_matrix_list[sound_index][n, :] = mean_activity
        temporal_var = np.arange(0, len(mean_activity)) / srate - stim_length / 2
        axes.fill_between(temporal_var, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1,
                          color=colors_list[sound_index])
        axes.plot(temporal_var, mean_activity, color=colors_list[sound_index], linewidth=2)
        axes.vlines(0, -0.05, 0.15, color='k')

        # axes.set_yticks([])
        axes.set_ylim([-0.05, 0.15])
        if sound_index == 0:
            axes.set_ylabel('Mean Activity', fontsize=15)
        axes.set_xlabel('time [s]', fontsize=15)

        axes_0.fill_between(temporal_var, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1,
                            color=colors_list[sound_index])
        axes_0.plot(temporal_var, mean_activity, color=colors_list[sound_index], linewidth=2)
        # axes.set_yticks([])
        axes_0.vlines(0, -0.05, 0.15, color='k')
        axes_0.set_ylim([-0.05, 0.15])
        axes_0.set_ylabel('Mean Activity', fontsize=15)
        axes_0.set_xlabel('time [s]', fontsize=15)

        axes = figure.add_subplot(gs[5:12, sound_index * 8: sound_index * 8 + 6])
        final_matrix[0, :] += final_matrix[0].min()
        axes.plot(temporal_var, final_matrix[0], c='k')
        for j in range(1, len(sounds_list[sound_index])):
            final_matrix[j] += final_matrix[j].min() + 0.1 * j  # + final_matrix[:j].max()
            axes.plot(temporal_var, final_matrix[j], c='k')
        axes.set_yticks([])
        axes.set_title('SOUND = ' + f'{sound_index + 1}', fontsize=12)
        #    axes.set_xlabel('t [s]', fontsize=10)
        if sound_index == 0:
            axes.set_ylabel('Trials', fontsize=20)

    figure.set_size_inches([25, 10])
    figure.patch.set_facecolor('white')
    figure.savefig(figure_directory + 'traces_binary_' + f' {n}' + '.png')

    plt.close()
