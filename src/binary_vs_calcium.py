import numpy as np
import matplotlib.pyplot as plt
import src.plotting_function as plotting
from sklearn.decomposition import PCA
import pickle

input_directory = '/home/melisamc/Documentos/acc_network_analysis/data/traces/'
output_directory = '/home/melisamc/Documentos/acc_network_analysis/data/binary/'
figure_directory = '/home/melisamc/Documentos/acc_network_analysis/figures/'

colors_list = ['b','r','g','orange','magenta','cyan']

file_name = 'nike_calcium_trace.npy'
#file_name = 'havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces_accepted.npy'
srate = 30
calcium_trace = np.load(input_directory + file_name)
calcium_mean = np.mean(calcium_trace, axis=1)
selected_order = np.argsort(calcium_mean)
calcium_trace = calcium_trace[selected_order[:50],:]
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

n_seconds = 12
ITI = False
VMAX_1 = 75
VMAX_3 = 150
VMAX_5 = 150

data_matrix = np.zeros((calcium_trace.shape[0],18,n_seconds*30,6))
data_matrix1_odd = np.zeros((calcium_trace.shape[0],9,n_seconds*30,6))
data_matrix1_even = np.zeros((calcium_trace.shape[0],9,n_seconds*30,6))

data_matrix2 = np.zeros((calcium_trace.shape[0],18*n_seconds*30,6))
data_matrix_odd = np.zeros((calcium_trace.shape[0],9*n_seconds*30,6))
data_matrix_even = np.zeros((calcium_trace.shape[0],9*n_seconds*30,6))

for sound in range(len(sounds_list)):
    for i in range(18):
        time1 = sounds_list[sound][i]
        time2 = time1 + n_seconds*30
        if ITI:
            time1 = time1 + 4*30
            time2 = time1 + n_seconds*30
        data_matrix[:,i,:,sound] = binary_signal[:,time1:time2]
        data_matrix2[:,i*n_seconds*30:(i+1)*n_seconds*30,sound] = binary_signal[:,time1:time2]
        if i%2 == 0:
            data_matrix_even[:,int(i/2)*n_seconds*30:(int(i/2)+1)*n_seconds*30,sound]  = binary_signal[:,time1:time2]
            data_matrix1_even[:, int(i/2), :, sound] = binary_signal[:,time1:time2]
        else:
            data_matrix_odd[:,int(i/2)*n_seconds*30:(int(i/2)+1)*n_seconds*30,sound]  = binary_signal[:,time1:time2]
            data_matrix1_odd[:, int(i/2), :, sound] = binary_signal[:,time1:time2]

signal_matrix = np.mean(data_matrix,axis = 1)
signal_matrix_odd = np.mean(data_matrix1_odd, axis = 1)
signal_matrix_even = np.mean(data_matrix1_even, axis = 1)

signal_vector = np.zeros((calcium_trace.shape[0],18*n_seconds*30,6))
signal_vector_odd = np.zeros((calcium_trace.shape[0],9*n_seconds*30,6))
signal_vector_even = np.zeros((calcium_trace.shape[0],9*n_seconds*30,6))

for sound in range(len(sounds_list)):
    for i in range(18):
        signal_vector[:,i*n_seconds*30:(i+1)*n_seconds*30,sound] = signal_matrix[:,:,sound]
        if i%2 == 0:
            signal_vector_even[:, int(i/2) * n_seconds * 30:(int(i/2) + 1) * n_seconds * 30, sound] = signal_matrix_even[:, :, sound]
        else:
            signal_vector_odd[:, int(i/2) * n_seconds * 30:(int(i/2) + 1) * n_seconds * 30, sound] = signal_matrix_odd[:, :, sound]

pop_vector = np.mean(signal_matrix,axis = 1)
pop_vector_odd =  np.mean(signal_matrix_odd,axis = 1)
pop_vector_even =  np.mean(signal_matrix_even,axis = 1)

##### with calcium traces

data_matrix_calcium = np.zeros((calcium_trace.shape[0],18,n_seconds*30,6))
data_matrix_calcium1_odd = np.zeros((calcium_trace.shape[0],9,n_seconds*30,6))
data_matrix_calcium1_even = np.zeros((calcium_trace.shape[0],9,n_seconds*30,6))

data_matrix_calcium2 = np.zeros((calcium_trace.shape[0],18*n_seconds*30,6))
data_matrix_calcium_odd = np.zeros((calcium_trace.shape[0],9*n_seconds*30,6))
data_matrix_calcium_even = np.zeros((calcium_trace.shape[0],9*n_seconds*30,6))

for sound in range(len(sounds_list)):
    for i in range(18):
        time1 = sounds_list[sound][i]
        time2 = time1 + n_seconds*30
        if ITI:
            time1 = time1 + 2*30
            time2 = time1 + n_seconds*30
        data_matrix_calcium[:,i,:,sound] = calcium_trace[:,time1:time2]
        data_matrix_calcium2[:,i*n_seconds*30:(i+1)*n_seconds*30,sound] = calcium_trace[:,time1:time2]
        if i%2 == 0:
            data_matrix_calcium_even[:,int(i/2)*n_seconds*30:(int(i/2)+1)*n_seconds*30,sound]  = calcium_trace[:,time1:time2]
            data_matrix_calcium1_even[:, int(i/2), :, sound] = calcium_trace[:, time1:time2]
        else:
            data_matrix_calcium_odd[:,int(i/2)*n_seconds*30:(int(i/2)+1)*n_seconds*30,sound]  = calcium_trace[:,time1:time2]
            data_matrix_calcium1_odd[:, int(i/2), :, sound] = calcium_trace[:, time1:time2]


signal_matrix_calcium = np.mean(data_matrix_calcium,axis = 1)
signal_matrix_calcium_even = np.mean(data_matrix_calcium1_even,axis = 1)
signal_matrix_calcium_odd = np.mean(data_matrix_calcium1_odd,axis = 1)

signal_vector_calcium = np.zeros((calcium_trace.shape[0],18*n_seconds*30,6))
signal_vector_calcium_even = np.zeros((calcium_trace.shape[0],9*n_seconds*30,6))
signal_vector_calcium_odd = np.zeros((calcium_trace.shape[0],9*n_seconds*30,6))

for sound in range(len(sounds_list)):
    for i in range(18):
        signal_vector_calcium[:,i*n_seconds*30:(i+1)*n_seconds*30,sound] = signal_matrix_calcium[:,:,sound]
        if i%2 == 0:
            signal_vector_calcium_even[:, int(i/2) * n_seconds * 30:(int(i/2) + 1) * n_seconds * 30, sound] = signal_matrix_calcium_even[:, :, sound]
        else:
            signal_vector_calcium_odd[:, int(i/2) * n_seconds * 30:(int(i/2) + 1) * n_seconds * 30, sound] = signal_matrix_calcium_odd[:, :, sound]


pop_vector_calcium = np.mean(signal_matrix_calcium,axis = 1)
pop_vector_calcium_even = np.mean(signal_matrix_calcium_even,axis = 1)
pop_vector_calcium_odd = np.mean(signal_matrix_calcium_odd,axis = 1)

order_calcium = np.argsort(pop_vector_calcium[:,0])[::-1]
order= np.argsort(pop_vector[:,0])[::-1]

# ##### POPULATION VECTOR FIGURE #####
figure = plt.figure()
gs = plt.GridSpec(6, 2)
axes = figure.add_subplot(gs[0, 0])
for sound in range(6):
    value = pop_vector_calcium[order_calcium,sound] / np.max(pop_vector_calcium)
    axes.plot(value,color = colors_list[sound])

axes.set_xlabel('Neurons', fontsize = 15)
axes.set_ylabel('Act [a.u]', fontsize = 15)
axes.set_title('Population Vector', fontsize = 20)
axes = figure.add_subplot(gs[0, 1])
axes.set_xlabel('Neurons', fontsize = 15)
axes.set_ylabel('Act [a.u]', fontsize = 15)
axes.set_title('Population Vector', fontsize = 20)
for sound in range(6):
    value = pop_vector[order, sound] / np.max(pop_vector)
    axes.plot(value, color=colors_list[sound])
axes = figure.add_subplot(gs[2:5, 0])
axes.set_title('PopVector Correlation', fontsize = 20)
x_label_list = ['Control', 'Fear1', 'Fear2', 'Fear3','Fear4','Fear5']
axes.set_xticks(np.arange(0,6))
axes.set_xticklabels(axes.get_xticks(), rotation = 45)
axes.set_xticklabels(x_label_list)
axes.set_yticks(np.arange(0,6))
axes.set_yticklabels(x_label_list)
x = axes.imshow(np.corrcoef(pop_vector_calcium.T,pop_vector_calcium.T)[0:6,0:6])
figure.colorbar(x)
axes = figure.add_subplot(gs[2:5, 1])
axes.set_title('PopVector Correlation', fontsize = 20)
x_label_list = ['Control', 'Fear1', 'Fear2', 'Fear3','Fear4','Fear5']
axes.set_xticks(np.arange(0,6))
axes.set_xticklabels(axes.get_xticks(), rotation = 45)
axes.set_xticklabels(x_label_list)
axes.set_yticks(np.arange(0,6))
axes.set_yticklabels(x_label_list)
y = axes.imshow(np.corrcoef(pop_vector.T,pop_vector.T)[0:6,0:6])
figure.colorbar(y)
figure.suptitle('Calcium Data vs Binary Data', fontsize = 25)
figure.set_size_inches([10,7])
if ITI:
    figure_name = 'Nike_calcium_vs_binary_ITI_'+str(n_seconds)+'sec.png'
else:
    figure_name = 'Nike_calcium_vs_binary_'+str(n_seconds)+'sec.png'
figure.savefig(figure_directory + figure_name)
plt.show()
#
# #### CORRELATION FIGURES #####
#
figure, axes = plt.subplots(6,3)
order = np.argsort(pop_vector_calcium[:,0])[::-1]

for sound in range(len(sounds_list)):
    response = data_matrix_calcium2[order_calcium,:,sound]
    signal_response = signal_matrix_calcium[order_calcium,:,sound]
    noise_response = response - signal_vector_calcium[order_calcium,:,sound]
    response_correlation = np.corrcoef(response,response)[0:response.shape[0],0:response.shape[0]]
    signal_correlation = np.corrcoef(signal_response,signal_response)[0:response.shape[0],0:response.shape[0]]
    noise_correlation = np.corrcoef(noise_response,noise_response)[0:response.shape[0],0:response.shape[0]]

    x1 = axes[sound, 0].imshow(np.abs(response_correlation), vmin=0, vmax=1)
    x2 = axes[sound, 1].imshow(np.abs(signal_correlation), vmin=0, vmax=1)
    x3 = axes[sound, 2].imshow(np.abs(noise_correlation), vmin=0, vmax=1)

    if sound == 0:
        axes[0,0].set_title('Complete Response', fontsize = 15)
        axes[0,1].set_title('Signal', fontsize=15)
        axes[0,2].set_title('Noise', fontsize=15)

figure.set_size_inches([7,14])
if ITI:
    figure_name = 'Nike_signal_noise_correlation_ITI_'+str(n_seconds)+'sec_calcium.png'
else:
    figure_name = 'Nike_signal_noise_correlation_'+str(n_seconds)+'sec_calcium.png'
figure.savefig(figure_directory + figure_name)
plt.show()

figure, axes = plt.subplots(6,3)

for sound in range(len(sounds_list)):
    response = data_matrix2[order,:,sound]
    signal_response = signal_matrix[order,:,sound]
    noise_response = response - signal_vector[order,:,sound]
    response_correlation = np.corrcoef(response,response)[0:response.shape[0],0:response.shape[0]]
    signal_correlation = np.corrcoef(signal_response,signal_response)[0:response.shape[0],0:response.shape[0]]
    noise_correlation = np.corrcoef(noise_response,noise_response)[0:response.shape[0],0:response.shape[0]]

    x1 = axes[sound, 0].imshow(np.abs(response_correlation), vmin=0, vmax=1)
    x2 = axes[sound, 1].imshow(np.abs(signal_correlation), vmin=0, vmax=1)
    x3 = axes[sound, 2].imshow(np.abs(noise_correlation), vmin=0, vmax=1)

    if sound == 0:
        axes[0,0].set_title('Complete Response', fontsize = 15)
        axes[0,1].set_title('Signal', fontsize=15)
        axes[0,2].set_title('Noise', fontsize=15)

figure.set_size_inches([7,14])
if ITI:
    figure_name = 'Nike_signal_noise_correlation_ITI_'+str(n_seconds)+'sec_binary.png'
else:
    figure_name = 'Nike_signal_noise_correlation_'+str(n_seconds)+'sec_binary.png'
figure.savefig(figure_directory + figure_name)
plt.show()

##### CORRELATION OF CORRELATION FIGURES ################3

n_neurons = len(order)
signal_correlation_matrix = np.zeros((n_neurons*n_neurons,6))
signal_correlation_matrix_even = np.zeros((n_neurons*n_neurons,6))
signal_correlation_matrix_odd = np.zeros((n_neurons*n_neurons,6))
noise_correlation_matrix = np.zeros((n_neurons*n_neurons,6))
noise_correlation_matrix_even = np.zeros((n_neurons*n_neurons,6))
noise_correlation_matrix_odd = np.zeros((n_neurons*n_neurons,6))
response_correlation_matrix = np.zeros((n_neurons*n_neurons,6))
response_correlation_matrix_even = np.zeros((n_neurons*n_neurons,6))
response_correlation_matrix_odd = np.zeros((n_neurons*n_neurons,6))

BINARY = True
for sound in range(len(sounds_list)):
    if BINARY == True:
        response = data_matrix2[order_calcium, :, sound]
        response_even = data_matrix_even[order_calcium, :, sound]
        response_odd = data_matrix_odd[order_calcium, :, sound]

        signal_response = signal_matrix[order_calcium,:,sound]
        signal_response_even = signal_matrix_even[order_calcium,:,sound]
        signal_response_odd = signal_matrix_odd[order_calcium,:,sound]

        noise_response = response - signal_vector[order_calcium,:,sound]
        noise_response_even = response_even - signal_vector_even[order_calcium,:,sound]
        noise_response_odd = response_odd - signal_vector_odd[order_calcium,:,sound]
    else:
        response = data_matrix_calcium2[order_calcium,:,sound]
        response_even = data_matrix_calcium_even[order_calcium,:,sound]
        response_odd = data_matrix_calcium_odd[order_calcium,:,sound]

        signal_response = signal_matrix_calcium[order_calcium,:,sound]
        signal_response_even = signal_matrix_calcium_even[order_calcium,:,sound]
        signal_response_odd = signal_matrix_calcium_odd[order_calcium,:,sound]

        noise_response = response - signal_vector_calcium[order_calcium,:,sound]
        noise_response_even = response_even -signal_vector_calcium_even[order_calcium,:,sound]
        noise_response_odd = response_odd - signal_vector_calcium_odd[order_calcium,:,sound]

    response_correlation = np.corrcoef(response,response)[0:response.shape[0],0:response.shape[0]]
    response_correlation_even = np.corrcoef(response_even,response_even)[0:response.shape[0],0:response.shape[0]]
    response_correlation_odd = np.corrcoef(response_odd,response_odd)[0:response.shape[0],0:response.shape[0]]

    signal_correlation = np.corrcoef(signal_response,signal_response)[0:response.shape[0],0:response.shape[0]]
    signal_correlation_even = np.corrcoef(signal_response_even,signal_response_even)[0:response.shape[0],0:response.shape[0]]
    signal_correlation_odd = np.corrcoef(signal_response_odd,signal_response_odd)[0:response.shape[0],0:response.shape[0]]

    noise_correlation = np.corrcoef(noise_response,noise_response)[0:response.shape[0],0:response.shape[0]]
    noise_correlation_even = np.corrcoef(noise_response_even,noise_response_even)[0:response.shape[0],0:response.shape[0]]
    noise_correlation_odd = np.corrcoef(noise_response_odd,noise_response_odd)[0:response.shape[0],0:response.shape[0]]

    signal_correlation_matrix[:,sound] = np.nan_to_num(signal_correlation.flatten(),0)
    signal_correlation_matrix_even[:,sound] = np.nan_to_num(signal_correlation_even.flatten(),0)
    signal_correlation_matrix_odd[:,sound] = np.nan_to_num(signal_correlation_odd.flatten(),0)

    noise_correlation_matrix[:,sound] = np.nan_to_num(noise_correlation.flatten(),0)
    noise_correlation_matrix_even[:,sound] = np.nan_to_num(noise_correlation_even.flatten(),0)
    noise_correlation_matrix_odd[:,sound] = np.nan_to_num(noise_correlation_odd.flatten(),0)

    response_correlation_matrix_even[:,sound] = np.nan_to_num(response_correlation_even.flatten(),0)
    response_correlation_matrix_odd[:,sound] = np.nan_to_num(response_correlation_odd.flatten(),0)
    response_correlation_matrix[:,sound] = np.nan_to_num(response_correlation.flatten(),0)


figure, axes = plt.subplots(6,6)
for sound1 in range(len(sounds_list)):
    for sound2 in range(len(sounds_list)):
        if sound1 == sound2:
            x = signal_correlation_matrix_even[:, sound1]
            y = signal_correlation_matrix_odd[:, sound1]
            # xedges = np.linspace(-1, 1, 50)
            # yedges = np.linspace(-1, 1, 50)
            # H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            # H = H.T
            # axes[sound1][sound2].imshow(H, interpolation='nearest', origin='lower',
            #                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=VMAX_1)
            axes[sound1,sound2].scatter(x,y,color = colors_list[sound1])
            axes[sound1,sound2].set_ylim([-1,1])
            axes[sound1,sound2].set_xlim([-1,1])
        else:
            x = signal_correlation_matrix[:,sound1]
            y = signal_correlation_matrix[:,sound2]
            # xedges = np.linspace(-1, 1, 50)
            # yedges = np.linspace(-1, 1, 50)
            # H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            # H = H.T
            # axes[sound1,sound2].imshow(H, interpolation='nearest', origin='lower',
            #                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin = 0, vmax = VMAX_1)
            axes[sound1,sound2].scatter(x,y,color = 'k')
            axes[sound1,sound2].set_ylim([-1,1])
            axes[sound1,sound2].set_xlim([-1,1])
figure.set_size_inches([15,15])
if BINARY:
    if ITI:
        figure_name = 'Nike_signal_sounds_correlation_matrix_ITI_' + str(n_seconds) + 'sec_binary.png'
    else:
        figure_name = 'Nike_signal_sounds_correlation_matrix_' + str(n_seconds) + 'sec_binary.png'
else:
    if ITI:
        figure_name = 'Nike_signal_sounds_correlation_matrix_ITI_'+str(n_seconds)+'sec_calcium.png'
    else:
        figure_name = 'Nike_signal_sounds_correlation_matrix_'+str(n_seconds)+'sec_calcium.png'
figure.savefig(figure_directory + figure_name)
plt.show()

figure, axes = plt.subplots(6,6)
for sound1 in range(len(sounds_list)):
    for sound2 in range(len(sounds_list)):
        if sound1 == sound2:
            x = response_correlation_matrix_even[:, sound1]
            y = response_correlation_matrix_odd[:, sound1]
            # xedges = np.linspace(-1, 1, 50)
            # yedges = np.linspace(-1, 1, 50)
            # H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            # H = H.T
            # axes[sound1][sound2].imshow(H, interpolation='nearest', origin='lower',
            #                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=VMAX_3)
            axes[sound1, sound2].scatter(x, y, color=colors_list[sound1])
            axes[sound1, sound2].set_ylim([-1, 1])
            axes[sound1, sound2].set_xlim([-1, 1])
        else:
            x = response_correlation_matrix[:,sound1]
            y = response_correlation_matrix[:,sound2]
            # xedges = np.linspace(-1, 1, 50)
            # yedges = np.linspace(-1, 1, 50)
            # H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            # H = H.T
            # axes[sound1,sound2].imshow(H, interpolation='nearest', origin='lower',
            #                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin = 0, vmax = VMAX_3)
            axes[sound1,sound2].scatter(x,y,color = 'k')
            axes[sound1,sound2].set_ylim([-1,1])
            axes[sound1,sound2].set_xlim([-1,1])
figure.set_size_inches([15,15])
if BINARY:
    if ITI:
        figure_name = 'Nike_response_correlation_matrix_ITI_'+str(n_seconds)+'sec_binary.png'
    else:
        figure_name = 'Nike_response_correlation_matrix_'+str(n_seconds)+'sec_binary.png'
else:
    if ITI:
        figure_name = 'Nike_response_correlation_matrix_ITI_'+str(n_seconds)+'sec_calcium.png'
    else:
        figure_name = 'Nike_response_correlation_matrix_'+str(n_seconds)+'sec_calcium.png'
figure.savefig(figure_directory + figure_name)
plt.show()


figure, axes = plt.subplots(6, 6)
for sound1 in range(len(sounds_list)):
    for sound2 in range(len(sounds_list)):
        if sound1 == sound2:
            x = noise_correlation_matrix_even[:, sound1]
            y = noise_correlation_matrix_odd[:, sound1]
            # xedges = np.linspace(-1, 1, 50)
            # yedges = np.linspace(-1, 1, 50)
            # H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            # H = H.T
            # axes[sound1][sound2].imshow(H, interpolation='nearest', origin='lower',
            #                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=VMAX_5)
            axes[sound1, sound2].scatter(x, y, color=colors_list[sound1])
            axes[sound1, sound2].set_ylim([-1, 1])
            axes[sound1, sound2].set_xlim([-1, 1])
        else:
            x = noise_correlation_matrix[:,sound1]
            y = noise_correlation_matrix[:,sound2]
            # xedges = np.linspace(-1, 1, 50)
            # yedges = np.linspace(-1, 1, 50)
            # H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            # H = H.T
            # axes[sound1,sound2].imshow(H, interpolation='nearest', origin='lower',
            #                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin = 0, vmax = VMAX_5)
            axes[sound1,sound2].scatter(x,y,color = 'k')
            axes[sound1,sound2].set_ylim([-1,1])
            axes[sound1,sound2].set_xlim([-1,1])
if BINARY:
    if ITI:
        figure_name = 'Nike_noise_sounds_correlation_matrix_ITI_' + str(n_seconds) + 'sec_binary.png'
    else:
        figure_name = 'Nike_noise_sounds_correlation_matrix_' + str(n_seconds) + 'sec_binary.png'
else:
    if ITI:
        figure_name = 'Nike_noise_sounds_correlation_matrix_ITI_'+str(n_seconds)+'sec_calcium.png'
    else:
        figure_name = 'Nike_noise_sounds_correlation_matrix_'+str(n_seconds)+'sec_calcium.png'
figure.set_size_inches([15,15])
figure.savefig(figure_directory + figure_name)
plt.show()

####################################################################################################
