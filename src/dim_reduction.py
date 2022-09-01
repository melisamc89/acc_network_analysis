import numpy as np
import matplotlib.pyplot as plt
import src.plotting_function as plotting
from sklearn.decomposition import PCA
import pickle

input_directory = '/home/melisamc/Documentos/acc_network_analysis/data/traces/'
output_directory = '/home/melisamc/Documentos/acc_network_analysis/data/binary/'
figure_directory = '/home/melisamc/Documentos/acc_network_analysis/figures/'

file_name = 'nike_calcium_trace.npy'
#file_name = 'havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces_accepted.npy'
srate = 30
calcium_trace = np.load(input_directory + file_name)
calcium_mean = np.mean(calcium_trace, axis=1)
calcium_std = np.std(calcium_trace, axis=1)
n_std = 1
spikes_threshold = calcium_mean + n_std * calcium_std

binary_signal = np.zeros_like(calcium_trace)
for neuron in range(calcium_trace.shape[0]):
    binary_signal[neuron, np.where(calcium_trace[neuron,:] > spikes_threshold[neuron])[0]] = 1

pca = PCA(n_components=10)
pca.fit(binary_signal.T)
X = pca.transform(binary_signal.T)
pca.fit(calcium_trace.T)
X = pca.transform(calcium_trace.T)

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

figure = plt.figure()
axes = plt.axes()#projection='3d')
#axes.scatter(X[:,0],X[:,1],X[:,2], color = 'k',alpha = 0.2)
colors_list = ['b','r','g','orange','magenta','cyan']
mean_list = []
for sound in range(6):
    mean_activity_vector = np.zeros((3,len(sounds_list[sound])))
    for i in range(len(sounds_list[sound])-2):
        time1 = sounds_list[sound][i]
        time2 = sounds_list[sound][i] + 2*30
        intensity = 1-(i/len(sounds_list[sound]))/1.2
        print(intensity)
        mean_activity_vector[:,i]= np.mean(X[time1:time2,0:3],axis=0)
        axes.scatter(X[time1:time2,0],X[time1:time2,1],color = colors_list[sound],alpha = intensity)
    mean_list.append(mean_activity_vector)
plt.show()


figure = plt.figure()
gs = plt.GridSpec(4, 5)
#axes.scatter(X[:,0],X[:,1],X[:,2], color = 'k',alpha = 0.2)
colors_list = ['b','r','g','orange','magenta','cyan']
for i in range(4):
    for j in range(5):
        axes = figure.add_subplot(gs[i, j] ,projection='3d')
        axes.set_xlim([-1,5])
        axes.set_ylim([-1,1])
        axes.set_zlim([-1,2])

        for sound in range(6):
            if len(sounds_list[sound]) > (i*5+j):
                time1 = sounds_list[sound][i*5+j]
                time2 = sounds_list[sound][i*5+j] + 2*30
                axes.scatter(X[time1:time2,0],X[time1:time2,1],X[time1:time2,2],color = colors_list[sound])
plt.show()


figure = plt.figure()
gs = plt.GridSpec(2,3)
n_seconds = 4
#axes.scatter(X[:,0],X[:,1],X[:,2], color = 'k',alpha = 0.2)
colors_list = ['b','r','g','orange','magenta','cyan']
import matplotlib.cm as cm
cmap = cm.jet
mean_activity = np.zeros((3,6,n_seconds*30))
for sound1 in range(2):
    for sound2 in range(3):
        sound = sound1*3 + sound2
        axes = figure.add_subplot(gs[sound1, sound2] )#,projection='3d')
        axes.set_xlim([-1,6])
        axes.set_ylim([-2,1])
        #axes.set_zlim([-1,2])

        data_points = np.zeros((3,len(sounds_list[sound])*n_seconds*30))
        data_matrix = np.zeros((3,len(sounds_list[sound]),n_seconds*30))
        color = np.linspace(0, 20, len(sounds_list[sound])*n_seconds*30)
        for i in range(len(sounds_list[sound])):
            time1 = sounds_list[sound][i]
            time2 = sounds_list[sound][i] + n_seconds*30
            data_points[:,i*n_seconds*30:(i+1)*n_seconds*30] = X[time1:time2,0:3].T
            data_matrix[:,i,:] =   X[time1:time2,0:3].T
            #intensity = 1 - (i / len(sounds_list[sound])) / 1.2
        mean_activity[:,sound,:] = np.mean(data_matrix,axis = 1)
            #axes.scatter(X[time1:time2,0],X[time1:time2,1],[time1:time2,2],color = colors_list[sound])
        axes.scatter(data_points[0,:],data_points[1,:],c= color , cmap = cmap)
plt.show()

figure = plt.figure()
axes = plt.axes(projection='3d')
for sound in range(6):
    axes.scatter(mean_activity[0,sound,:],mean_activity[1,sound,:],mean_activity[2,sound,:],color = colors_list[sound])
plt.show()

figure = plt.figure()
axes = plt.axes(projection='3d')
for sound in range(6):
    for i in range(len(sounds_list[sound])):
        axes.plot(mean_list[sound][0,:],mean_list[sound][1,:],mean_list[sound][2,:],color = colors_list[sound])
plt.show()