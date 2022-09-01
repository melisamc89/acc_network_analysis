import coniii
import numpy as np
from pathlib import Path
import pickle
import random
from multiprocessing import Pool
import time

starting_time = time.time()

# PARAMETERS
n_boots = 20
sample_size = 0.5
n_cpus = 10
stimulus_duration = 60  # in indexes
stimuli = [1, 2, 3, 4, 5, 6]

input_data = '../../../../data/raw_data/nike_calcium_trace.npy'
output_path = '../../../../data/processed_data/nike'


# make output dir
Path(output_path).mkdir(parents=True, exist_ok=True)


def main():

    # load  and preprocess data
    traces = np.load('../../../../data/raw_data/nike_calcium_trace.npy')
    X = spinize(traces).T

    # clean X
    X = X[:, (np.sum(X == 1, axis=0)/X.shape[0]) > 0.01]
    print(f"# of neurons: {X.shape[1]}")

    # import behaviour data
    with open('../../../../data/raw_data/behaviour.pickle', 'rb') as f:
        behaviour = pickle.load(f)

    # compute network for non-stimulus periods
    print('Computing network for non-stimulus period ...')
    stim_mask = np.zeros(X.shape[0])
    for s in stimuli:
        for onset in behaviour[f'sound{s}']:
            stim_mask[onset:onset+stimulus_duration] = 1

    X0 = X[stim_mask == 0]

    inferred_params_dict = bootstrap_solve_PSEUDO(X0,
                                                  n_boots=n_boots,
                                                  sample_size=sample_size,
                                                  n_cpus=n_cpus)
    print('Saving inferred parameters.')
    out_file_name = 'ising_parameters_stim_0.pickle'
    with open(output_path+'/'+out_file_name, 'wb') as handle:
        pickle.dump(inferred_params_dict, handle)

    # compute network for each stimulus

    for s in stimuli:

        print(f'Computing network for stimulus {s} ... ')
        stim_mask = np.zeros(X.shape[0])
        for onset in behaviour[f'sound{s}']:
            stim_mask[onset:onset+stimulus_duration] = 1

        X0 = X[stim_mask == 1]

        inferred_params_dict = bootstrap_solve_PSEUDO(X0,
                                                      n_boots=n_boots,
                                                      sample_size=sample_size,
                                                      n_cpus=n_cpus)

        # save network params
        print('Saving inferred parameters.')
        out_file_name = f'ising_parameters_stim_{s}.pickle'
        with open(output_path+'/'+out_file_name, 'wb') as handle:
            pickle.dump(inferred_params_dict, handle)

    return

# FUNCTIONS


def spinize(traces):
    stds = np.std(traces, axis=-1)
    means = np.mean(traces, axis=-1)
    out = np.zeros_like(traces)
    for i in range(traces.shape[0]):
        for j in range(traces.shape[1]):
            if traces[i, j]-means[i] > stds[i]:
                out[i][j] = 1
            else:
                out[i][j] = -1

    return out


def bootstrap_solve_PSEUDO(X, n_boots=10, sample_size=0.5, n_cpus=4):
    print('Computing pseudolikelihood solution with:')
    print(f'Data size: {X.shape}')
    print(f'n_boots: {n_boots}')
    print(f'number of workers: {n_cpus}')
    starting_time = time.time()

    p = Pool(n_cpus)
    X_samples = []
    for _ in range(n_boots):
        sample_idxs = random.choices(
            range(len(X)), k=int(X.shape[0]*sample_size))

        X_samples.append(X[sample_idxs])

    inferred_params = p.map(solve_PSEUDO, X_samples)

    # reshape for saving as dict
    inferred_params_dict = {'h': np.asarray([l[0] for l in inferred_params]),
                            'J': np.asarray([l[1] for l in inferred_params])}

    ending_time = time.time()
    print(f'Done in {(ending_time-starting_time):.2f} seconds')

    return inferred_params_dict


def solve_PSEUDO(X):
    n_units = X.shape[-1]
    solver = coniii.Pseudo(X, iprint=False)

    solver.solve()
    multipliers = solver.multipliers
    h, J = multipliers[:n_units], multipliers[n_units:]

    return [h, J]


if __name__ == "__main__":
    main()
