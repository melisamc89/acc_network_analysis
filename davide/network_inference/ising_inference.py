import coniii
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import random
from multiprocessing import Pool

# PARAMETERS
n_boots = 10
sample_size = 0.5
n_cpus = 10

input_data = '../../../../data/raw_data/nike_calcium_trace.npy'
output_path = '../../../../data/processed_data/nike'
out_file_name = 'ising_parameters_all_trial.pickle'

# make output dir
Path(output_path).mkdir(parents=True, exist_ok=True)


def main():

    # load  and preprocess data
    traces = np.load('../../../../data/raw_data/nike_calcium_trace.npy')
    X = spinize(traces).T

    # clean X
    X = X[:, (np.sum(X == 1, axis=0)/X.shape[0]) > 0.01]
    print(f"# of neurons: {X.shape[1]}")

    # prepare for parallelization
    p = Pool(n_cpus)
    X_samples = []
    for _ in range(n_boots):
        sample_idxs = random.choices(
            range(len(X)), k=int(X.shape[0]*sample_size))

        X_samples.append(X[sample_idxs])

    print('Starting network inference ... ')
    inferred_params = p.map(solve_PSEUDO, X_samples)

    # reshape for saving as dict
    inferred_params_dict = {'h': np.asarray([l[0] for l in inferred_params]),
                            'J': np.asarray([l[1] for l in inferred_params])}

    # save network params
    print('Done, saving inferred parameters.')
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


def bootstrap_solve_ACE(X, n_boots=10, sample_size=0.5, threshold=pow(10, 2), n_cpus=8):
    hs = []
    Js = []
    n_units = X.shape[-1]
    for _ in tqdm(range(n_boots)):
        sample_idxs = random.choices(
            range(len(X)), k=int(X.shape[0]*sample_size))
        X_sample = X[sample_idxs]
        print(f'sample shape: {X_sample.shape}')
        solver = coniii.ClusterExpansion(X_sample, n_cpus=n_cpus)
        multipliers = solver.solve(threshold)
        h, J = multipliers[:n_units], multipliers[n_units:]
        hs.append(h)
        Js.append(J)

    return {'h': hs, 'J': Js}


def bootstrap_solve_PSEUDO(X, n_boots=10, sample_size=0.5):
    hs = []
    Js = []
    n_units = X.shape[-1]
    for _ in tqdm(range(n_boots)):
        sample_idxs = random.choices(
            range(len(X)), k=int(X.shape[0]*sample_size))
        X_sample = X[sample_idxs]
        print(f'sample shape: {X_sample.shape}')
        solver = coniii.Pseudo(X_sample)
        solver.solve()
        multipliers = solver.multipliers
        h, J = multipliers[:n_units], multipliers[n_units:]

        print(f'max h: {max(h)}')
        print(f'max J: {max(J)}')

        hs.append(h)
        Js.append(J)

    return {'h': hs, 'J': Js}


def solve_PSEUDO(X):
    n_units = X.shape[-1]
    solver = coniii.Pseudo(X)
    solver.solve()
    multipliers = solver.multipliers
    h, J = multipliers[:n_units], multipliers[n_units:]
    return [h, J]


if __name__ == "__main__":
    main()
