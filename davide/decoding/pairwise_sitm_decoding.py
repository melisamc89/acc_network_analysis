from itertools import product
from re import L
from tqdm import tqdm
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import pandas as pd


# PARAMETERS
stimuli = [1, 2, 3, 4, 5, 6]
output_path = '../../../../data/processed_data/decoding/nike'
output_fname = output_path + '/pairwise_stim_decoding.scv'

data_path = '../../../../data/raw_data'
stimulus_duration = 60  # in indexes

n_splits = 10  # number of cv splits


def main():

    # create output folder
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # import data
    traces = np.load(data_path+'/nike_calcium_trace.npy')
    with open(data_path+'/behaviour.pickle', 'rb') as f:
        behaviour_data = pickle.load(f)

    # define data container for output
    performance_data = {'stim1': [], 'stim2': [],
                        'fold': [], 'f1': [], 'f1_x_shuff': [], 'f1_y_shuff': []}

    for stim1, stim2 in product(stimuli, stimuli):
        if stim1 == stim2:
            continue

        print(f'Decoding stimulus {stim1} from {stim2} ...')

        y = build_labels(behaviour_data=behaviour_data,
                         y_len=traces.shape[1],
                         stimulus_duration=stimulus_duration,
                         stim1=stim1,
                         stim2=stim2)

        # build feature matrix, only take stimulus presentation periods
        stim_mask = y > 0
        X = traces.T[stim_mask]
        y = y[stim_mask]

        print(
            f'stimulus {stim1}, len: {sum(y==stim1)}, fraction: {sum(y==stim1)/len(y)}')
        print(
            f'stimulus {stim2}, len: {sum(y==stim1)},fraction: {sum(y==stim2)/len(y)}')

        performance, x_baseline, y_baseline = compute_cv_performance(
            X, y, n_splits=n_splits)

        performance_data['stim1'] += [stim1 for _ in range(n_splits)]
        performance_data['stim2'] += [stim2 for _ in range(n_splits)]
        performance_data['fold'] += [i for i in range(n_splits)]
        performance_data['f1'] += performance
        performance_data['f1_x_shuff'] += x_baseline
        performance_data['f1_y_shuff'] += y_baseline

    performance_data = pd.DataFrame.from_dict(performance_data)

    performance_data.to_csv(output_fname)
    print(f"Done, data saved at {output_fname}")

    return


def build_labels(behaviour_data, y_len, stimulus_duration, stim1, stim2):
    y = np.zeros(y_len)
    for onset in behaviour_data[f'sound{stim1}']:
        y[onset:onset+stimulus_duration] = stim1

    for onset in behaviour_data[f'sound{stim2}']:
        y[onset:onset+stimulus_duration] = stim2

    return y


def compute_cv_performance(X, y, n_splits):
    logistic = linear_model.LogisticRegression(
        solver="newton-cg", tol=1, C=1000)
    scaler = StandardScaler()
    classifier = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    performance = []
    y_shuffled_performance = []
    x_shuffled_performance = []

    X_shuff = np.random.permutation(X)

    for i, (train_index, test_index) in tqdm(enumerate(kf.split(X, y))):
        X_train, X_test = X[train_index], X[test_index]
        X_shuff_train, X_shuff_test = X_shuff[train_index], X_shuff[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # LR classifier
        split_classifier = clone(classifier)
        split_classifier.fit(X_train, y_train)

        y_pred = split_classifier.predict(X_test)
        performance.append(f1_score(y_test, y_pred, average='weighted'))

        # shuffled baselines
        y_shuffled_performance.append(
            f1_score(y_test, np.random.permutation(y_test), average='weighted'))

        shuff_classifier = clone(classifier)
        shuff_classifier.fit(X_shuff_train, y_train)
        y_pred = split_classifier.predict(X_shuff_test)
        x_shuffled_performance.append(
            f1_score(y_test, y_pred, average='weighted'))

    return performance, x_shuffled_performance, y_shuffled_performance


if __name__ == '__main__':
    main()
