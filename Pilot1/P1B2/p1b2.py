from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

import os
import sys
import logging
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

from file_utils import get_file


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import candle

#url_p1b2 = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/'
#file_train = 'P1B2.train.csv'
#file_test = 'P1B2.test.csv'

logger = logging.getLogger(__name__)

additional_definitions = [
{'name':'reg_l2',
'type': float,
'default': 0.,
'help':'weight of regularization for l2 norm of nn weights'}
]

required = [
    'data_url',
    'train_data',
    'test_data',
    'activation',
    'batch_size',
    'dense',
    'dropout',
    'epochs',
    'feature_subsample',
    'initialization',
    'learning_rate',
    'loss',
    'optimizer',
    'reg_l2',
    'rng_seed',
    'scaling',
    'val_split',
    'shuffle'
]

class BenchmarkP1B2(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions

def extension_from_parameters(params, framework):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    ext += '.A={}'.format(params['activation'])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.D={}'.format(params['dropout'])
    ext += '.E={}'.format(params['epochs'])
    if params['feature_subsample']:
        ext += '.F={}'.format(params['feature_subsample'])
    for i, n in enumerate(params['dense']):
        if n:
            ext += '.D{}={}'.format(i+1, n)
    ext += '.S={}'.format(params['scaling'])

    return ext


def load_data_one_hot(params, seed):
   # fetch data
    file_train = candle.fetch_file(params['data_url'] + params['train_data'],subdir='Pilot1')
    file_test = candle.fetch_file(params['data_url'] + params['test_data'],subdir='Pilot1')

    return candle.load_Xy_one_hot_data2(file_train, file_test, class_col=['cancer_type'],
                                           drop_cols=['case_id', 'cancer_type'],
                                           n_cols=params['feature_subsample'],
                                           shuffle=params['shuffle'],
                                           scaling=params['scaling'],
                                           validation_split=params['val_split'],
                                           dtype=params['data_type'],
                                           seed=seed)


def load_data(params, seed):
   # fetch data
    file_train = candle.fetch_file(params['data_url'] + params['train_data'],subdir='Pilot1')
    file_test = candle.fetch_file(params['data_url'] + params['test_data'],subdir='Pilot1')

    return candle.load_Xy_data2(file_train, file_test, class_col=['cancer_type'],
                                  drop_cols=['case_id', 'cancer_type'],
                                  n_cols=params['feature_subsample'],
                                  shuffle=params['shuffle'],
                                  scaling=params['scaling'],
                                  validation_split=params['val_split'],
                                  dtype=params['data_type'],
                                  seed=seed)

def load_data2(params, seed, shuffle=True, n_cols=None):
    train_path = get_file('P1B2.train.csv', origin='http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/P1B2.train.csv')
    test_path = get_file('P1B2.test.csv', origin='http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/P1B2.test.csv')

    usecols = list(range(n_cols)) if n_cols else None

    df_train = pd.read_csv(train_path, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_path, engine='c', usecols=usecols)

    # df_train = candle.fetch_file(params['data_url'] + params['train_data'],subdir='Pilot1')
    # df_test  = candle.fetch_file(params['data_url'] + params['test_data'],subdir='Pilot1')

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    X_train = df_train.iloc[:, 2:].values
    X_test = df_test.iloc[:, 2:].values

    y_train = pd.get_dummies(df_train[['cancer_type']]).values
    y_test = pd.get_dummies(df_test[['cancer_type']]).values

    return (X_train, y_train), (X_test, y_test)


def evaluate_accuracy_one_hot(y_pred, y_test):
    def map_max_indices(nparray):
        maxi = lambda a: a.argmax()
        iter_to_na = lambda i: np.fromiter(i, dtype=np.float)
        return np.array([maxi(a) for a in nparray])
    ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
    accuracy = accuracy_score(ya, ypa)
    # print('Accuracy: {}%'.format(100 * accuracy))
    return {'accuracy': accuracy}


def evaluate_accuracy(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy: {}%'.format(100 * accuracy))
    return {'accuracy': accuracy}
