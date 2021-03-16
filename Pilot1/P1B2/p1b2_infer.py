from __future__ import print_function

import argparse

import numpy as np

from keras import backend as K
from keras import optimizers
from keras.models import Model, Sequential, model_from_json
from keras.layers import Activation, Dense, Dropout, Input
from keras.initializers import RandomUniform
from keras.callbacks import Callback, ModelCheckpoint, History
from keras.regularizers import l2

#import sys,os

import p1b2
import candle

def initialize_parameters(default_model = 'p1b2_default_model.txt'):

    # Build benchmark object
    p1b2Bmk = p1b2.BenchmarkP1B2(p1b2.file_path, default_model, 'keras',
    prog='p1b2_baseline', desc='Train Classifier - Pilot 1 Benchmark 2')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p1b2Bmk)
    #p1b2.logger.info('Params: {}'.format(gParameters))

    return gParameters


def run(gParameters):
    
    # Construct extension to save model
    ext = p1b2.extension_from_parameters(gParameters, '.keras')
    candle.verify_path(gParameters['save_path'])
    prefix = '{}{}'.format(gParameters['save_path'], ext)
    logfile = gParameters['logfile'] if gParameters['logfile'] else prefix+'.log'
    #candle.set_up_logger(logfile, p1b2.logger, gParameters['verbose'])
    #p1b2.logger.info('Params: {}'.format(gParameters))

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = candle.keras_default_config()
    seed = gParameters['rng_seed']
    
    # Load dataset
    #(X_train, y_train), (X_test, y_test) = p1b2.load_data(gParameters, seed)
    #(X_train, y_train), (X_val, y_val), (X_test, y_test) = p1b2.load_data_one_hot(gParameters, seed)
    (X_train, y_train), (X_test, y_test) = p1b2.load_data2(gParameters, seed)

    # print ("Shape X_train: ", X_train.shape)
    # print ("Shape X_val: ", X_val.shape)
    print ("Shape X_test: ", X_test.shape)
    # print ("Shape y_train: ", y_train.shape)
    # print ("Shape y_val: ", y_val.shape)
    print ("Shape y_test: ", y_test.shape)

    # print ("Range X_train --> Min: ", np.min(X_train), ", max: ", np.max(X_train))
    # print ("Range X_val --> Min: ", np.min(X_val), ", max: ", np.max(X_val))
    print ("Range X_test --> Min: ", np.min(X_test), ", max: ", np.max(X_test))
    # print ("Range y_train --> Min: ", np.min(y_train), ", max: ", np.max(y_train))
    # print ("Range y_val --> Min: ", np.min(y_val), ", max: ", np.max(y_val))
    print ("Range y_test --> Min: ", np.min(y_test), ", max: ", np.max(y_test))

    # Define optimizer
    optimizer = candle.build_optimizer(gParameters['optimizer'],
                                                gParameters['learning_rate'],
                                                kerasDefaults)

    # load json and create model
    json_file = open('p1b2.model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model_json.load_weights('p1b2.model.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model_json.compile(loss=gParameters['loss'], optimizer=optimizer, metrics=['accuracy'])
    y_pred = loaded_model_json.predict(X_test)
    scores = p1b2.evaluate_accuracy_one_hot(y_pred, y_test)
    print('Evaluation on test data:', scores)

def main():
   params = initialize_parameters()
   run(params)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
