from __future__ import print_function

import numpy as np

from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint
from keras.regularizers import l2

import p1b2


BATCH_SIZE = 64
NB_EPOCH = 20                 # number of training epochs
PENALTY = 0.00001             # L2 regularization penalty
ACTIVATION = 'sigmoid'
FEATURE_SUBSAMPLE = None
DROP = None

L1 = 1024
L2 = 512
L3 = 256
L4 = 0
LAYERS = [L1, L2, L3, L4]


class BestLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf
        self.best_model = None

    def on_epoch_end(self, batch, logs={}):
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            self.best_model = self.model
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


def extension_from_parameters():
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(ACTIVATION)
    ext += '.B={}'.format(BATCH_SIZE)
    ext += '.D={}'.format(DROP)
    ext += '.E={}'.format(NB_EPOCH)
    if FEATURE_SUBSAMPLE:
        ext += '.F={}'.format(FEATURE_SUBSAMPLE)
    for i, n in enumerate(LAYERS):
        if n:
            ext += '.L{}={}'.format(i+1, n)
    ext += '.P={}'.format(PENALTY)
    return ext


def main():
    (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=FEATURE_SUBSAMPLE)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # load json and create model
    json_file = open('p1b2.model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model_json.load_weights('p1b2.model.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model_json.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(loaded_model_json.summary())

    y_pred = loaded_model_json.predict(X_test)

    scores = p1b2.evaluate(y_pred, y_test)
    print('Evaluation on test data:', scores)

if __name__ == '__main__':
    main()
