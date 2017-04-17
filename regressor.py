import utils
from random import shuffle
import numpy as np
import sys
import os
import pdb
import argparse
import logging
import time
from os import listdir
import os.path

# Keras import
import keras
from keras.initializers import glorot_normal
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, concatenate, Lambda
from keras.layers.merge import Add
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint

from batch_generator import data_generator


# Script to run in the terminal
# python -u regressor.py </dev/null &>/dev/null & disown
# or run ./run_regressor.sh


def arg_parser():
    '''
    Parse cmd arguments and set logging verbosity
    '''
    parser = argparse.ArgumentParser( description='Train Regressor Network' )
    parser.add_argument("-v", "--verbose", action='store_true', help="Increase output verbosity")
    parser.add_argument('-i', '--train_dir', type=str, default='/media/storage/capstone/data/ILSVRC2013/svm_triples/train',
                        help='Training Data path')

    parser.add_argument('--valid_dir', type=str, default='/media/storage/capstone/data/ILSVRC2013/svm_triples/validate',
                        help='Validation Data path')
    parser.add_argument('-l', '--loss_type', type=str, default='mean_squared_error',
                        help='Type of loss for the model : mean_squared_error '
                             'or mean_absolute_error or cosine_proximity')
    parser.add_argument('-s', '--save_file', type=str, default=os.path.join('..', 'models/default.h5'),
                        help='Trained model will be saved to this file')
    parser.add_argument('-e', '--epoch', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size for each iteration')
    parser.add_argument('-m', '--model_type', type=str, default='layer_4_leaky',
                        help='Model type for training')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Debugging Verbosity Set')
    logging.debug(args)

    return args

#
#def get_training_data(train_dir):
#    '''
#    Fetch the training samples and return the normalized weight vectors.
#    :param train_dir: parent directory of the training data.
#    :return: X and Y which can be directly passed to the model
#    '''
#    samples = [x for x in listdir(train_dir) if '.npz' in x]
#    logging.debug('Loading ' + str(len(samples)) + ' samples')
#    X = []
#    Y = []
#    for f in samples:
#        data = np.load(os.path.join(train_dir, f))
#        X1 = data['X1'] / np.linalg.norm(data['X1'])
#        X2 = data['X2'] / np.linalg.norm(data['X2'])
#        X.append(np.hstack((X1, X2)))
#        Y.append(data['Y']/np.linalg.norm(data['Y']))
#    X = np.array(X)
#    Y = np.array(Y)
#    return X, Y
#

if __name__ == "__main__":
    # Handle Input Arguments
    args = arg_parser()

    if args.model_type == 'layer_4_leaky':
        # Model definition in Keras
        alpha = 0.1
        model = Sequential()
        model.add(Dense(8192, input_dim=8192))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(8192))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(4096))

        # optimizer = optimizers.Adam(lr=0.005)
        optimizer = optimizers.SGD(lr=0.00001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

    elif args.model_type == 'layer_3_leaky':
        alpha = 0.1
        model = Sequential()
        model.add(Dense(8192, input_dim=8192))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(4096))

        optimizer = optimizers.Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

    elif args.model_type == 'layer_2_leaky':
        alpha = 0.1
        model = Sequential()
        model.add(Dense(8192, input_dim=8192))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(4096))

        optimizer = optimizers.Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

    elif args.model_type=='siamese_concat_2':
        X = Input(shape=(8192,))
        def split1(z):
            return z[:,:4096]
        def split2(z):
            return z[:,4096:]
        X1 = Lambda(split1)(X)
        X2 = Lambda(split2)(X)

        fc1 = Dense(4096, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')

        fc1_X1 = fc1(X1)
        fc1_X2 = fc1(X2)

        # test with concat, and with addition
        merged = concatenate([fc1_X1, fc1_X2], axis=-1)

        fc2 = Dense(4096, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')(merged)
        optimizer = optimizers.Adam(lr=0.001)
        model = Model(inputs=X, outputs=fc2)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

    elif args.model_type=='siamese_concat_3':
        X = Input(shape=(8192,))
        def split1(z):
            return z[:,:4096]
        def split2(z):
            return z[:,4096:]
        X1 = Lambda(split1)(X)
        X2 = Lambda(split2)(X)

        fc1 = Dense(4096, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')

        fc1_X1 = fc1(X1)
        fc1_X2 = fc1(X2)

        # test with concat, and with addition
        merged = concatenate([fc1_X1, fc1_X2], axis=-1)

        fc2 = Dense(8192, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')(merged)
        fc2 = Dense(4096, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')(merged)

        optimizer = optimizers.Adam(lr=0.001)
        model = Model(inputs=X, outputs=fc2)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

    elif args.model_type=='siamese_add_3':
        X = Input(shape=(8192,))
        def split1(z):
            return z[:,:4096]
        def split2(z):
            return z[:,4096:]
        X1 = Lambda(split1)(X)
        X2 = Lambda(split2)(X)

        fc1 = Dense(4096, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')

        fc1_X1 = fc1(X1)
        fc1_X2 = fc1(X2)

        # test with concat, and with addition
        merged = Add()([fc1_X1, fc1_X2])

        fc2 = Dense(8192, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')(merged)
        fc2 = Dense(4096, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')(merged)

        optimizer = optimizers.Adam(lr=0.001)
        model = Model(inputs=X, outputs=fc2)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

    elif args.model_type=='siamese_add_3_1':
        X = Input(shape=(8192,))
        def split1(z):
            return z[:,:4096]
        def split2(z):
            return z[:,4096:]
        X1 = Lambda(split1)(X)
        X2 = Lambda(split2)(X)

        fc1 = Dense(4096, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')

        fc1_X1 = fc1(X1)
        fc1_X2 = fc1(X2)

        # test with concat, and with addition
        merged = Add()([fc1_X1, fc1_X2])

        fc2 = Dense(8192, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')(merged)
        fc2 = Dense(4096, input_dim=4096, activation='relu',
                kernel_initializer=glorot_normal(),
                bias_initializer='zeros')(merged)

        optimizer = optimizers.Adam(lr=0.0001)
        model = Model(inputs=X, outputs=fc2)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

    else:
        print('Unknown Model. Exiting')
        sys.exit(1)


    # Initialize generators
    train_generator = data_generator(args.train_dir, args.batch_size)
    validation_generator = data_generator(args.valid_dir, args.batch_size)

    # Train the model
    checkpoint = ModelCheckpoint(filepath=args.save_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # epochs use to be 35000
    model.fit_generator(train_generator, steps_per_epoch=5000, epochs=args.epoch,
        validation_data=validation_generator, validation_steps=4000,max_q_size=64,workers=1, callbacks=callbacks_list)
    model.save(args.save_file)
