import keras
import numpy as np
import os
import random
import pdb


def get_file_names(data_dir):
    all_files = []
    for f in os.listdir(data_dir):
        if f.endswith('.npz'):
            all_files.append(os.path.join(data_dir, f))

    return all_files


def data_generator(data_dir, mini_batch_size=32):
    file_names = get_file_names(data_dir)
    while True:
        random.shuffle(file_names)
        for npz_file in file_names:
            data = np.load(npz_file)
            X1 = data['X1']
            X2 = data['X2']
            Y = data['Y']

            # Remove zero rows (why do these exist!?!)
            X1 = X1[~np.all(X1 == 0, axis=1)]
            X2 = X2[~np.all(X2 == 0, axis=1)]
            Y  = Y[ ~np.all(Y  == 0, axis=1)]

            n_samples = X1.shape[0]

            # Shuffle data within npz file
            perm_ind = np.arange(n_samples)
            np.random.shuffle(perm_ind)
            X1 = X1[perm_ind,:]
            X2 = X2[perm_ind,:]
            Y = Y[perm_ind,:]

            # Normalize (Just in case)
            X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
            X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
            Y  = Y  / np.linalg.norm(Y,  axis=1, keepdims=True)
            Y = Y - X1
            X = np.hstack((X1, X2))
            for i in xrange(0, n_samples, mini_batch_size):
                try:
                    batch_x = X[i:i+mini_batch_size,:]
                    batch_y = Y[i:i+mini_batch_size,:]
                    yield batch_x, batch_y
                except:
                    pass



