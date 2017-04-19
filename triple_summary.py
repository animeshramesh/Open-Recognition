import numpy as np
import os
import pdb
import re
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

'''
Used to generate histrogram data about
triple distributions
'''

def sort_nicely(l):
    """
    Sort the given list in the way that humans expect.
    From Ned Batchelder
    https://nedbatchelder.com/blog/200712/human_sorting.html
    """
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        def tryint(s):
            try:
                return int(s)
            except:
                return s
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l

def get_file_names(data_dir):
    all_files = []
    for f in os.listdir(data_dir):
        if f.endswith('.npz'):
            all_files.append(os.path.join(data_dir, f))
    return all_files

if __name__ == '__main__':
    '''
    Loads data and calculates the following statistics:
    1) Histogram of euclidean distance between X1 and Y
    2) [ToDo] Histogram of cosine similarity
    '''
    data_dir = '/media/storage/capstone/data/ILSVRC2013/svm_triples/train'
    file_names = sort_nicely(get_file_names(data_dir))

    # Remove the last entry; useful when data is still being created
    remove_last = True
    if remove_last:
       del file_names[-1]

    # Euclidean Hist Param
    euclidean_bin_width = 0.01
    euclidean_bin_edges = np.arange(0, 1.5, euclidean_bin_width)
    digi_delta_Y_euclidean_cum = np.zeros_like(euclidean_bin_edges)

    # Cosaine Hist Param
    cosine_bin_width = 0.005
    cosine_bin_edges =  np.arange(0, 0.35, cosine_bin_width)
    digi_delta_Y_cosine_cum =  np.zeros_like(cosine_bin_edges)

    # ITERATE THROUGH THE FILES
    for npz_file in file_names:
        # Load in data
        data = np.load(npz_file)
        X1 = data['X1']
        X2 = data['X2']
        Y = data['Y']

        # Remove zero rows (why do these exist!?!)
        X1 = X1[~np.all(X1 == 0, axis=1)]
        X2 = X2[~np.all(X2 == 0, axis=1)]
        Y  = Y[ ~np.all(Y  == 0, axis=1)]

        # Normalize (Just in case)
        X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
        X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
        Y  = Y  / np.linalg.norm(Y,  axis=1, keepdims=True)

        delta_Y = Y - X1
        delta_Y_euclidean = np.linalg.norm(delta_Y, axis=1)

        # EUCLIDEAN DIST HIST DATA
        digi_delta_Y_euclidean = np.digitize(delta_Y_euclidean, euclidean_bin_edges) - 1
        for i in digi_delta_Y_euclidean:
            digi_delta_Y_euclidean_cum[i] += 1

        # COSINE DIST HIST DATA
        n_rows = X1.shape[0]
        delta_Y_cosine = np.zeros((n_rows,))
        for i in xrange(n_rows):
            delta_Y_cosine[i] = cosine(X1[i,:], Y[i,:])
        digi_delta_Y_cosine = np.digitize(delta_Y_cosine, cosine_bin_edges) - 1
        for i in digi_delta_Y_cosine:
            digi_delta_Y_cosine_cum[i] += 1

    # Create Euclidean histogram
    # normalize bincounts into a probability distribution
    digi_delta_Y_euclidean_cum = digi_delta_Y_euclidean_cum / np.sum(digi_delta_Y_euclidean_cum)

    # plot bargraph (x label are the left binning)
    fig1, ax1 = plt.subplots()
    rects1 = ax1.bar(left=euclidean_bin_edges,
                    height=digi_delta_Y_euclidean_cum,
                    width=euclidean_bin_width,
                    color='r')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Percent Occurence')
    plt.title('Euclidean Distance Occurence')

    # Create Cosine histogram
    # normalize bincounts into a probability distribution
    digi_delta_Y_cosine_cum = digi_delta_Y_cosine_cum / np.sum(digi_delta_Y_cosine_cum)

    # plot bargraph (x label are the left binning)
    fig2, ax2 = plt.subplots()
    rects2 = ax2.bar(left=cosine_bin_edges,
                    height=digi_delta_Y_cosine_cum,
                    width=cosine_bin_width,
                    color='r')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Percent Occurence')
    plt.title('Cosine Distance Occurence')

    plt.show()
