import numpy as np
import pegasos
import utils
import pdb
import os
import sys
from random import shuffle
from keras.models import load_model
import argparse
import logging

def arg_parser():
    '''
    Parse cmd arguments and set logging verbosity
    '''
    parser = argparse.ArgumentParser( description='Compute the accuracy of a trained model' )
    parser.add_argument("-v", "--verbose", action='store_true',
            help="Increase output verbosity")
    parser.add_argument('-b', '--base_num', type=int, default=10,
            help='Number of base classes')
    parser.add_argument('-i', '--input_model', type=str,
            default=os.path.join('..', 'models', 'regressor_shared_fc1.h5'),
            help='Input trained model')

    parser.add_argument('-f', '--features_path', type=str,
            default='/media/storage/capstone/data/ILSVRC2013/vgg16/ILSVRC2013_validate',
            help='Input VGG features')

    parser.add_argument('--input_dir', type=str,
            default='/media/storage/capstone/data/ILSVRC2013/svm_triples/validate',
            help='Path to triplet directory. (Can be train or validation)')

    parser.add_argument('--svm_library', type=str, default='pegasos',
            help='Set SVM Library')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Debugging Verbosity Set')
    logging.debug(args)

    return args

# def test():
#     GT_DIR = '/capstone/GT_Pugh'
#     meta_filename = os.path.join(GT_DIR, 'meta.txt')
#     # Load in the meta word mapping data
#     word_list, word_map = utils.load_word_map(meta_filename)

#     # Load in all data (features) from disk
#     data_handler = utils.DataHandler()


#     # Load and parse all the meta data
#     with open(meta_filename, 'r') as f:
#         content = f.readlines()
#     content = [line.rstrip('\n') for line in content]
#     del content[0] #remove the first word_list line
#     n_triples = len(content)

#     # Load up a dummy SVM
#     classifier = pegasos.PegasosSVMClassifier()
#     classifier.fit(np.zeros((2,4096)), np.asarray([1,0]))

#     for exemplar_triple in content:
#         split_exemplar = filter(None, exemplar_triple.split(' '))
#         split_exemplar = [int(x) for x in split_exemplar]

#         # parse the exemplar
#         identity = split_exemplar[0]
#         base_class_idx = split_exemplar[1]
#         base_class_name = word_list[base_class_idx]
#         novel_class_idx = split_exemplar[2]
#         novel_class_name = word_list[novel_class_idx]
#         neg_class_idxs = np.asarray(split_exemplar[3:])
#         neg_class_names = [word_list[x] for x in neg_class_idxs]

#         # load in the SVM weights
#         weight_filename = os.path.join(GT_DIR, str(identity) + '.npz')
#         #dictionary with keys X1, X2, and Y
#         # X1: base_svm, X2: novel_svm, Y: updated base_svm
#         weights = np.load(weight_filename)
#         classifier.weight_vector.weights = weights['X1'] / np.linalg.norm(weights['X1'])

#         #This is totally just for debugging
#         classifier.predict(data_handler.features['fire'])
#         pdb.set_trace()
#     pos_category_name = 'zebra'
#     neg_category_name = 'fish'


def validate_model(args):

    meta_filename = os.path.join(args.input_dir, '1.meta')

    # Load in the meta word mapping data
    word_list, word_map = utils.load_word_map(meta_filename)

    # Load in all data (features) from disk
    data_handler = utils.DataHandler(args.features_path)

    # Load and parse all the meta data
    with open(meta_filename, 'r') as f:
        content = f.readlines()
    content = [line.rstrip('\n') for line in content]
    del content[0] #remove the first word_list line
    n_triples = len(content)

    # Load the X1, X2 and Y data from the npz file
    triplets = np.load(os.path.join(args.input_dir, '1.npz'))
    X1 = triplets['X1']
    X2 = triplets['X2']
    Y = triplets['Y']

    # Normalize data
    X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    Y  = Y  / np.linalg.norm(Y,  axis=1, keepdims=True)

    # Load the trained keras model
    model = load_model(args.input_model)

    # Iterate through all the rows in the npz file (2000 x 4096)
    for i, exemplar_triple in enumerate(content):
        split_exemplar = filter(None, exemplar_triple.split(' '))
        split_exemplar = [int(x) for x in split_exemplar]

        # parse the exemplar
        identity = split_exemplar[0]
        base_class_idx = split_exemplar[1]
        base_class_name = word_list[base_class_idx]
        novel_class_idx = split_exemplar[2]
        novel_class_name = word_list[novel_class_idx]
        neg_class_idxs = np.asarray(split_exemplar[3:])
        neg_class_names = [word_list[x] for x in neg_class_idxs]

        # X1 and X2 have features of shape (4096,). Need to reshape them
        X1_net = np.reshape(X1[i,:], (1,4096))
        X2_net = np.reshape(X2[i,:], (1,4096))
        X_net = np.hstack((X1_net, X2_net))

        Y_hat_delta = model.predict(X_net)

        Y_hat = X1_net + Y_hat_delta

        # Get the positive and negative features for the current base class
        pos_features, neg_features = prepare_features(base_class_name, [novel_class_name] + neg_class_names, data_handler)

        # Compute the accuracy of the svm and the regressor network
        accuracy_old_svm = compute_accuracy(X1[i,:], pos_features, neg_features)
        accuracy_goal_svm = compute_accuracy(Y[i,:], pos_features, neg_features)
        accuracy_regressor = compute_accuracy(Y_hat, pos_features, neg_features)

        logging.debug('Old SVM accuracy = {0}, Goal SVM accuracy = {1} Regressor network acc = {2}'.format(
                str(accuracy_old_svm), str(accuracy_goal_svm), str(accuracy_regressor)))


def compute_accuracy(weight_vector, positive_features, neg_features):
    '''
    Computes the accuracy of the model, given its weights.
    :param weight_vector: 4096 weight vector to be loaded to SVM
    :param positive_features: SVM should return +1 for these samples
    :param neg_features: SVM should return -1 for these samples
    :return: accuracy : Accuracy of class classification
    '''

    classifier = pegasos.PegasosSVMClassifier()
    classifier.fit(np.zeros((2, 4096)), np.asarray([1, 0]))
    classifier.weight_vector.weights = weight_vector

    # Concat data and pass to SVM
    result = classifier.predict(np.vstack((positive_features, neg_features)))
    ground_truth = np.concatenate((np.ones(len(positive_features)), np.zeros(len(neg_features))))

    return np.average(np.equal(ground_truth, result))


def prepare_features(pos_class, neg_classes, data_handler):
    '''
    The output of this function is fed to the SVM classifier (for computing accuracy).
    :param pos_class:
    :param neg_classes:
    :param data_handler:
    :return: pos_features : (nx4096) -> features of n positive samples
             neg_features : (nx4096) -> features of n negative samples
    '''
    pos_features = data_handler.get_features(pos_class)
    neg_features = []
    for neg_class in neg_classes:
        neg_features.extend(data_handler.get_features(neg_class))
    neg_features = np.array(neg_features)

    return pos_features, neg_features


def simulate_validation(model, svm_lib, base_num, features_path):

    data_handler = utils.DataHandler(features_path)
    svm_handler = utils.SVMHandler(data_handler)
    all_categories = data_handler.get_all_categories()
    shuffle(all_categories)
    base_classes = all_categories[:base_num]
    novel_classes = all_categories[base_num:]

    svm_acc = []
    regressor_acc = []

    for pos_class in base_classes:
        # Calculate the 'Old SVM' Parameters
        neg_classes_old = base_classes[:]
        neg_classes_old.remove(pos_class)
        X1 = svm_handler.get_svm_weights(pos_class, neg_classes_old,
                                         svm_library=svm_lib)

        for novel_class in novel_classes:
            # Calculate the Novel SVM Parameters
            X2 = svm_handler.get_svm_weights(novel_class, base_classes)

            # Calculate the New SVM Parameters
            neg_classes_new = neg_classes_old[:]
            neg_classes_new.append(novel_class)
            Y = svm_handler.get_svm_weights(pos_class, neg_classes_new)		# This is not needed. But just in case..

            X = np.array([np.hstack((X1, X2))])
            # Y_hat_delta = model.predict(X)

            X1_net = np.reshape(X1, (1,4096))
            X2_net = np.reshape(X2, (1,4096))

            Y_hat_delta = model.predict([X1_net, X2_net])
            Y_hat = X1_net + Y_hat_delta

            # Compare results of SVM and deep model
            pos_features, neg_features = prepare_features(pos_class, neg_classes_new, data_handler)
            accuracy_svm = compute_accuracy(X1, pos_features, neg_features)
            accuracy2 = compute_accuracy(Y_hat, pos_features, neg_features)

            # Append to global accuracies
            svm_acc.append(accuracy_svm)
            regressor_acc.append(accuracy2)

            # logging.debug('Base class = {0}, Novel class = {1} -> SVM accuracy = {2}, Regressor network acc = {3}'.format(pos_class, novel_class, str(accuracy_svm), str(accuracy2)))
            logging.debug('SVM accuracy = {0}, Regressor network acc = {1}'.format(str(accuracy_svm), str(accuracy2)))

def main():
    # test()
    args = arg_parser()
    validate_model(args)
    # simulate_validation(model, args.svm_library, args.base_num, args.features_path)

if __name__ == '__main__':
    main()
