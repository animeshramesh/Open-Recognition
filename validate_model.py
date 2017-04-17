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
    parser.add_argument('-i', '--input_model', type=str,
            default=os.path.join('..', 'models', 'regressor_shared_fc1.h5'),
            help='Input trained model')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Debugging Verbosity Set')
    logging.debug(args)

    return args


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


def simulate_validation(model, svm_lib, base_num):

    data_handler = utils.DataHandler()
    svm_handler = utils.SVMHandler(data_handler)
    all_categories = data_handler.get_all_categories()
    shuffle(all_categories)
    base_classes = all_categories[:base_num]
    novel_classes = all_categories[base_num:]

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
            Y_hat = X1 + Y_hat_delta

            # Compare results of SVM and deep model
            pos_features, neg_features = prepare_features(pos_class, neg_classes_new, data_handler)
            accuracy_svm = compute_accuracy(X1, pos_features, neg_features)
            accuracy2 = compute_accuracy(Y_hat, pos_features, neg_features)
            logging.debug('Base class = {0}, Novel class = {1} -> X1 = {2}, Y_hat = {3}'.format(pos_class, novel_class, str(accuracy_svm), str(accuracy2)))


def main():
    args = arg_parser()
    model = load_model(args.input_model)
    validate_model(model)

if __name__ == '__main__':
    main()
