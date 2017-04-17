import utils
from random import shuffle
import random
import numpy as np
import sys
import os
import pdb
import argparse
import logging
import time

# Script to run in the terminal
# python -u main.py </dev/null &>/dev/null & disown
# or run ./profile_main.sh for debugging

def arg_parser():
    '''
    Parse cmd arguments and set logging verbosity
    '''
    parser = argparse.ArgumentParser( description='Generate Linear SVM Weight Triples' )
    parser.add_argument("-v", "--verbose", action='store_true',
            help="Increase output verbosity")
    parser.add_argument('-b', '--base_num', type=int, default=10,
            help='Number of base classes')
    parser.add_argument('-c', '--chunk', type=int, default=2000,
            help='Number of base classes')
    parser.add_argument('-s', '--samples_num', type=int, default=500000,
            help='Number of samples to generate')
    parser.add_argument('-i', '--input_dir', type=str,default=None,
            help='Input features directory path')
    parser.add_argument('-o', '--output', type=str,
            default=None,
            help='Save ground truth path')
    parser.add_argument('--svm_library', type=str, default='pegasos',
            help='Set SVM Library')
    parser.add_argument('-m', '--meta', action='store_true',
            help='Record Metadata')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Debugging Verbosity Set')
    logging.debug(args)

    return args

if __name__ == "__main__":
    # Handle Input Arguments
    args = arg_parser()

    #Setup Data and SVM Handlers
    logging.debug('Loading feature vectors into memory')
    data_handler = utils.DataHandler(args.input_dir)
    logging.debug('Data successfully loaded into memory')
    svm_handler = utils.SVMHandler(data_handler)
    all_categories = data_handler.get_all_categories() # list of strings (classes)

    meta_path = os.path.join(args.output, '0.meta')
    meta_saver = utils.MetaSaver(meta_path, all_categories)

    logging.debug('Beginning SVM Iterator')
    n_samples = 0

    t_cum = 0
    save_X1 = np.zeros((args.chunk, 4096), dtype=np.float)
    save_X2 = np.zeros((args.chunk, 4096), dtype=np.float)
    save_Y = np.zeros((args.chunk, 4096), dtype=np.float)


    for n_samples in xrange(args.samples_num):
        t_start = time.time()
        logging.debug('Processing sample ' + str(n_samples))
        # Randomly create base and novel lists
        shuffle(all_categories)
        # Select random number of negative classes
        num_base_classes = args.base_num
        base_classes = all_categories[:num_base_classes]
        novel_classes = all_categories[num_base_classes:]


        pos_class = random.choice(base_classes)
        neg_classes_old = base_classes[:]
        neg_classes_old.remove(pos_class)

        novel_class = random.choice(novel_classes)
        neg_classes_new = neg_classes_old[:]
        neg_classes_new.append(novel_class)

        X1 = svm_handler.get_svm_weights(pos_class, neg_classes_old,
                    svm_library=args.svm_library)
        X2 = svm_handler.get_svm_weights(novel_class, base_classes)
        Y = svm_handler.get_svm_weights(pos_class, neg_classes_new)

        save_X1[n_samples%args.chunk, :] = X1
        save_X2[n_samples%args.chunk, :] = X2
        save_Y[n_samples%args.chunk, :] = Y

        meta_saver.save(str(n_samples), pos_class, novel_class, neg_classes_old)

        # save and report time
        t_end = time.time()
        t_diff = t_end - t_start
        t_cum += t_diff
        t_start = time.time()
        t_avg = t_cum / (n_samples + 1)
        print('Iteration %d Time: %.2fs. Avg Time: %.2fs. Cum Time: %.2fs'
                % (n_samples, t_diff, t_avg, t_cum))
        sys.stdout.flush()


        if (((n_samples-1) % args.chunk == 0) and n_samples > 1):
            save_path = os.path.join(args.output, str(int((n_samples-1)/args.chunk))+'.npz')
            np.savez(save_path, X1=save_X1, X2= save_X2, Y=save_Y)

            save_X1.fill(0)
            save_X2.fill(0)
            save_Y.fill(0)

            meta_path = os.path.join(args.output, str(int((n_samples-1)/args.chunk))+'.meta')
            meta_saver = utils.MetaSaver(meta_path, all_categories)



        # for pos_class in base_classes:
        #     # Calculate the 'Old SVM' Parameters
        #     neg_classes_old = base_classes[:]
        #     neg_classes_old.remove(pos_class)
        #     X1 = svm_handler.get_svm_weights(pos_class, neg_classes_old,
        #             svm_library=args.svm_library)

        #     for novel_class in novel_classes:
        #         # Calculate the Novel SVM Parameters
        #         X2 = svm_handler.get_svm_weights(novel_class, base_classes)

        #         # Calcuate the New SVM Parameters
        #         neg_classes_new = neg_classes_old[:]
        #         neg_classes_new.append(novel_class)
        #         Y = svm_handler.get_svm_weights(pos_class, neg_classes_new)

        #         # save and report time
        #         t_end = time.time()
        #         t_diff = t_end - t_start
        #         t_cum += t_diff
        #         t_start = time.time()
        #         t_avg = t_cum / (n_samples + 1)
        #         logging.debug('Iteration %d Time: %.2fs. Avg Time: %.2fs. Cum Time: %.2fs'
        #                 % (n_samples, t_diff, t_avg, t_cum))

        #         save_path = os.path.join(args.output, str(n_samples) + '.npz')
        #         np.savez(save_path, X1=X1, X2=X2, Y=Y)
        #         meta_saver.save(str(n_samples), pos_class,
        #                 novel_class, neg_classes_old)
        #         #exit condition
        #         n_samples += 1
        #         if n_samples >= args.samples_num:
        #             sys.exit()
