from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import cv2
import keras
from keras import backend as K
import argparse
import logging
import os


class FeatureExtractor:

    def __init__(self, model=None):
        if model == None:
            self.model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
        else:
            self.model = model
        self.get_model_features = K.function([self.model.layers[0].input], [self.model.layers[21].output])


    def get_feature(self, img_path):
        im = cv2.resize(cv2.imread(img_path), (224, 224))
        im = np.expand_dims(im, axis=0)
        feature = self.get_model_features([im])[0]
        return feature.flatten()

def arg_parser():
    '''
    Parsing cmd arguments and set logging verbosity
    '''
    parser = argparse.ArgumentParser(description='Extract VGG features for images in the dataset')
    parser.add_argument("-v", "--verbose", action='store_true', help='Increase output verbosity')
    parser.add_argument('-i', '--input_dir', type=str, default='/media/storage/ILSVRC2016/ILSVRC/Data/DET/train/ILSVRC2013_train',
                        help='Dataset path')
    parser.add_argument('-o', '--output_dir', type=str, default=os.path.join('..', 'vgg_features_new'),
                        help='Feature output path')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug('Debugging Verbosity Set')
    logging.debug(args)

    return args

if __name__ == "__main__":
    # Handle Input Arguments
    args = arg_parser()
    feature_extractor = FeatureExtractor()
    classes = os.listdir(args.input_dir)
    for index, class_name in enumerate(classes):
        logging.debug('Processing class ' + str(index) + '/' + str(len(classes)))
        imgs = os.listdir(os.path.join(args.input_dir, class_name))

        if not os.path.exists(os.path.join(args.output_dir, class_name)):
            os.makedirs(os.path.join(args.output_dir, class_name))

        for img in imgs:
            # print img
            feature = feature_extractor.get_feature(os.path.join(args.input_dir, class_name, img))
            img_name = img.split('.')[0]
            save_path = os.path.join(args.output_dir, class_name, img_name+'.npz')
            np.savez(save_path, feature=feature)



