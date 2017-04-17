import os
import numpy as np
import pegasos #fast SVM solver
import pdb

def load_word_map(meta_filename):
    # Read in all the lines
    with open(meta_filename, 'r') as f:
        content = f.readline()
    #strip the \n
    content = content.rstrip('\n')

    # Get the word mapping
    # Split up the string and remove empty strings
    word_list = filter(None, content.split(' '))
    n_word = len(word_list)
    word_idx = np.arange(n_word)

    word_map = dict(zip(word_list, word_idx))

    return word_list, word_map


class DataHandler:

    def __init__(self, features_path):
        '''
        Loads feature vectors and metadata into memroy.

        Sets the following attributes:
        features_path - path to folder containing class folders containing feature vector files
                        [Default - ../features]
        categories - list of strings.  Each string is a class name.
                     Folder ontology dictates cateogory names.
        features - dictionary with class names as keys.
        '''
        self.features = {}      # class_name -> nx4096 feature matrix
        if not features_path:
            self.features_path = os.path.join('..','features')
        else:
            self.features_path = features_path

        # load category names
        self.categories = os.listdir(self.features_path)

        # load all feature vectors
        for category in self.categories:
            self.features[category] = self.get_features(category)

    def get_features(self, category):
        '''
        Loads all feature vectors from disk for a given category.

        inputs:
        category - [string] name of category label to load features of

        output:
        2D numpy matrix of features (n_samples, features)
        '''
        # get all filenames in the category directory
        feature_files = os.listdir(os.path.join(self.features_path, category))
        features = []
        for f in feature_files:
            path = os.path.join(self.features_path, category, f)
            try:
                if path[-1]=='z':
                    load_in = np.load(path)
                    features.append(load_in[load_in.keys()[0]])
                else:
                    features.append(np.load(path))
            except:
                print "Fail to load %s" % path
        # convert the list into a 2d numpy matrix
        return np.array(features)

    def prepare_data_for_svm(self, positive_class, neg_classes):
        '''
        Prepares pre-loaded data for being fed into an SVM

        inputs:
        positive_class - string of positive class label
        neg_classes - list of strings of negative class labels

        outputs:
        x - 2d matrix of feature vectors (n_samples, features)
        y - binary label vectory (n_samples,)
        '''
        pos_features = self.features[positive_class]
        neg_features = []
        for neg_class in neg_classes:
            neg_features.extend(self.features[neg_class])
        x = np.vstack((pos_features, neg_features))
        y = np.hstack((np.ones( len(pos_features)),
                       np.zeros(len(neg_features))))
        return x, y

    def get_all_categories(self):
        #return self.features.keys()
        return self.categories

class SVMHandler:
    '''
    Exapandable Generic SVM class
    '''
    def __init__(self, data_handler):
        self.data_handler = data_handler

        #dictionary used to determine what fit function to call
        self.fit_jump_dict={
                'pegasos':self.__fit_pegasos,
                }

        #dictionary used to determine how to return SVM weights
        self.get_weights_jump_dict={
                'pegasos':self.__get_pegasos_weights,
                }

    def __fit_pegasos(self, pos_class, neg_classes):
        '''
        Uses the Pegasos Library to calculate a Linear SVM
        inputs:
        positive_class - string of positive class label
        neg_classes - list of strings of negative class labels
        outputs:
        svm object
        '''
        x, y = self.data_handler.prepare_data_for_svm(pos_class, neg_classes)
        svm = pegasos.PegasosSVMClassifier()
        svm.fit(x, y)
        return svm

    def __get_pegasos_weights(self, svm):
        return svm.weight_vector.weights

    ###################
    # Primary Methods #
    ###################
    def fit(self, pos_class, neg_classes, svm_library='pegasos'):
        'Returns a trained SVM using the specified library'
        return self.fit_jump_dict[svm_library](pos_class, neg_classes)

    def get_weights(self, svm, svm_library='pegasos'):
        'Returns weights using the specified library'
        return self.get_weights_jump_dict[svm_library](svm)

    def get_svm_weights(self, pos_class, neg_classes, svm_library='pegasos'):
        'Calculates an SVM and returns the SVM parameters (weights).'
        svm = self.fit(pos_class, neg_classes, svm_library)
        return self.get_weights(svm, svm_library)

class MetaSaver:
    '''
    meta data saver

    save file format:
    - First line is the list of classes (space separated strings)
    All following lines have the following format (col space separated):
    1) first col = file save name
    2) second col = base positive class index (0-indexing)
    3) third col = novel class index (0-indexing)
    4) remaining col = negative class indices (sorted least to greatest
    '''
    def __init__(self, filename, class_list, active=True):
        '''
        sets:
        class_list - list of classes strings
        filename - name of file to be writing into
        f - filestream to be writing into
        encoder - dictionary to map class strings to 0-indexing
        '''
        self.active = active
        if self.active:
            class_list.sort(key=str.lower)
            self.class_list = class_list[:]
            self.filename = filename
            f = open(filename, 'wb')

            #build encoder dictionary
            self.encoder = {}
            for i in xrange(len(self.class_list)):
                self.encoder[self.class_list[i]]=i

            # write class_list to first line
            for label in class_list:
                f.write('%s ' % label)
            f.write('\n')
            f.close()

    def save(self, name, pos_class, novel_class, neg_classes):
        if self.active:
            f = open(self.filename, 'ab')
            # save filename, and cipher classes (sorted)
            f.write('%s ' % name)
            f.write('%s ' % str(self.encoder[pos_class]))
            f.write('%s ' % str(self.encoder[novel_class]))

            # encode and sort labels
            encoded = [self.encoder[label] for label in neg_classes]
            encoded.sort()
            for label in encoded:
                f.write('%s ' % str(label))
            f.write('\n')
            f.close()
