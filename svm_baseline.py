import numpy as np
import os
from os.path import join as join
import time
from random import shuffle
import pegasos
import utils
import random
import sys

def get_data_splits(parent_dir, pos_class, neg_classes):
    
    all_pos_files = os.listdir(join(parent_dir, pos_class))
    all_pos_files.sort()
    num_pos_features = len(all_pos_files)
    train_pos_files = all_pos_files[:int(0.8*num_pos_features)]
    train_val_files = all_pos_files[int(0.8*num_pos_features):]
    num_val_samples = len(train_val_files)
    
    train_pos = []
    train_neg = []
    val_pos = []
    val_neg = []

    # Get 80% of positive features for training
    for f in train_pos_files:
        pos_feature = np.load(join(parent_dir, pos_class, f))['feature']
        train_pos.append(pos_feature)

    # Rest of the positive samples are for validation
    for f in train_val_files:
        pos_feature = np.load(join(parent_dir, pos_class, f))['feature']
        val_pos.append(pos_feature)

    # Get only 1000 negative training samples
    for i in xrange(1000):
        neg_class = random.choice(neg_classes)
        neg_samples = os.listdir(join(parent_dir, neg_class))
        neg_sample = random.choice(neg_samples)
        neg_feature = np.load(join(parent_dir, neg_class, neg_sample))['feature']
        train_neg.append(neg_feature)

    # Get 'n' number of negative validation samples
    for i in xrange(num_val_samples):
        neg_class = random.choice(neg_classes)
        neg_samples = os.listdir(join(parent_dir, neg_class))
        neg_sample = random.choice(neg_samples)
        neg_feature = np.load(join(parent_dir, neg_class, neg_sample))['feature']
        val_neg.append(neg_feature)

    return np.array(train_pos), np.array(train_neg), np.array(val_pos), np.array(val_neg)


def get_x_y(pos_features, neg_features):
    x = np.vstack((pos_features, neg_features))
    y = np.hstack((np.ones( len(pos_features)),
                   np.zeros(len(neg_features))))
    return x, y


def compute_accuracy(weight_vector, positive_features, neg_features):
    classifier = pegasos.PegasosSVMClassifier()
    classifier.fit(np.zeros((2, 4096)), np.asarray([1, 0]))
    classifier.weight_vector.weights = weight_vector

    # Concat data and pass to SVM
    result = classifier.predict(np.vstack((positive_features, neg_features)))
    ground_truth = np.concatenate((np.ones(len(positive_features)), np.zeros(len(neg_features))))
    return np.average(np.equal(ground_truth, result))


input_dir = '/media/storage/capstone/VGG16_features'
NUM_SAMPLES = 1000
NUM_BASE_CLASSES = 100
t_cum = 0

acc = np.zeros((NUM_SAMPLES), dtype=np.float)
data_handler = utils.DataHandler(input_dir)
print('Data loaded ...')
all_categories = data_handler.get_all_categories()

for i in xrange(NUM_SAMPLES):
    t_start = time.time()
    shuffle(all_categories)
    base_classes = all_categories[:NUM_BASE_CLASSES]
    novel_classes = all_categories[NUM_BASE_CLASSES:]

    pos_class = random.choice(base_classes)
    neg_classes = base_classes[:]
    neg_classes.remove(pos_class)

    novel_class = random.choice(novel_classes)
    neg_classes.append(novel_class)
    
    # Train SVM
    pos_train, neg_train, pos_val, neg_val = get_data_splits(input_dir, pos_class, neg_classes)
    x_train, y_train = get_x_y(pos_train, neg_train)
    svm = pegasos.PegasosSVMClassifier()
    svm.fit(x_train, y_train)
    weight_vector = svm.weight_vector.weights

    # Evaluate SVM on validation data
    acc[i]=compute_accuracy(weight_vector, pos_val, neg_val)
    
    t_end = time.time()
    t_diff = t_end - t_start
    t_cum += t_diff
    t_avg = t_cum / (i + 1)
    sys.stdout.flush()
    print('Iteration %d Time: %.2fs. Avg Time: %.2fs. Cum Time: %.2fs. Accuracy: %.5f'
                            % (i, t_diff, t_avg, t_cum, acc[i]))

print (np.mean(acc))
