# -*- coding: utf-8 -*-
"""
Using the MNIST dataset
..note: if the dataset is not present it will be downloaded using urllib

"""

__docformat__ = ’restructedtext en’


import os
import sys
import numpy
import cPickle
from urllib import urlretrieve
from config import mnist_dataset_origin as origin

def load_data(path):
    print "Checking if the MNIST dataset is present..."
    path_dir, path_file = os.path.split(path)
    if path_dir == "" and not os.path.isfile(path):
        print "Path seems to be incorrect. Checking if the MNIST dataset is in the directory..."
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", path)
        if os.path.isfile(new_path) or path_file == 'mnist.pkl.gz':
            print "Found in :", new_path
            path = new_path
    if (not os.path.isfile(path)) and path_file == "mnist.pkl.gz":
        print "Failed to find data. Downloading from :", origin, " ..."
        urlretrieve(origin, path)
        
    print "Lodaing data..."
    
    # open the file
    f = gzip.open(path, 'rb')
    training_set, validation_set, test_set = cPickle.load(f)
    f.close()
    def into_shared(data_xy, borrow = True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),
                                borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
                                borrow=borrow)
        return shared_x, T.cast(shared_y, ’int32’)
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(validation_set)
    train_set_x, train_set_y = shared_dataset(training_set)
    set_set = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return set_set
