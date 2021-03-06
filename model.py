# -*- coding: utf-8 -*-

"""
Will contain the learning models

"""
__docformat__ = ’restructedtext en’

import os
import sys
import numpy
import theano
import cPickle
import theano.tensor as T

def LogisticRegressionModel(object):
    """Logistic Regression Class that encapsulates the
    basic behaviour of logistic regression
    """
    def __init__(self, input, k_data, k_target):
        """Initialize the Logistic Regression class parameters

        :param k_data   : dimension of the datapoint space,
                            number of input units
        :param k_target : dimension of the target space,
                            number of output units
        """

        #weight matrix
        self.W = theano.shared(value = numpy.zero((k_data, k_target), dtype = theano.config.floatX), name = 'W')
        
        #bias vector
        self.b = theano.shared(value = numpy.zeros((k_target,), dtype = theano.config.floatX), name = 'b')

        #class-membership probabilities vector
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+ self.b)

        #prediction
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        
    #loss function
    def negative_loglikelihood(self, y):
        """Return the mean NLL of the model's prediction
        for current distribution of the target space
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arrange(y.shape[0]), y])

    def errors(self, y):
        """Return the number of errors in the batch over its
        number of examples
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError('Type Error :: y and slef.y_pred should have the same dimension',
                            'y', target.type, 'y_pred', self.y_pred.type)
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        

def HiddenLayer():
    pass

def Perceptron():
    pass

