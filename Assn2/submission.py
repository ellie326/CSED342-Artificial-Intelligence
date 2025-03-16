#!/usr/bin/python

import random
import collections # you can use collections.Counter if you would like
import math

import numpy as np

from util import *

SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, interesting, great, plot, bored, not
    """
    # BEGIN_YOUR_ANSWER
    return {'so' : 0, 'interesting' : 0, 
            'great' : 1, 'plot' : 1, 
            'bored' : -1, 'not' : -1}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER
    
    word_cnt = {} # initialize 
    
    for word in x.split(): # split x by whitespace 
        word_cnt[word] = word_cnt.get(word, 0) + 1 # increment each dictionary by one when detected  
    
    return word_cnt

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER

    def dotProduct(d1, d2):
        if len(d1) < len(d2):
            return dotProduct(d2, d1)
        else: # calculated dot product by iteration 
            return sum(d1.get(f, 0) * v for f, v in d2.items()) 
    
    def deriveLossNLL(x, y, w): 
        loss = dict() # initialize 
        dotXW = dotProduct(x, w) # calculate dot product 

        # calculated the derivative of the loss function 
        if y == 1:
            derivative = sigmoid(dotXW) - 1 
        elif y == -1: 
            derivative = sigmoid(dotXW)  

        #update the loss dictionary with the feature value * derivative 
        for word, value in x.items():
            loss[word] = value * derivative
        return loss
            
    x = list() # initialize - feature vectors 
    y = list() # initialize - labels 

    # store train examples into x and y 
    for trainSet in trainExamples: 
        x.append(featureExtractor(trainSet[0]))
        y.append(trainSet[1])

    # initialize weight to 0 
    for dictTemp in x:
        for word in dictTemp.keys():
            weights[word] = 0.0

    # update weights 
    for i in range(numIters): 
        for num, word in enumerate(x):
            GradientLoss = deriveLossNLL(word, y[num], weights)
            for gradWord in GradientLoss.keys():
                weights[gradWord] -= eta * GradientLoss[gradWord]
    

    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER
    
    words = x.split() 
    
    features = {} # initialize - n gram features 
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n]) # join n consecutive words 
        if ngram in features:
            features[ngram] += 1
        else:
            features[ngram] = 1
    
    return features

    # END_YOUR_ANSWER

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

class MLPBinaryClassifier:
    """
    A binary classifier with a 2-layer neural network
        input --(hidden layer)--> hidden --(output layer)--> output
    Each layer consists of an affine transformation and a sigmoid activation.
        layer(x) = sigmoid(x @ W + b)
    """
    def __init__(self):
        self.input_size = 2  # input feature dimension
        self.hidden_size = 16  # hidden layer dimension
        self.output_size = 1  # output dimension

        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.init_weights()

    def init_weights(self):
        weights = np.load("initial_weights.npz")
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]

    def forward(self, x):
        """
        Inputs
            x: input 2-dimensional feature (B, 2), B: batch size
        Outputs
            pred: predicted probability (0 to 1), (B,)
        """
        # BEGIN_YOUR_ANSWER
        
        #sigmoid function used in learn Predictor 
        def sigmoid(x):
                return 1 / (1 + np.exp(-x))
        
        # z1 = first layer's weight sum 
        # x = input features 
        # w1 = first set of weights 
        # b1 = bias 
        # a1 = activation of the first layer 
        self.z1 = x.dot(self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.x = x

        z2 = self.a1.dot(self.W2) + self.b2

        # a2 = predicted probabilities 
        a2 = sigmoid(z2)

        # convert shape using squeeze 
        return a2.squeeze()

        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            loss: negative log likelihood loss, (B,)
        """
        # BEGIN_YOUR_ANSWER

        # for the positive class (target = 1), calculate log(pred)
        # for the negative class (target = 0), calculate log(1- pred)
        loss = -(target * np.log(pred) + (1 - target) * np.log(1 - pred))

        return loss
    
        # END_YOUR_ANSWER

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        
        # BEGIN_YOUR_ANSWER

        B = len(target) #Batch = 4 for first testcase 

        dZ2 = (pred - target).reshape(B, 1)

        dW2 = self.a1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * (self.a1 * (1 - self.a1))

        dW1 = self.x.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        gradient = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


        return gradient

        # END_YOUR_ANSWER


    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]  
        # END_YOUR_ANSWER

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 2), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the negative log likelihood loss of the last step
        """
        # BEGIN_YOUR_ANSWER

        for epoch in range(epochs):
            total_loss = 0
            for i in range(X.shape[0]):
                x_i = X[i:i+1]
                y_i = Y[i:i+1]
                
                pred = self.forward(x_i)
                gradients = self.backward(pred, y_i)
                loss = self.loss(pred, y_i)
                self.update(gradients, learning_rate)
                
                total_loss += loss
    
        return loss
    
        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x))