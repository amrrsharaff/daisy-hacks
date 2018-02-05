import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
#from sklearn.datasets import load_boston

def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)

#def read_boston_data():
#    boston = load_boston()
#    print(boston.data)
#    features = np.array(boston.data)
#    labels = np.array(boston.target)
#    return features, labels

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l

if __name__ == "__main__":
    array = read_dataset("/Users/ASharaf/Desktop/hackathon_data/trial.csv")
    #features, labels = read_boston_data()
    normalized_features = feature_normalize(features)
    f, l = append_bias_reshape(normalized_features, labels)
    n_dim = f.shape[1]
    learning_rate = 0.01
    training_epochs = 1000
    cost_history = np.empty(shape=[1], dtype=float)

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.ones([n_dim, 1]))

    init = tf.initialize_all_variables()

    y_ = tf.matmul(X, W)
    cost = tf.reduce_mean(tf.square(y_ - Y))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    print(array)