#!/usr/bin/python
#
# MNIST digit recognizer
#
# The simple recognizer is implememted purely in Python.  The purpose of
# this program is to present the details how to constructa simple neural
# network for prediction from scratch:
#
# 1. To build a 3-layer neural network (only one hidden layer).
# 2. To train a model with self-implemented SGD (stochastic gradient descent).
# 3. To predict data with the trained model.
#
# This program is based on the exercise of Andrew Ng's machine learning
# course on Coursera: https://www.coursera.org/learn/machine-learning
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import math
import pandas as pd
import scipy.io as sio


# Structure of the 3-layer neural network.
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10


def load_training_data(training_file='mnistdata.mat'):
    '''Load training data (mnistdata.mat) and return (inputs, labels).

    The training data is from Andrew Ng's exercise of the Coursera
    machine learning course (ex4data1.mat).
    '''
    training_data = sio.loadmat(training_file)
    return (training_data['X'], training_data['y'])


def load_weights(weight_file='ex4weights.mat'):
    weights = sio.loadmat(weight_file)
    #print(weights.keys())
    return weights


def rand_init_weights(size_in, size_out):
    epsilon_init = 0.12
    return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init


def sigmoid(z):
    return 1.0 / (1 + pow(math.e, -z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cost_function(theta1, theta2, input_layer_size, hidden_layer_size, output_layer_size, inputs, labels, regular=0):
    # transfer tensors to numpy arrays
    inputs = np.array(inputs)  # 5000x400
    theta1 = np.array(theta1)  # 25x401
    theta2 = np.array(theta2)  # 10x26

    # construct neural network
    input_layer = np.insert(inputs, 0, 1, axis=1)  # add bias, 5000x401

    hidden_layer = np.dot(input_layer, np.transpose(theta1))
    hidden_layer = pd.DataFrame(hidden_layer).apply(sigmoid).as_matrix()
    hidden_layer = np.insert(hidden_layer, 0, 1, axis=1)  # add bias, 5000x26

    output_layer = np.dot(hidden_layer, np.transpose(theta2))  # 5000x10
    output_layer = pd.DataFrame(output_layer).apply(sigmoid).as_matrix()

    # forward propagation: calculate cost
    cost = 0.0
    for training_index in xrange(len(inputs)):
        # transform label y[i] from a number to a vector.
        #
        # Note:
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #    1  2  3  4  5  6  7  8  9 10
        #
        #   if y[i] is 0 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        #   if y[i] is 1 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        outputs = [0] * output_layer_size
        outputs[labels[training_index]-1] = 1

        for k in xrange(output_layer_size):
            cost += -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
    cost /= len(inputs)

    # back propagatino: calculate gradiants
    theta1_grad = np.zeros(theta1.shape)  # 25x401
    theta2_grad = np.zeros(theta2.shape)  # 10x26
    for training_index in xrange(len(inputs)):
        # transform label y[i] from a number to a vector.
        outputs = [0] * output_layer_size
        outputs[labels[training_index]-1] = 1

        # calculate delta3
        delta3 = output_layer[training_index] - outputs  # 10x1

        # calculate delta2
        z2 = np.dot(theta1, input_layer[training_index])
        z2 = np.insert(z2, 0, 1)  # add bias, 26x1
        delta2 = np.multiply(
            np.dot(np.transpose(theta2), delta3),
            list(map(sigmoid_gradient, z2))
        )
        delta2 = delta2[1:]  # 25x1

        # calculate gradients of theta1 and theta2
        theta1_grad += np.dot(
            np.asmatrix(delta2).transpose(),
            np.asmatrix(input_layer[training_index])
        )  # 25x401
        theta2_grad += np.dot(
            np.asmatrix(delta3).transpose(),
            np.asmatrix(hidden_layer[training_index])
        )  # 10x26
    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)

    return cost, (theta1_grad, theta2_grad)


def gradient_descent(inputs, labels, learningrate=0.8, iteration=50):
    '''
    @return cost and trained model (weights).
    '''
    rand_theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
    rand_theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    theta1 = rand_theta1
    theta2 = rand_theta2
    cost = 0.0
    for i in xrange(iteration):
        cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            Input_layer_size, Hidden_layer_size, Output_layer_size,
            inputs, labels, regular=0)
        theta1 -= learningrate * theta1_grad
        theta2 -= learningrate * theta2_grad
        print('Iteration {0} (learning rate {2}, iteration {3}), cost: {1}'.format(i+1, cost, learningrate, iteration))
    return cost, (theta1, theta2)


def train(inputs, labels, learningrate=0.8, iteration=50):
    cost, model = gradient_descent(inputs, labels, learningrate, iteration)
    return model


def predict(model, inputs):
    theta1, theta2 = model
    inputs = np.array(inputs)  # 5000x400
    theta1 = np.array(theta1)  # 25x401
    theta2 = np.array(theta2)  # 10x26
    a1 = np.insert(inputs, 0, 1, axis=1) # add bias, 5000x401
    a2 = np.dot(a1, theta1.transpose())
    a2 = pd.DataFrame(a2).apply(sigmoid).as_matrix()
    a2 = np.insert(a2, 0, 1, axis=1)     # add bias, 5000x26
    a3 = np.dot(a2, theta2.transpose())
    a3 = pd.DataFrame(a3).apply(sigmoid).as_matrix()  # 5000x10
    return [i.argmax()+1 for i in a3]


if __name__ == '__main__':
    # Note: There are 10 units which present the digits [1-9, 0]
    # (in order) in the output layer.
    inputs, labels = load_training_data()

    # (optional) load pre-trained model for debugging neural network construction
    weights = load_weights()
    theta1 = weights['Theta1']  # size: 25 entries, each has 401 numbers
    theta2 = weights['Theta2']  # size: 10 entries, each has  26 numbers
 
    cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2, Input_layer_size, Hidden_layer_size, Output_layer_size, inputs, labels, regular=0)
    print('cost:', cost)

    # train the model from scratch and predict based on it
    # learning rate 0.10, iteration  60: 36% (cost: 3.217)
    # learning rate 1.75, iteration  50: 77%
    # learning rate 1.90, iteration  50: 75%
    # learning rate 2.00, iteration  50: 82%
    # learning rate 2.00, iteration 100: 87%
    # learning rate 2.00, iteration 200: 93% (cost: 0.572)
    # learning rate 2.00, iteration 300: 94% (cost: 0.485)
    # learning rate 2.05, iteration  50: 79%
    # learning rate 2.20, iteration  50: 64%
    model = train(inputs, labels, learningrate=0.1, iteration=60)
    #model = (theta1, theta2)
    outputs = predict(model, inputs)

    correct_prediction = 0
    for i, predict in enumerate(outputs):
        if predict == labels[i][0]:
            correct_prediction += 1
    precision = float(correct_prediction) / len(labels)
    print('precision: {}'.format(precision))
