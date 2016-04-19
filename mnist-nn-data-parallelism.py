#!/usr/bin/python
#
# MNIST digit recognizer in GPU mode
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

import functools
import numpy as np
import math
import pandas as pd
import os
import scipy.io as sio
import time

from mpi4py import MPI

if os.getenv('MNISTNN_GPU_MODE') == 'yes':
    Gpu_mode = True
else:
    Gpu_mode = False

Gpu_mode = True

if Gpu_mode is True:
    import theano
    import theano.tensor as T

Distributed = True
#Distributed = False


# Init MPI
comm = MPI.COMM_WORLD

# Structure of the 3-layer neural network.
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10

# Matrix product function.  Default is to use CPU mode.
Matrix_dot = np.dot


def convert_memory_ordering_f2c(array):
    if np.isfortran(array) is True:
        return np.ascontiguousarray(array)
    else:
        return array


def load_training_data(training_file='mnistdata.mat'):
    '''Load training data (mnistdata.mat) and return (inputs, labels).

    inputs: numpy array with size (5000, 400).
    labels: numpy array with size (5000, 1).

    The training data is from Andrew Ng's exercise of the Coursera
    machine learning course (ex4data1.mat).
    '''
    training_data = sio.loadmat(training_file)
    inputs = training_data['X'].astype('d')
    inputs = convert_memory_ordering_f2c(inputs)
    labels = training_data['y'].reshape(training_data['y'].shape[0])
    labels = convert_memory_ordering_f2c(labels)
    return (inputs, labels)


def load_weights(weight_file='mnistweights.mat'):
    '''Load training data (mnistdata.mat) and return (inputs, labels).

    The weights file is from Andrew Ng's exercise of the Coursera
    machine learning course (ex4weights.mat).
    '''
    weights = sio.loadmat(weight_file)
    theta1 = convert_memory_ordering_f2c(weights['Theta1'].astype('d'))  # size: 25 entries, each has 401 numbers
    theta2 = convert_memory_ordering_f2c(weights['Theta2'].astype('d'))  # size: 10 entries, each has  26 numbers
    return (theta1, theta2)


def rand_init_weights(size_in, size_out):
    epsilon_init = 0.12
    return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init


def sigmoid(z):
    return 1.0 / (1 + pow(math.e, -z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


if Gpu_mode is True:
    def gpu_matrix_dot():
        time_start = time.time()
        x = T.matrix('x')
        y = T.matrix('y')
        z = T.dot(x, y)
        f = theano.function([x, y], z, allow_input_downcast=True)
        time_end = time.time()
        print('theano expression creation costs {} secs'.format(time_end - time_start))
        return f
else:
    def gpu_matrix_dot():
        pass


def cost_function(theta1, theta2, input_layer_size, hidden_layer_size, output_layer_size, inputs, labels, regular=0):
    '''
    Note: theta1, theta2, inputs, labels are numpy arrays:

        theta1: (25, 401)
        theta2: (10, 26)
        inputs: (5000, 400)
        labels: (5000, 1)
    '''
    # DEBUG: display parameters' information
    #print('-'*80)
    #if Distributed is True:
    #    if comm.rank == 0:
    #        print('Parameter inputs:')
    #    print('[{0}]\tlen: {1}'.format(comm.rank, len(inputs)))
    #    print('[{0}]\tsum: {1}'.format(comm.rank, inputs.sum()))
    #    print('[{0}]\tshape: {1}'.format(comm.rank, inputs.shape))
    #    print('[{0}]\tsum list (every 100 pixels): {1}'.format(
    #        comm.rank,
    #        [inputs[i:i+100].sum() for i in xrange(0, len(inputs), 100)]
    #    ))
    #    comm.Barrier()
    #else:
    #    print('Parameter inputs:')
    #    print('\tlen: {}'.format(len(inputs)))
    #    print('\tsum: {}'.format(inputs.sum()))
    #    print('\tshape: {}'.format(inputs.shape))
    #    print('\tsum list (every 100 pixels): {}'.format(
    #        [inputs[i:i+100].sum() for i in xrange(0, len(inputs), 100)]
    #    ))
    #if Distributed is True:
    #    if comm.rank == 0:
    #        print('Parameter labels:')
    #    print('[{0}]\tlen: {1}'.format(comm.rank, len(labels)))
    #    print('[{0}]\tsum: {1}'.format(comm.rank, labels.sum()))
    #    print('[{0}]\tshape: {1}'.format(comm.rank, labels.shape))
    #    comm.Barrier()
    #else:
    #    print('Parameter labels:')
    #    print('\tlen: {}'.format(len(labels)))
    #    print('\tsum: {}'.format(labels.sum()))
    #    print('\tshape: {}'.format(labels.shape))
    #if Distributed is True:
    #    if comm.rank == 0:
    #        print('Parameter theta{1,2}:')
    #    print('[{0}]\ttheta1.sum: {1}'.format(comm.rank, theta1.sum()))
    #    print('[{0}]\ttheta2.sum: {1}'.format(comm.rank, theta2.sum()))
    #    print('[{0}]\ttheta1.shape: {1}'.format(comm.rank, theta1.shape))
    #    print('[{0}]\ttheta2.shape: {1}'.format(comm.rank, theta2.shape))
    #    comm.Barrier()
    #else:
    #    print('Parameter theta{1,2}:')
    #    print('\ttheta1.sum: {}'.format(theta1.sum()))
    #    print('\ttheta2.sum: {}'.format(theta2.sum()))
    #    print('\ttheta1.shape: {}'.format(theta1.shape))
    #    print('\ttheta2.shape: {}'.format(theta2.shape))

    # construct neural network
    #   1st half:
    #     input_layer sum: 135486.766496
    #     hidden_layer sum: 33121.1570819
    #     output_layer sum: 2510.58921348
    #   2nd half:
    #     input_layer sum: 132191.493663
    #     hidden_layer sum: 33282.7074777
    #     output_layer sum: 2512.64556285
    #   all:
    #     input_layer sum: 267678.26016
    #     hidden_layer sum: 66403.8645595
    #     output_layer sum: 5023.23477633

    input_layer = np.insert(inputs, 0, 1, axis=1)  # add bias, 5000x401
    #print('-'*80)
    #print('Network structure:')
    #if Distributed is True:
    #    print('[{0}]\tinput_layer sum: {1}, theta1.T sum: {2}'.format(comm.rank, input_layer.sum(), theta1.T.sum()))
    #    print('[{0}]\tinput_layer[0]: {1}'.format(comm.rank, input_layer[0]))
    #    print('[{0}]\ttheta1: {1}'.format(comm.rank, theta1))
    #    comm.Barrier()
    #else:
    #    print('\tinput_layer sum: {0}, theta1.T sum: {1}'.format(input_layer.sum(), theta1.T.sum()))
    #    print('\tinput_layer[0]: {}'.format(input_layer[0]))
    #    print('\ttheta1: {}'.format(theta1))

    time_start = time.time()
    hidden_layer = Matrix_dot(input_layer, theta1.T)
    #if Distributed is True:
    #    print('[{0}]\thidden_layer.sum: {1} (dot)'.format(comm.rank, hidden_layer.sum()))
    #else:
    #    print('\thidden_layer.sum: {} (dot)'.format(hidden_layer.sum()))
    hidden_layer = sigmoid(hidden_layer)
    hidden_layer = np.insert(hidden_layer, 0, 1, axis=1)  # add bias, 5000x26
    time_end = time.time()
    #if Distributed is True:
    #    print('[{0}]\thidden_layer sum: {1}'.format(comm.rank, hidden_layer.sum()))
    #    comm.Barrier()
    #else:
    #    print('\thidden_layer sum: {}'.format(hidden_layer.sum()))
    if comm.rank == 0:
        print('\tconstruction: hidden layer dot costs {} secs'.format(time_end - time_start))

    time_start = time.time()
    output_layer = Matrix_dot(hidden_layer, theta2.T)  # 5000x10
    output_layer = sigmoid(output_layer)
    time_end = time.time()
    #if Distributed is True:
    #    print('[{0}]\toutput_layer sum: {1}'.format(comm.rank, output_layer.sum()))
    #    comm.Barrier()
    #else:
    #    print('\toutput_layer sum: {}'.format(output_layer.sum()))
    if comm.rank == 0:
        print('\tconstruction: output layer dot costs {} secs'.format(time_end - time_start))

    # forward propagation: calculate cost
    #if Distributed is False or comm.rank == 0:
    #    print('-'*80)
    time_start = time.time()
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
        #print('\ttype, outputs: {0}, labels: {1}'.format(type(outputs), type(labels)))
        #print('\tlabels[{0}]: {1}, type: {2}'.format(training_index, labels[training_index], type(labels[training_index])))
        #label_index = int(labels[training_index] - 1)
        #outputs[label_index] = 1
        outputs[labels[training_index]-1] = 1

        for k in xrange(output_layer_size):
            error = -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
            cost += error
        #if training_index % 500 == 0:
        #    if Distributed is True:
        #        print('[{0}]\ttraining input #{1}:'.format(comm.rank, training_index))
        #        print('[{0}]\t\tcurrent cost: {1}\n\t\tlabel[{2}]: {3}\n\t\toutputs: {4}'.format(
        #            comm.rank,
        #            cost,
        #            training_index,
        #            labels[training_index],
        #            outputs
        #        ))
        #        comm.Barrier()
        #    else:
        #        print('\ttraining {0}:'.format(training_index))
        #        print('\t\tcurrent cost: {0}\n\t\tlabel[{1}]: {2}\n\t\toutputs: {3}'.format(cost, training_index, labels[training_index], outputs))
    cost /= len(inputs)
    time_end = time.time()
    if comm.rank == 0:
        print('\tforward prop: costs {} secs'.format(time_end - time_start))

    # back propagation: calculate gradiants
    #if Distributed is False or comm.rank == 0:
    #    print('-'*80)
    time_start = time.time()
    theta1_grad = np.zeros_like(theta1)  # 25x401
    theta2_grad = np.zeros_like(theta2)  # 10x26
    for index in xrange(len(inputs)):
        # transform label y[i] from a number to a vector.
        outputs = np.zeros((1, output_layer_size))  # (1,10)
        outputs[0][labels[index]-1] = 1

        # calculate delta3
        delta3 = (output_layer[index] - outputs).T  # (10,1)

        # calculate delta2
        z2 = Matrix_dot(theta1, input_layer[index:index+1].T)  # (25,401) x (401,1)
        z2 = np.insert(z2, 0, 1, axis=0)  # add bias, (26,1)
        delta2 = np.multiply(
            Matrix_dot(theta2.T, delta3),  # (26,10) x (10,1)
            sigmoid_gradient(z2)  # (26,1)
        )
        delta2 = delta2[1:]  # (25,1)

        # calculate gradients of theta1 and theta2
        # (25,401) = (25,1) x (1,401)
        theta1_grad += Matrix_dot(delta2, input_layer[index:index+1])
        # (10,26) = (10,1) x (1,26)
        theta2_grad += Matrix_dot(delta3, hidden_layer[index:index+1])
        #if Distributed is False or comm.rank == 0:
        #    if index % 500 == 0:
        #        print('\ttraining {0}: theta1_grad.sum: {1}, theta2_grad.sum: {2}'.format(index, theta1_grad.sum(), theta2_grad.sum()))
    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)
    time_end = time.time()
    if comm.rank == 0:
        print('\tback prop: costs {} secs'.format(time_end - time_start))

    #print('-'*80)
    #if Distributed is True:
    #    print('[{0}]\tcost: {1}, theta1_grad.sum: {2}, theta2_grad.sum: {3}'.format(comm.rank, cost, theta1_grad.sum(), theta2_grad.sum()))
    #else:
    #    print('\tcost: {0}, theta1_grad.sum: {1}, theta2_grad.sum: {2}'.format(cost, theta1_grad.sum(), theta2_grad.sum()))
    return cost, (theta1_grad, theta2_grad)


def gradient_descent(inputs, labels, learningrate=0.8, iteration=50):
    '''
    @return cost and trained model (weights).
    '''
    if Distributed is True:
        if comm.rank == 0:
            rand_theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
            rand_theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
            theta1 = rand_theta1
            theta2 = rand_theta2

            # DEBUG: Load fixed weights instead of random weights to get the
            # same results everytime.
            #theta1, theta2 = load_weights()
        else:
            theta1 = np.zeros((Hidden_layer_size, Input_layer_size + 1))
            theta2 = np.zeros((Output_layer_size, Hidden_layer_size + 1))
        comm.Barrier()
        if comm.rank == 0:
            time_bcast_start = time.time()
        comm.Bcast([theta1, MPI.DOUBLE])
        comm.Barrier()
        comm.Bcast([theta2, MPI.DOUBLE])
        if comm.rank == 0:
            time_bcast_end = time.time()
            print('\tBcast theta1 and theta2 uses {} secs.'.format(time_bcast_end - time_bcast_start))
    else:
        theta1, theta2 = load_weights()
    #if Distributed is True and comm.rank == 0:
    #    print('[0] theta{1,2} must be the same in worknodes.')
    #print('theta1 sum: {0}, theta2 sum: {1}'.format(theta1.sum(), theta2.sum()))

    cost = 0.0
    for i in xrange(iteration):
        time_iter_start = time.time()

        # scatter training data and labels
        sliced_inputs = np.asarray(np.split(inputs, comm.size))
        sliced_labels = np.asarray(np.split(labels, comm.size))
        inputs_buf = np.zeros((len(inputs)/comm.size, Input_layer_size))
        labels_buf = np.zeros((len(labels)/comm.size), dtype='uint8')
        #print('sum, inputs: {0}, labels: {1}'.format(inputs.sum(), labels.sum()))

        # Iteration 1:
        #   local:
        #     cost: 0.287629165161
        #     theta1_grad sum: -0.0427745251859
        #     theta2_grad sum: 0.0688219590388
        #   distributed:
        #     cost: 5.47106596411
        #     theta1_grad sum: 52.7399899711
        #     theta2_grad sum: -9.33771026269
        if Distributed is True:
            # Sum of the 1st half inputs is 132986.76649618745
            # Sum of the 2nd half inputs is 129691.49366349436
            comm.Barrier()
            if comm.rank == 0:
                time_scatter_start = time.time()
            comm.Scatter(sliced_inputs, inputs_buf)
            #print('[{0}] inputs_buf.sum: {1}'.format(comm.rank, inputs_buf.sum()))
            if comm.rank == 0:
                time_scatter_end = time.time()
                print('\tScatter inputs uses {} secs.'.format(time_scatter_end - time_scatter_start))

            # Sum of the 1st half labels is 10000
            # Sum of the 2nd half labels is 17500
            comm.Barrier()
            if comm.rank == 0:
                time_scatter_start = time.time()
            comm.Scatter(sliced_labels, labels_buf)
            #print('[{0}] labels_buf.sum: {1}'.format(comm.rank, labels_buf.sum()))
            if comm.rank == 0:
                time_scatter_end = time.time()
                print('\tScatter labels uses {} secs.'.format(time_scatter_end - time_scatter_start))

            # Calculate distributed costs and gradients of this iteration
            # by cost function.
            comm.Barrier()
            cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
                Input_layer_size, Hidden_layer_size, Output_layer_size,
                inputs_buf, labels_buf, regular=0)

            # Gather distributed costs and gradients.
            comm.Barrier()
            cost_buf = [0] * comm.size
            try:
                cost_buf = comm.gather(cost)
                cost = sum(cost_buf) / len(cost_buf)
            except TypeError as e:
                print('[{0}] {1}'.format(comm.rank, e))
            #if comm.rank == 0:
            #    print('Gathered cost: {}'.format(cost))

            theta1_grad_buf = np.asarray([np.zeros_like(theta1_grad)] * comm.size)
            comm.Barrier()
            if comm.rank == 0:
                time_gather_start = time.time()
            comm.Gather(theta1_grad, theta1_grad_buf)
            if comm.rank == 0:
                time_gather_end = time.time()
                print('\tGather theta1 uses {} secs.'.format(time_gather_end - time_gather_start))
            comm.Barrier()
            theta1_grad = functools.reduce(np.add, theta1_grad_buf) / comm.size
            #if comm.rank == 0:
            #    print('Gathered theta1_grad sum: {}'.format(theta1_grad.sum()))

            theta2_grad_buf = np.asarray([np.zeros_like(theta2_grad)] * comm.size)
            comm.Barrier()
            if comm.rank == 0:
                time_gather_start = time.time()
            comm.Gather(theta2_grad, theta2_grad_buf)
            if comm.rank == 0:
                time_gather_end = time.time()
                print('\tGather theta2 uses {} secs.'.format(time_gather_end - time_gather_start))
            comm.Barrier()
            theta2_grad = functools.reduce(np.add, theta2_grad_buf) / comm.size
            #if comm.rank == 0:
            #    print('Gathered theta2_grad sum: {}'.format(theta2_grad.sum()))
        else:
            #cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            #    Input_layer_size, Hidden_layer_size, Output_layer_size,
            #    inputs[:len(inputs)/2], labels[:len(labels)/2], regular=0)
            #print('[part1] cost: {0}, theta1_grad.sum: {1}, theta2_grad.sum: {2}'.format(cost, theta1_grad.sum(), theta2_grad.sum()))
            #cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            #    Input_layer_size, Hidden_layer_size, Output_layer_size,
            #    inputs[len(inputs)/2:], labels[len(labels)/2:], regular=0)
            #print('[part2] cost: {0}, theta1_grad.sum: {1}, theta2_grad.sum: {2}'.format(cost, theta1_grad.sum(), theta2_grad.sum()))
            cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
                Input_layer_size, Hidden_layer_size, Output_layer_size,
                inputs, labels, regular=0)
            #print('[all] cost: {0}, theta1_grad.sum: {1}, theta2_grad.sum: {2}'.format(cost, theta1_grad.sum(), theta2_grad.sum()))

        #print('theta1_grad.sum: {0}, theta2_grad.sum: {1}'.format(theta1_grad.sum(), theta1_grad.sum()))

        theta1 -= learningrate * theta1_grad
        theta2 -= learningrate * theta2_grad

        if Distributed is True:
           comm.Bcast([theta1, MPI.DOUBLE])
           comm.Bcast([theta2, MPI.DOUBLE])
           comm.Barrier()

        time_iter_end = time.time()
        if comm.rank == 0:
            print('Iteration {0} (learning rate {2}, iteration {3}), cost: {1}, time: {4}'.format(
                i+1, cost, learningrate, iteration, time_iter_end - time_iter_start)
            )
    return cost, (theta1, theta2)


def train(inputs, labels, learningrate=0.8, iteration=50):
    cost, model = gradient_descent(inputs, labels, learningrate, iteration)
    return model


def predict(model, inputs):
    theta1, theta2 = model
    a1 = np.insert(inputs, 0, 1, axis=1)  # add bias, (5000,401)
    a2 = np.dot(a1, theta1.T)  # (5000,401) x (401,25)
    a2 = sigmoid(a2)
    a2 = np.insert(a2, 0, 1, axis=1)  # add bias, (5000,26)
    a3 = np.dot(a2, theta2.T)  # (5000,26) x (26,10)
    a3 = sigmoid(a3)  # (5000,10)
    return [i.argmax()+1 for i in a3]


if __name__ == '__main__':
    if Gpu_mode is True:
        print('GPU mode')
        Matrix_dot = gpu_matrix_dot()
    else:
        print('CPU mode')
        Matrix_dot = np.dot

    # Note: There are 10 units which present the digits [1-9, 0]
    # (in order) in the output layer.
    inputs, labels = load_training_data()
    #print('inputs is F-order: {0}, is aligned: {1}'.format(np.isfortran(inputs), inputs.flags['ALIGNED']))
    #print('labels is F-order: {0}, is aligned: {1}'.format(np.isfortran(labels), labels.flags['ALIGNED']))
    #if comm.rank == 0:
    #    print('labels, shape: {0}, len: {1}, labels: {2}'.format(labels.shape, len(labels), labels))
    #    print('inputs flags: {}'.format(inputs.flags))
    #    print('labels flags: {}'.format(labels.flags))

    # (optional) load pre-trained model for debugging neural network construction
    theta1, theta2 = load_weights()
    #print('theta1 is F-order: {0}, is aligned: {1}'.format(np.isfortran(theta1), theta1.flags['ALIGNED']))
    #print('theta2 is F-order: {0}, is aligned: {1}'.format(np.isfortran(theta2), theta2.flags['ALIGNED']))

    # FIXME: Memory alignment issue in Scipy.
    #   This issue leads Theano to complain that "The numpy.ndarray
    #   object is not aligned. Theano C code does not support that."
    #
    #   Related discussion: http://stackoverflow.com/questions/36321400/strange-typeerror-with-theano/36323861
    # workaround to avoid memory alignment error in Scipy
    #theta1 = np.array(theta1)
    #theta2 = np.array(theta2)
    #print('theta1 is F-order: {0}, is aligned: {1}'.format(np.isfortran(theta1), theta1.flags['ALIGNED']))
    #print('theta2 is F-order: {0}, is aligned: {1}'.format(np.isfortran(theta2), theta2.flags['ALIGNED']))
    #print('theta1: {}'.format(theta1))
    #print('theta1 flags: {}'.format(theta1.flags))
    #print('theta2: {}'.format(theta2))
    #print('theta2 flags: {}'.format(theta2.flags))
 
    #cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2, Input_layer_size, Hidden_layer_size, Output_layer_size, inputs, labels, regular=0)
    #print('cost:', cost)


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

    # Load pretrained weights for debugging precision.
    # The precision will be around 97%.
    #weights = load_weights()
    #theta1 = np.asarray(weights['Theta1'])  # size: 25 entries, each has 401 numbers
    #theta2 = np.asarray(weights['Theta2'])  # size: 10 entries, each has  26 numbers
    #model = (theta1, theta2)

    outputs = predict(model, inputs)

    correct_prediction = 0
    for i, predict in enumerate(outputs):
        if predict == labels[i]:
            correct_prediction += 1
    precision = float(correct_prediction) / len(labels)
    # Precision of pretrained model is 0.9756
    print('precision: {}'.format(precision))
