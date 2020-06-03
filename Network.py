# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:54:44 2019

@author: no295d
"""

import numpy as np
import tensorflow as tf

class Network(object):
    
    def __init__(self, input_size, latent_size,
                 encoder_num_units=[100, 100], decoder_num_units=[100, 100], name='Unnamed',
                 tot_epochs=0, load_file=None):
        """
         Parameters:
         input_size: length of a single data vector.
         latent_size: number of latent neurons to be used.
         encoder_num_units, decoder_num_units: Number of neurons in encoder and decoder hidden layers. Everything is fully connected.
         name: Used for tensorboard
         tot_epochs and  load_file are used internally for loading and saving, don't pass anything to them manually.
        """

        self.graph = tf.Graph()

        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.name = name
        self.tot_epochs = tot_epochs

        # Set up neural network
        self.graph_setup()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            initialize_uninitialized(self.session)

        # Load saved network
        self.load_file = load_file
        if self.load_file is not None:
            self.load(self.load_file)


    def graph_setup(self):
        """
        Set up the computation graph for the neural network based on the parameters set at initialization
        """
        with self.graph.as_default():

            #######################
            # Define placeholders #
            #######################
            self.input = tf.placeholder(tf.float32, [None, self.input_size], name='input')
            self.centers = tf.placeholder(tf.float32, [None, self.input_size], name='centers')
            self.forgetting_factor = tf.placeholder(tf.float32, shape=[], name='forgetting_factor')
            self.c = tf.placeholder(tf.float32, shape=[], name='c')

            ##########################################
            # Set up variables and computation graph #
            ##########################################
            with tf.variable_scope('encoder'):
                temp_layer = self.input

                # input and output dimensions for each of the weight tensors
                enc_in_dims = [self.input_size] + self.encoder_num_units
                enc_out_dims = self.encoder_num_units + [2 * self.latent_size]

                for k in range(len(enc_in_dims)):
                    with tf.variable_scope('{}th_enc_layer'.format(k)):
                        w = tf.get_variable('w', [enc_in_dims[k], enc_out_dims[k]],
                                            initializer=tf.initializers.random_normal(stddev=2. / np.sqrt(enc_in_dims[k] + enc_out_dims[k])))
                        b = tf.get_variable('b', [enc_out_dims[k]],
                                            initializer=tf.initializers.constant(0.))
                        squash = ((k + 1) != len(enc_in_dims))  # don't squash latent layer
                        temp_layer = forwardprop(temp_layer, w, b, name='enc_layer_{}'.format(k), squash=squash)

            with tf.variable_scope('latent_layer'):
                self.log_sigma = temp_layer[:, :self.latent_size]
                self.mu = temp_layer[:, self.latent_size:]
                self.mu_sample = tf.add(self.mu, tf.exp(self.log_sigma) * self.epsilon, name='add_noise')
            with tf.name_scope('kl_loss'):
                self.kl_loss = kl_divergence(self.mu, self.log_sigma, dim=self.latent_size)

            with tf.variable_scope('decoder'):
                temp_layer = self.mu_sample

                dec_in_dims = [self.latent_size] + self.decoder_num_units
                dec_out_dims = self.decoder_num_units + [self.input_size]
                for k in range(len(dec_in_dims)):
                    with tf.variable_scope('{}th_dec_layer'.format(k)):
                        w = tf.get_variable('w', [dec_in_dims[k], dec_out_dims[k]],
                                            initializer=tf.initializers.random_normal(stddev=2. / np.sqrt(dec_in_dims[k] + dec_out_dims[k])))
                        b = tf.get_variable('b', [dec_out_dims[k]],
                                            initializer=tf.initializers.constant(0.))
                        squash = ((k + 1) != len(dec_in_dims))  # don't squash latent layer
                        temp_layer = forwardprop(temp_layer, w, b, name='dec_layer_{}'.format(k), squash=squash)

                self.output = temp_layer

            with tf.name_scope('recon_loss'):
                self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.input, self.output), axis=1))

            #####################
            # Cost and training #
            #####################
            with tf.name_scope('cost'):
                self.cost = self.recon_loss + self.beta * self.kl_loss
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gvs = optimizer.compute_gradients(self.cost)
                capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
                self.training_op = optimizer.apply_gradients(capped_gvs)

            #########################
            # Tensorboard summaries #
            #########################
            tf.summary.histogram('latent_means', self.mu)
            tf.summary.histogram('latent_log_sigma', self.log_sigma)
            tf.summary.histogram('ouput_means', self.output)
            tf.summary.scalar('recon_loss', self.recon_loss)
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('beta', self.beta)

            self.summary_writer = tf.summary.FileWriter(io.tf_log_path + self.name + '/', graph=self.graph)
            self.summary_writer.flush()  # write out graph
            self.all_summaries = tf.summary.merge_all()


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))