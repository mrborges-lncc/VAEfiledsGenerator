#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:07:26 2023
@author: mrborges
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys
import random
import math
###############################################################################
###############################################################################
def plot_examples(images):
    '''Display a 5x7 plot of 35 images'''
    fig = plt.figure(figsize=(14,10))
    M = np.size(images,0)
    cor = 'jet'
    for n in range(1, 36):
        fig.add_subplot(5, 7, n)
        nr  = random.randint(0,M-1)
        img = images[nr,:,:,0]
        plt.imshow(img, cmap=cor, aspect='equal', interpolation='none', 
                   alpha = 1.0, origin='upper')
        plt.axis('off')
    #name = 'figuras/example_' + namep + '.png'
    #plt.savefig(name, transparent=True, dpi=300)
    plt.show()
###############################################################################

###############################################################################
###############################################################################
def load_dataset(dataname,prep,namein,inputshape,datasize):
    '''Load the dataset, split it into training and test sets, and scale then'''
    if dataname == 'MNIST':
        (train, _), (test, _) = tf.keras.datasets.mnist.load_data()
    if dataname == 'FASHION_MNIST':
        (train, _), (test, _) = tf.keras.datasets.fashion_mnist.load_data()
    if dataname == 'PERM':
        train, test = load_PERM(namein,inputshape,datasize)
#==========================================================
    train = preprocess_images(train,dataname,prep)
    test  = preprocess_images(test,dataname,prep)
    return train, test
###############################################################################

###############################################################################
###############################################################################
def preprocess_images(images,dataname,prep):
    '''Normalize and reshape the images'''
    if dataname == 'PERM':
        nx  = images.shape[1]
        ny  = images.shape[2]
        nz  = images.shape[3]
        if prep:
            maxdata = np.max(images)
            mindata = np.min(images)
            images  = (images - mindata) / (maxdata - mindata)
            images  = images.reshape((images.shape[0], nx, ny, nz))
            return images.astype('float32')
        else:
            return images.astype('float32')
    else:
        if prep:
            images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
            return np.where(images > .5, 1.0, 0.0).astype('float32')
        else:
            images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
            return images.astype('float32')
###############################################################################

###############################################################################
###############################################################################
def loaddataPERM(nx,ny,nz,data_size,namein):
    '''Load the dataset, split it into training and test sets, and scale then'''
    datain  = np.fromfile(namein, dtype=np.float32)
    datain  = np.reshape(datain, (data_size,nx*ny*nz))
    data_in = np.zeros((data_size,nx,ny,nz),dtype=np.float32)
    for i in range(data_size):
        n = 0
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    data_in[i,x,y,z] = datain[i,n]
                    n += 1
    return data_in
###############################################################################

###############################################################################
###############################################################################
def load_PERM(namein,inputshape,datasize):
    '''Load the dataset, split it into training and test sets, and scale then'''
    nx  = inputshape[0]
    ny  = inputshape[1]
    nz  = inputshape[2]
    data_size = datasize  # size of dataset 
    #==========================================================================
    # LOAD DATA ===============================================================
    # name of data files
    data   = loaddataPERM(nx,ny,nz,data_size,namein)
    lista  = list(range(data_size))
    ntrain = math.ceil(0.98 * data_size)
    random.shuffle(lista)
    train  = data[lista[0:ntrain],:,:,:]
    test   = data[lista[ntrain:data_size],:,:,:]
    
    return train, test
###############################################################################

###############################################################################
###############################################################################
class Sampling(layers.Layer):
    '''Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.'''
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
###############################################################################
###############################################################################
class VAE(keras.Model):
    '''Combines the encoder and decoder into an end-to-end model for training.'''
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
###############################################################################

###############################################################################
###############################################################################
def plot_latent_space(vae, n=15, figsize=15):
    '''Display a n*n 2D manifold of digits'''
    # display an n*n 2D manifold of digits
    digit_size = 28
    A = -4.
    B = 4.
    C = -4
    D = 4.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(A, B, n)
    grid_y = np.linspace(C, D, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
###############################################################################

###############################################################################
###############################################################################
def fieldgenerator(model,latent_dim,nf):
    '''Display a n*n 2D manifold of digits'''
    outputs = [layer.output for layer in model.encoder.layers]
    cont = 0
    for i in outputs:
      name = i.name
      if name[0:6] == 'z_mean':
        j = cont
      if name[0:6] == 'z_log_':
        m = cont
      cont += 1
    
    mean_layer = model.encoder.layers[j]
    std_layer  = model.encoder.layers[m]
    zmean      = mean_layer.get_weights()[1]
    zlogvar    = std_layer.get_weights()[1]   
    z          = reparameterize(zmean, zlogvar)
    z          = z.numpy()
    z          = z.reshape((1, latent_dim))
#    z          = np.array([[-20,20]])
    x_decoded  = model.decoder.predict(z)
    figure     = x_decoded[0].reshape(28,28)
    plt.imshow(figure, cmap="jet")
    return zmean, zlogvar, z
###############################################################################

###############################################################################
###############################################################################
def plot_label_clusters(vae, data, labels):
    '''Display a 2D plot of the digit classes in the latent space'''
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
###############################################################################

###############################################################################
###############################################################################
def reparameterize(mean, logvar):
    '''Reparameterization trick by sampling from an isotropic unit Gaussian.'''
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean
###############################################################################

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

###############################################################################
###############################################################################
###############################################################################
###############################################################################

###############################################################################
###############################################################################
###############################################################################
###############################################################################
