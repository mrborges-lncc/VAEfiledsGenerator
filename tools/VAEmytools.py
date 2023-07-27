#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:55:06 2023

@author: mrborges
"""
import sys
import math
import random
import numpy as np
import tensorflow as tf
import PIL
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
print('Tensorflow version == ',tf.__version__)

###############################################################################
def plot_latent_images(model, n, digit_size=50):
    """Plots n x n digit images decoded from the latent space."""
    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size*n
    image_height = image_width
    image = np.zeros((image_height, image_width))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            image[i * digit_size: (i + 1) * digit_size,
                  j * digit_size: (j + 1) * digit_size] = digit.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.show()
###############################################################################

###############################################################################
def display_image(epoch_no):
    return PIL.Image.open('figuras/image_at_epoch_{:04d}.png'.format(epoch_no))
###############################################################################

###############################################################################
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)
###############################################################################

###############################################################################
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,
                                                        labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz   = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
###############################################################################

###############################################################################
@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters."""
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
###############################################################################

###############################################################################
def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(test_sample[i, :, :, 0], cmap='jet')
        plt.axis('off')
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='jet')
        plt.axis('off')
    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('figuras/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
###############################################################################
    
###############################################################################
def generate_numbers(model, epoch, test_sample):
    number = np.expand_dims(test_sample[0],axis=0)
    mean, logvar = model.encode(number)
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        z = model.reparameterize(mean, logvar)
#        print('1',z)
        z = tf.random.normal(shape=mean.shape)
#        print('2',z)
        plt.subplot(4, 4, i + 1)
        predictions = model.sample(z)
        plt.imshow(predictions[0, :, :, 0], cmap='jet')
        plt.axis('off')
    plt.show()
###############################################################################

###############################################################################
class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, latent_dim, inputshape):
        super(CVAE, self).__init__()
        filter1 = 32
        filter2 = 64
        print(inputshape)
#        sys.exit()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=inputshape),
                tf.keras.layers.Conv2D(filters=filter1, kernel_size=2,
                                       strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(filters=filter2, kernel_size=2, 
                                       strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=filter2, kernel_size=2, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=filter1, kernel_size=2, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )
    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self,x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
###############################################################################

###############################################################################
def plot_examples(images):
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
def load_dataset(dataname,prep,namein):
    if dataname == 'MNIST':
        (train, _), (test, _) = tf.keras.datasets.mnist.load_data()
    if dataname == 'FASHION_MNIST':
        (train, _), (test, _) = tf.keras.datasets.fashion_mnist.load_data()
    if dataname == 'PERM':
        train, test = load_PERM(namein)
#==========================================================
    train = preprocess_images(train,dataname,prep)
    test  = preprocess_images(test,dataname,prep)
    return train, test
###############################################################################

###############################################################################
def preprocess_images(images,dataname,prep):
    if dataname == 'PERM':
        if prep:
            maxdata = np.max(images)
            mindata = np.min(images)
            images  = (images - mindata) / (maxdata - mindata)
            images  = images.reshape((images.shape[0], 50, 50, 3))
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
def loaddataPERM(nx,ny,nz,data_size,namein):
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
def load_PERM(namein):
    nx  = 50
    ny  = 50
    nz  = 3
    data_size = 20000                # size of dataset 
    ###########################################################################
    # LOAD DATA ###############################################################
    # name of data files
    data   = loaddataPERM(nx,ny,nz,data_size,namein)
    lista  = list(range(data_size))
    ntrain = math.ceil(0.8 * data_size)
    random.shuffle(lista)
    train  = data[lista[0:ntrain],:,:,:]
    test   = data[lista[ntrain:data_size],:,:,:]
    
    return train, test
##############################################################################
