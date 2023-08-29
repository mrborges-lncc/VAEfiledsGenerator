#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:05:56 2023

@author: mrborges
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
sys.path.append("./tools/")
from mytools import Sampling, VAE, fieldgenerator, plot_latent_space
from mytools import plot_label_clusters, perm_info, save_model_weights
from mytools import load_dataset, plot_examples, conference, plot_latent_hist
###############################################################################
###############################################################################
# Load data base ==============================================================
input_shape= (28, 28, 1)
data_size  = 20000
home       = './KLE/fields/'
namein     = home + 'sexp_1.00x1.00x0.06_50x50x3_l0.10x0.10x0.05_20000.mat'
namein     = home + 'sexp_1.00x1.00x0.01_28x28x1_l0.10x0.10x0.10_20000.mat'
porous     = False
porosity   = 0.20
infoperm   = perm_info(namein, porous, input_shape, data_size, porosity)
#==============================================================================
data_name  = ['MNIST', 'PERM', 'FASHION_MNIST']
dataname   = data_name[1]
namefig    = './figuras/' + dataname
preprocess = False
train_images, test_images = load_dataset(dataname,preprocess,infoperm)
plot_examples(train_images, namefig)
print("Data interval [%g,%g]" % (np.min(train_images),np.max(train_images)))
###############################################################################
# Parameters ==================================================================
train_size = np.size(train_images,0)
batch_size = 128
test_size  = np.size(test_images,0)
input_shape= train_images.shape[1:]
lrate      = 1.e-4
optimizer  = tf.keras.optimizers.Adam(learning_rate = lrate)
epochs     = 500
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 32
num_examples_to_generate = 16
#==============================================================================
###############################################################################
# Build the encoder ===========================================================
conv_filters = [64, 64, 64, 64, 64, 64]
conv_strides = [2, 1, 1, 1, 1, 1, 1]
conv_kernels = [2, 2, 2, 2, 2, 2, 2]
conv_activat = ["relu", "relu", "relu", "relu", "relu", "relu", "relu"]
conv_padding = ["same", "same", "same", "same", "same", "same", "same"]
dens_neurons = [256, 128, 128]
dens_activat = ["relu", "relu", "relu", "relu"]
#==============================================================================
encoder_inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(filters = conv_filters[0], kernel_size = conv_kernels[0],
                  strides = conv_strides[0], activation = conv_activat[0],
                  padding = conv_padding[0])(encoder_inputs)
for i in range(1,len(conv_filters)):
  x = layers.Conv2D(filters = conv_filters[i], kernel_size = conv_kernels[i],
                    strides = conv_strides[i], activation = conv_activat[i],
                    padding = conv_padding[i])(x)
x = layers.Flatten()(x)
for i in range(len(dens_neurons)):
  x = layers.Dense(dens_neurons[i], activation = dens_activat[i])(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling(name='z')([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()
#==============================================================================
###############################################################################
# Collecting information for building the mirrored decoder ====================
outputs = [layer.output for layer in encoder.layers]
cont = 0
for i in outputs:
  name = i.name
  if name[0:7] == 'flatten':
    j = cont
  cont += 1
layer = outputs[j-1]
layer_shape = layer.shape[1:]
layerdense  = layer_shape[0] * layer_shape[1] * layer_shape[2]
ndens = len(dens_neurons) - 1
nconv = len(conv_filters) - 1
#==============================================================================
###############################################################################
# Build the decoder ===========================================================
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(dens_neurons[ndens],
                 activation = dens_activat[ndens])(latent_inputs)
for i in range(ndens-1,-1,-1):
  x = layers.Dense(dens_neurons[i], activation = dens_activat[i])(x)

x = layers.Dense(layer_shape[0] * layer_shape[1] * layer_shape[2], 
                 activation="relu")(x)
x = layers.Reshape((layer_shape[0], layer_shape[1], layer_shape[2]))(x)

for i in range(nconv,-1,-1):
  x = layers.Conv2DTranspose(filters = conv_filters[i], 
                             kernel_size = conv_kernels[i],
                             strides = conv_strides[i], 
                             activation = conv_activat[i],
                             padding = conv_padding[i])(x)
decoder_outputs = layers.Conv2DTranspose(filters = input_shape[-1],
                                         kernel_size = conv_kernels[0],
                                         activation="linear", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
#==============================================================================
###############################################################################
# Train the VAE ===============================================================
vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizer)
vae.fit(train_images, epochs=epochs, batch_size=batch_size)
# saving model ================================================================
save_model_weights(vae, dataname)
#==============================================================================
###############################################################################
# Display how the latent space clusters different digit classes ===============
if dataname == 'MNIST':
    if latent_dim == 2:
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255
        plot_label_clusters(vae, x_train, y_train, namefig)
        # Display a grid of sampled digits ====================================
        plot_latent_space(vae, namefig)
#==============================================================================
###############################################################################
# Show latent space ===========================================================
Zparam = plot_latent_hist(vae, train_images, latent_dim, namefig, 64)
#==============================================================================
###############################################################################
# Generator ===================================================================
zmu,zvar,z = fieldgenerator(vae, latent_dim, input_shape, Zparam, namefig, 36)
#==============================================================================
###############################################################################
# Comparison between data and predictions =====================================
conference(vae, train_images, latent_dim, input_shape, namefig)
#==============================================================================