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
from mytools import load_dataset, plot_examples, comparison, plot_latent_hist
from mytools import build_encoder3D, build_decoder3D, build_encoder2D
from mytools import build_decoder2D, net_info, fieldplot3D, plot_losses, plot_3D
###############################################################################
###############################################################################
# Load data base ==============================================================
Lx  = 1.0
Ly  = 1.0
Lz  = 0.01
nx  = 100
ny  = 100
nz  = 1
num_channel = 1
if nz == 1:
    input_shape= (nx, ny, num_channel)
else:
    input_shape= (nx, ny, nz, num_channel)
#==============================================================================
data_size  = 20000
home       = '/prj/prjmurad/mrborges/Dropbox/fieldsCNN/'
namein     = home + 'sexp_1.00x1.00x0.01_28x28x1_l0.10x0.10x0.10_20000.mat'
namein     = home + 'exp_1.00x1.00x0.29_28x28x8_l0.10x0.10x0.05_2000.mat'
namein     = home + 'exp_1.00x1.00x0.01_100x100x1_l0.20x0.20x0.00_20000.mat'
porous     = False
porosity   = 0.20
infoperm   = perm_info(namein, porous, input_shape, data_size, porosity, 
                       Lx, Ly, Lz, nx, ny, nz)
#==============================================================================
data_name  = ['MNIST', 'PERM', 'FASHION_MNIST']
dataname   = data_name[1]
name_ext   = '_teste'
namefig    = './figuras/' + dataname + name_ext
preprocess = False
train_images, test_images = load_dataset(dataname,preprocess,infoperm)
plot_examples(train_images, namefig)
print("Data interval [%g,%g]" % (np.min(train_images),np.max(train_images)))
if nz > 1:
    #fieldplot3D(train_images[0,:,:,:],Lx,Ly,Lz,nx,ny,nz,dataname) 
    plot_3D(train_images[0,:,:,:], infoperm, namefig)
#sys.exit()
###############################################################################
# Parameters ==================================================================
train_size = np.size(train_images,0)
batch_size = 128
test_size  = np.size(test_images,0)
inputshape = train_images.shape[1:]
lrate      = 1.e-4
optimizer  = tf.keras.optimizers.Adam(learning_rate = lrate)
epochs     = 1000
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 128
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
net          = net_info(conv_filters, conv_strides, conv_kernels, conv_activat, 
                        conv_padding, dens_neurons, dens_activat)
#==============================================================================
if input_shape[2] == 1:
    encoder = build_encoder2D(net, input_shape, latent_dim)
else:
    encoder = build_encoder3D(net, input_shape, latent_dim)
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
if input_shape[2] == 1: 
    decoder = build_decoder2D(net, input_shape, latent_dim, ndens,
                              layer_shape, nconv)
else:        
    decoder = build_decoder3D(net, input_shape, latent_dim, ndens,
                              layer_shape, nconv)
#==============================================================================
###############################################################################
# Train the VAE ===============================================================
vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizer)
history = vae.fit(train_images, epochs=epochs, batch_size=batch_size)
# saving model ================================================================
name = dataname + name_ext
save_model_weights(vae, name, latent_dim)
plot_losses(history, namefig)
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
Zparam = plot_latent_hist(vae, train_images, latent_dim, namefig, 16)
#==============================================================================
###############################################################################
# Generator ===================================================================
zmu,zvar,z = fieldgenerator(vae, latent_dim, input_shape, Zparam, 
                            namefig, infoperm, 16)
#==============================================================================
###############################################################################
# Comparison between data and predictions =====================================
comparison(vae, train_images, latent_dim, input_shape, namefig, infoperm)
#==============================================================================
