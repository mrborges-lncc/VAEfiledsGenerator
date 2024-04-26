#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:12:13 2023

@author: mrborges
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
sys.path.append("./tools/")
from mytools import load_model_weights, VAE, plot_latent_hist, fieldgenerator
from mytools import comparison, perm_info, load_dataset, plot_examples
from mytools import random_generator, save_original_fields, plot_3D
from mytools import plot_label_clusters
###############################################################################
###############################################################################
# Load data base ==============================================================
Lx  = 100.0
Ly  = 100.0
Lz  = 0.01
nx  = 28
ny  = 28
nz  = 1
num_channel = 1
if nz == 1:
    input_shape= (nx, ny, num_channel)
else:
    input_shape= (nx, ny, nz, num_channel)
#==============================================================================
data_size  = 35000
ratio_valid= 0.05
ratio_test = 0.05
home       = '/home/mrborges/Dropbox/fieldsCNN/'
home       = '/prj/prjmurad/mrborges/Dropbox/fieldsCNN/'
home       = '/media/mrborges/borges/fieldsCNN/'
namein     = home + 'mix_100.00x100.00x0.01_100x100x1_35000.mat'
porous     = False
porosity   = 0.20
infoperm   = perm_info(namein, porous, input_shape, data_size, porosity, 
                       Lx, Ly, Lz, nx, ny, nz)
#==============================================================================
data_name  = ['MNIST', 'PERM', 'FASHION_MNIST']
dataname   = data_name[0]
name_ext   = '_MNIST_ls2'
namefig    = './figuras/' + dataname + name_ext
preprocess = True # Normalize
train_images, valid_images, test_images = load_dataset(dataname,preprocess,
                                                       infoperm,ratio_valid,
                                                       ratio_test)
plot_examples(train_images, namefig)
print("Data interval [%g,%g]" % (np.min(train_images),np.max(train_images)))
#if nz > 1:
    #fieldplot3D(train_images[0,:,:,:],Lx,Ly,Lz,nx,ny,nz,dataname) 
    #plot_3D(train_images[0,:,:,:], infoperm, namefig)

###############################################################################
ld = 2
name = dataname + name_ext
encoder, decoder = load_model_weights(name, ld)
print(encoder.summary())
print(decoder.summary())
#==============================================================================
###############################################################################
# Collecting information for building the mirrored decoder ====================
outputs = [layer.output for layer in encoder.layers]
cont = 0
for i in outputs:
  name = i.name
  cont += 1
layer = outputs[cont-1]
latent_dim = layer.shape[-1]
###############################################################################
# Train the VAE ===============================================================
vae = VAE(encoder, decoder)
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
comparison(vae, test_images, latent_dim, input_shape, namefig, infoperm)
#==============================================================================
#==============================================================================
###############################################################################
# Random generation ===========================================================
fname = './KLE/fields/field_' + dataname + '_'
N = 50
random_generator(decoder, latent_dim, input_shape, Zparam, fname, N)
#==============================================================================
###############################################################################
# Save original fields ========================================================
fname = './KLE/fields/field_orig_' + dataname + '_'
save_original_fields(train_images, input_shape, fname, N)
