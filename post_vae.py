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
from mytools import conference, perm_info, load_dataset, plot_examples
from mytools import random_generator
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
ld = 2
encoder, decoder = load_model_weights(dataname, ld)
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
###############################################################################
# Random generation ===========================================================
fname = './KLE/fields/field_' + dataname + '_'
N = 5000
random_generator(decoder, latent_dim, input_shape, Zparam, fname, N)