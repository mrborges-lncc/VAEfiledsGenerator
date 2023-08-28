#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:51:35 2023

@author: mrborges
"""
from IPython import display
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import imageio
import glob
sys.path.append("./tools/")
from VAEmytools import load_dataset, plot_examples, CVAE, generate_numbers
from VAEmytools import generate_and_save_images, train_step, compute_loss
from VAEmytools import display_image, plot_latent_images
import tensorflow as tf

###############################################################################
# Load data base ==============================================================
home       = './'
namein     = home + 'KLE/fields/sexp_1.00x1.00x0.06_50x50x3_l0.10x0.10x0.05_20000.mat'
namein     = home + 'KLE/fields/sexp_1.00x1.00x0.06_28x28x1_l0.10x0.10x0.05_20000.mat'
data_name  = ['MNIST', 'PERM', 'FASHION_MNIST']
dataname   = data_name[0]
preprocess = True
train_images, test_images = load_dataset(dataname,preprocess,namein)
plot_examples(train_images)
###############################################################################
# Parameters ==================================================================
train_size = np.size(train_images,0)
batch_size = 32
test_size  = np.size(test_images,0)
input_shape= train_images.shape[1:]
lrate      = 5e-4
optimizer  = tf.keras.optimizers.Adam(learning_rate = lrate)
epochs     = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16
#==============================================================================
###############################################################################
# Batch and shuffle the data ==================================================
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset  = (tf.data.Dataset.from_tensor_slices(test_images)
                 .shuffle(test_size).batch(batch_size))

###############################################################################
#def sample_z(args):
#    z_mu, z_sigma = args
#    eps = tf.random_normal

filterszconv = [32, 64, 64, 64]
numneurons   = [32, 256, 64]
encoder = tf.keras.Sequential(
    [
     tf.keras.layers.InputLayer(input_shape=input_shape, name = 'encoder_input'),
     tf.keras.layers.Conv2D(filters=filterszconv[0], kernel_size=2, 
                            strides=(1, 1), padding='same', activation='relu'),
     tf.keras.layers.Conv2D(filters=filterszconv[1], kernel_size=2, 
                            strides=(2, 2), padding='same', activation='relu'),
     tf.keras.layers.Conv2D(filters=filterszconv[2], kernel_size=2, 
                           strides=(1, 1), padding='same', activation='relu'),
     tf.keras.layers.Conv2D(filters=filterszconv[3], kernel_size=2, 
                           strides=(1, 1), padding='same', activation='relu'),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(numneurons[0], activation='relu',
                               kernel_initializer='glorot_uniform', 
                               bias_initializer='glorot_uniform'),
#     tf.keras.layers.Dense(numneurons[1], activation='relu',
#                               kernel_initializer='glorot_uniform', 
#                               bias_initializer='glorot_uniform'),
#     tf.keras.layers.Dense(numneurons[2], activation='relu',
#                               kernel_initializer='glorot_uniform', 
#                               bias_initializer='glorot_uniform'),
     tf.keras.layers.Dense(latent_dim, name = 'z_mu'),
     tf.keras.layers.Dense(latent_dim, name = 'z_sigma'),
    ]
    )
print(encoder.summary())
outputs = [layer.output for layer in encoder.layers]  # all layer outputs
cont = 0
for i in outputs:
    name = i.name
    if name[0:7] == 'flatten':
        j = cont
    print(name[0:7])
    cont += 1
layer   = outputs[j-1]
layer_shape = layer.shape[1:]
layerdense  = layer_shape[0] * layer_shape[1] * layer_shape[2]
print(layerdense, layer_shape)

