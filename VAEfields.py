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
batch_size = 64
test_size  = np.size(test_images,0)
input_shape= train_images.shape[1:]
lrate      = 1e-4
optimizer  = tf.keras.optimizers.Adam(learning_rate = lrate)
epochs     = 50
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
# Model =======================================================================
model = CVAE(latent_dim, input_shape)

###############################################################################
# Pick a sample of the test set for generating output images ==================
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
#==============================================================================
generate_and_save_images(model, 0, test_sample)
###########################################################
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)
    generate_numbers(model, epoch, test_sample)
###########################################################
plt.imshow(display_image(epoch))
plt.axis('off')  # Display images
###########################################################

###########################################################
anim_file = 'figuras/cvae.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('figuras/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython
if IPython.version_info >= (6, 2, 0, ''):
    display.Image(filename=anim_file)
###########################################################
plot_latent_images(model, 20)
###########################################################


