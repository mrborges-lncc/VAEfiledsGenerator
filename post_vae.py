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
fig_print = True
###############################################################################
# Load data base ==============================================================
Lx  = 100.0
Ly  = 100.0
Lz  = 0.01
nx  = 50
ny  = 50
nz  = 1
num_channel = 1
if nz == 1:
    input_shape= (nx, ny, num_channel)
else:
    input_shape= (nx, ny, nz, num_channel)
#==============================================================================
data_size  = 60000
ratio_valid= 0.1
ratio_test = 0.1
home       = '/prj/prjmurad/mrborges/Dropbox/fieldsCNN/'
home       = '/prj/prjmurad/mrborges/fieldsCNN/'
home       = '/prj/prjmurad/mrborges/Dropbox/matricesKLE/'
#home       = '/media/mrborges/borges/fieldsCNN/'
namein     = home + 'mix3_100.00x100.00x1.00_50x50x1_60000.mat'
porous     = False
porosity   = 0.20
infoperm   = perm_info(namein, porous, input_shape, data_size, porosity, 
                       Lx, Ly, Lz, nx, ny, nz)
#==============================================================================
data_name  = ['MNIST', 'PERM', 'FASHION_MNIST']
dataname   = data_name[1]
name_ext   = '_cilamce64'
namefig    = './figuras/' + dataname + name_ext
preprocess = True # Normalize
train_images, valid_images, test_images, maxdata, mindata = \
 load_dataset(dataname,preprocess, infoperm, ratio_valid, ratio_test)
fname = 'model/min_max' + name_ext + '.dat'
#np.savetxt(fname, (mindata, maxdata), fmt='%.8e', delimiter=' ',
#           newline='\n', header='', footer='', comments='# ', encoding=None)
plot_examples(train_images, namefig, fig_print)
print("Original Data interval......... [%g,%g]" % (mindata,maxdata))
print("Post-processed Data interval... [%g,%g]" % (np.min(train_images),np.max(train_images)))
###############################################################################
ld = 64
hm   = 'model/'
ename  = hm + 'encoder_model_' + dataname + name_ext + '_128.h5'
dname  = hm + 'decoder_model_' + dataname + name_ext + '_128.h5'
ewname = hm + 'encoder_' + dataname + name_ext + '_128.weights.h5'
dwname = hm + 'decoder_' + dataname + name_ext + '_128.weights.h5'

dataname = 'tosin_PERM_'
name_ext = 'teste'
hm   = '/prj/prjmurad/mrborges/Dropbox/vae_files/'
ename  = hm + 'encoder_model_' + dataname + name_ext + '_64.h5'
dname  = hm + 'decoder_model_' + dataname + name_ext + '_64.h5'
ewname = hm + 'encoder_' + dataname + name_ext + '_64.weights.h5'
dwname = hm + 'decoder_' + dataname + name_ext + '_64.weights.h5'

encoder, decoder = load_model_weights(ename, dname, ewname, dwname)
print(encoder.summary())
print(decoder.summary())
filen = 'model/encoder_summary' + name_ext + '.txt'
with open(filen, 'w') as f:
    encoder.summary(print_fn=lambda x: f.write(x + '\n'))
filen = 'model/decoder_summary' + name_ext + '.txt'
with open(filen, 'w') as f:
    decoder.summary(print_fn=lambda x: f.write(x + '\n'))
#==============================================================================
###############################################################################
lrate      = 1.0e-4
optimizer  = tf.keras.optimizers.Adam(learning_rate = lrate)
decoder.compile()
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
Zparam = plot_latent_hist(vae, train_images, latent_dim, namefig, fig_print, 16)
#==============================================================================
###############################################################################
# Generator ===================================================================
zmu,zvar,z = fieldgenerator(vae, latent_dim, input_shape, Zparam, 
                            namefig, infoperm, fig_print, 36)
#==============================================================================
###############################################################################
# Comparison between data and predictions =====================================
nsample = 5000
for i in range(0,20):
    comparison(vae, test_images, latent_dim, input_shape, namefig,
               infoperm, nsample, fig_print, i)
    nsample = 0
#==============================================================================
#==============================================================================
###############################################################################
# Random generation ===========================================================
fname = './KLE/fields/field_' + dataname + '_'
N = 5
random_generator(decoder, latent_dim, input_shape, Zparam, fname, N, mindata,
                 maxdata)
#==============================================================================
###############################################################################
# Save original fields ========================================================
fname = './KLE/fields/field_orig_' + dataname + '_'
save_original_fields(train_images, input_shape, fname, N)

#==============================================================================
###############################################################################
enc_name = ename[0:-3] + '.keras'
vae.encoder.save(enc_name)
dec_name = dname[0:-3] + '.keras'
vae.decoder.save(dec_name)
#==============================================================================
###############################################################################
# Field generation ============================================================
dec = vae.decoder
z   = 