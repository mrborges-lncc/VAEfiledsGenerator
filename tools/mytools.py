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
#import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.colors
import sys
import random
import math
import scipy.stats as stats
import pyvista as pv
import vtk
import matplotlib.ticker as ticker
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
#==============================================================================
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.family':'Times'})

###############################################################################
###############################################################################
def GPU_controler(control,memory_control):
    if control:
        print(tf.test.is_built_with_cuda())
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        #tf.debugging.set_log_device_placement(True)
        if memory_control:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
    return len(tf.config.experimental.list_physical_devices('GPU'))
###############################################################################

###############################################################################
###############################################################################
def save_original_fields(data,inputshape,home,nf):
    nx    = inputshape[0]
    ny    = inputshape[1]
    nz    = inputshape[2]
    # loop ====================================================================
    for i in range(0,nf): 
        img = data[0,:,:,:].reshape(nx * ny * nz)
        fname = home + str(i) + '.dat'
        print(fname)
        np.savetxt(fname, img, fmt='%.8e', delimiter=' ', newline='\n', 
                   header='', footer='', comments='# ', encoding=None)
###############################################################################

###############################################################################
###############################################################################
def plot_losses(history, namefig):
    '''Plot the losses of the training process'''
    lista_metric = list() 
    for i in history.history.keys():
        lista_metric.append(i)
    #total_loss   = lista_metric[0]
    reconstruction_loss  = lista_metric[1]
    kl_loss      = lista_metric[2]
    #=========================================================================
    fig = plt.figure(constrained_layout=True, figsize=(30,25))
    fig, axs = plt.subplots(nrows=1,ncols=2, constrained_layout=True)
    #=========================================================================
    ya = np.min(history.history[reconstruction_loss]) * .95
    yb = np.max(history.history[reconstruction_loss]) * 1.05
    dy = (yb-ya)/4
    axy = np.arange(ya,yb+1e-5,dy)
    axx = np.arange(0, len(history.history['reconstruction_loss'])+1,
                    len(history.history['reconstruction_loss'])/4)
    axs[0].plot(history.history[reconstruction_loss],'tab:blue',linewidth=3)
    axs[0].set_title(r'Reconstruction loss',fontsize=14)
#    axs[0].legend([], loc='upper right')
    axs[0].set_ylim((ya,yb))
#    axs[0].set_yticks(axy)
    axs[0].set_xticks(axx)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12)
    axs[0].set_yscale('log')
    axs[0].set_xlabel(r'\textbf{Epoch}', fontsize=14, weight='bold', color='k')
    axs[0].set_ylabel(r'$\mathsf{MSE}$', fontsize=14, weight='bold', color='k')
    axs[0].set_box_aspect(1)
    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1e'))
#    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1f'))
    #=========================================================================
    ya = np.min(history.history[kl_loss]) * .95
    yb = np.max(history.history[kl_loss]) * 1.05
    dy = (yb-ya)/4
    axy = np.arange(ya,yb+1e-5,dy)
    axs[1].plot(history.history[kl_loss],'tab:orange',linewidth=3)
    axs[1].set_title(r'KL-divergence loss',fontsize=14)
#    axs[1].legend([], loc='upper right')
    axs[1].set_ylim((ya,yb))
    axs[1].set_yticks(axy)
    axs[1].set_xticks(axx)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12)
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'\textbf{Epoch}', fontsize=14, weight='bold', color='k')
    axs[1].set_ylabel(r'$\mathcal{D}_{\mathsf{KL}}$', fontsize=14, 
                      weight='bold', color='k')
    axs[1].set_box_aspect(1)
    axs[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1e'))
    name = namefig + '_losses.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
###############################################################################

###############################################################################
###############################################################################
class net_info:
    '''Class to store the information of the permeability dataset'''
    def __init__(self, conv_filters, conv_strides, conv_kernels, conv_activat,
                 conv_padding, dens_neurons, dens_activat):
        self.conv_filters = conv_filters
        self.conv_strides = conv_strides
        self.conv_kernels = conv_kernels
        self.conv_activat = conv_activat
        self.conv_padding = conv_padding
        self.dens_neurons = dens_neurons
        self.dens_activat = dens_activat
###############################################################################

###############################################################################
###############################################################################
def build_encoder3D(net, input_shape, latent_dim):
    '''Build the encoder network for 3D data'''
    print(net.conv_filters,input_shape)
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv3D(filters = net.conv_filters[0], 
                      kernel_size = net.conv_kernels[0],
                      strides = net.conv_strides[0], 
                      activation = net.conv_activat[0],
                      padding = net.conv_padding[0])(encoder_inputs)

    for i in range(1,len(net.conv_filters)):
      x = layers.Conv3D(filters = net.conv_filters[i], 
                        kernel_size = net.conv_kernels[i],
                        strides = net.conv_strides[i], 
                        activation = net.conv_activat[i],
                        padding = net.conv_padding[i])(x)
    x = layers.Flatten()(x)
    for i in range(len(net.dens_neurons)):
      x = layers.Dense(net.dens_neurons[i], activation= net.dens_activat[i])(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name='z')([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder
###############################################################################

###############################################################################
###############################################################################
def build_decoder3D(net, input_shape, latent_dim, ndens, layer_shape, nconv):
    '''Build the decoder network for 3D data'''
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(net.dens_neurons[ndens],
                     activation = net.dens_activat[ndens])(latent_inputs)
    for i in range(ndens-1,-1,-1):
      x = layers.Dense(net.dens_neurons[i], 
                       activation = net.dens_activat[i])(x)

    x = layers.Dense(layer_shape[0] * layer_shape[1] * layer_shape[2] *
                     layer_shape[3], activation = "relu")(x)
    x = layers.Reshape(layer_shape)(x)

    for i in range(nconv,-1,-1):
      x = layers.Conv3DTranspose(filters = net.conv_filters[i], 
                                 kernel_size = net.conv_kernels[i],
                                 strides = net.conv_strides[i], 
                                 activation = net.conv_activat[i],
                                 padding = net.conv_padding[i])(x)
    decoder_outputs = layers.Conv3DTranspose(filters = input_shape[-1],
                                             kernel_size = net.conv_kernels[0],
                                             activation="linear", 
                                             padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder
###############################################################################

###############################################################################
###############################################################################
def build_encoder2D(net, input_shape, latent_dim):
    '''Build the encoder network for 2D data'''
    encoder_inputs = keras.Input(shape=input_shape)
    if len(net.conv_filters) > 0:
        x = layers.Conv2D(filters     = net.conv_filters[0], 
                          kernel_size = net.conv_kernels[0],
                          strides     = net.conv_strides[0], 
                          activation  = net.conv_activat[0], 
                          padding     = net.conv_padding[0])(encoder_inputs)
        for i in range(1,len(net.conv_filters)):
            x = layers.Conv2D(filters     = net.conv_filters[i], 
                              kernel_size = net.conv_kernels[i],
                              strides     = net.conv_strides[i], 
                              activation  = net.conv_activat[i],
                              padding     = net.conv_padding[i])(x)
        x = layers.Flatten()(x)
    else:
        x = layers.Flatten()(encoder_inputs)
    #==========================================================================
    for i in range(len(net.dens_neurons)):
        x = layers.Dense(net.dens_neurons[i], activation= net.dens_activat[i])(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name='z')([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder
###############################################################################

###############################################################################
###############################################################################
def build_decoder2D(net, input_shape, latent_dim, ndens, layer_shape, nconv):
    '''Build the decoder network for 2D data'''
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(net.dens_neurons[ndens],
                     activation = net.dens_activat[ndens])(latent_inputs)
    for i in range(ndens-1,-1,-1):
      x = layers.Dense(net.dens_neurons[i], activation= net.dens_activat[i])(x)

    x = layers.Dense(layer_shape[0] * layer_shape[1] * layer_shape[2], 
                     activation="relu")(x)
    #==========================================================================
    if len(net.conv_filters) > 0:
        x = layers.Reshape(layer_shape)(x)
        for i in range(nconv,-1,-1):
          x = layers.Conv2DTranspose(filters = net.conv_filters[i], 
                                     kernel_size = net.conv_kernels[i],
                                     strides = net.conv_strides[i], 
                                     activation = net.conv_activat[i],
                                     padding = net.conv_padding[i])(x)
        decoder_outputs = layers.Conv2DTranspose(filters = input_shape[-1],
                                                 kernel_size = net.conv_kernels[0],
                                                 activation="linear", 
                                                 padding="same")(x)
    else:
        decoder_outputs = layers.Reshape(layer_shape)(x)
    #==========================================================================
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder
###############################################################################

###############################################################################
###############################################################################
def fieldplot3D(f,Lx,Ly,Lz,nx,ny,nz,name):
    '''Plot the 3D field'''
    field = f[:,:,0:nz]
    field = np.reshape(field,(nx*ny*nz))
    dx  = Lx/nx
    dy  = Ly/ny
    dz  = Lz/nz
    # mesh ####################################################################
    #values = np.linspace(0, 10, nx * ny * nz).reshape((nx, ny, nz))
    values = field.reshape((nx, ny, nz))
    values.shape
    # Create the spatial reference ############################################
    grid = pv.UniformGrid()
    # Initialize from a vtk.vtkImageData object ###############################
    vtkgrid = vtk.vtkImageData()
    grid = pv.UniformGrid(vtkgrid)
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    # the CELL data
    grid.dimensions = np.array(values.shape) + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (dx, dy, dz)  # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid['data'] = values.flatten(order="F")  # Flatten the array!
    # Now plot the grid!
    boring_cmap = plt.cm.get_cmap("viridis", 10)
    boring_cmap = 'jet'
    grid.plot(off_screen=False, interactive=False, eye_dome_lighting=False,
              return_viewer=False, show_edges=True, cmap = boring_cmap, 
              cpos=[-2, 5, 3], show_bounds=True, show_axes=True)
#    threshed = grid.threshold_percent([0.15, 0.50], invert=True)
#    threshed.plot(show_grid=True, cpos=[-2, 5, 3])
###############################################################################


###############################################################################
###############################################################################
def load_model_weights(dataname, ld):
    '''Load the weights of the encoder and decoder networks'''
    dataname= dataname + '_' + str(ld)
    outname = 'model/encoder_model_' + dataname + '.h5'
    print('Loading encoder model....: %s' % (outname))
    encoder = tf.keras.models.load_model(outname, 
                                         custom_objects={'Sampling':Sampling})
    outname = 'model/decoder_model_' + dataname + '.h5'
    print('Loading decoder model....: %s' % (outname))
    decoder = tf.keras.models.load_model(outname)
    outname = 'model/encoder_' + dataname + '.weights.h5'
    encoder.load_weights(outname)
    outname = 'model/decoder_' + dataname + '.weights.h5'
    decoder.load_weights(outname)
    return encoder, decoder
###############################################################################

###############################################################################
###############################################################################
def save_model_weights(vae, dataname, ld):
    '''Save the weights of the encoder and decoder networks'''
    dataname = dataname + '_' + str(ld)
    outname = 'model/encoder_model_' + dataname + '.h5'
    print('Saving the encoder model....: %s' % (outname))
    vae.encoder.save(outname)
    outname = 'model/decoder_model_' + dataname + '.h5'
    print('Saving the decoder model....: %s' % (outname))
    vae.decoder.save(outname)
    outname = 'model/encoder_' + dataname + '.weights.h5'
    vae.encoder.save_weights(outname)
    outname = 'model/decoder_' + dataname + '.weights.h5'
    vae.decoder.save_weights(outname)
###############################################################################

###############################################################################
###############################################################################
class perm_info:
    '''Class to store the information of the permeability dataset'''
    def __init__(self, namein, porous, input_shape, data_size, porosity,
                 Lx, Ly, Lz, nx, ny, nz):
        self.namein = namein
        self.porous = porous
        self.input_shape = input_shape
        self.data_size = data_size
        self.porosity = porosity
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nx = nx
        self.ny = ny
        self.nz = nz
###############################################################################

###############################################################################
###############################################################################
def Gfunction(x, porosity):
    '''G-function to generate porous media'''
    g = stats.norm.ppf(porosity)
    if x <= g:
        return 1.0
    else:
        return 0.0
###############################################################################

###############################################################################
###############################################################################
def plot_examples(images, namefig):
    '''Display a 5x7 plot of 35 images'''
    fig = plt.figure(figsize=(10,10))
    M = np.size(images,0)
    cor = 'jet'
    for n in range(1, 37):
        fig.add_subplot(6, 6, n)
        nr  = random.randint(0,M-1)
        img = images[nr,:,:,0]
        plt.imshow(img, cmap=cor, aspect='equal', interpolation='none', 
                   alpha = 1.0, origin='upper')
        plt.axis('off')
    name = namefig + '_data_examples.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
###############################################################################

###############################################################################
###############################################################################
def load_dataset(dataname,prep,infoperm,rvalid,rtest):
    '''Load the dataset, split it into training and test sets, and scale then'''
    ntrain = math.ceil((1. - (rvalid + rtest)) * infoperm.data_size)
    nvalid = math.ceil(rvalid * infoperm.data_size)
    ntest  = infoperm.data_size - (ntrain + nvalid)    
    if dataname == 'PERM':
        train, valid, test = load_PERM(infoperm.namein, infoperm.input_shape, 
                                       infoperm.data_size,ntrain,nvalid)
    else:
        if dataname == 'MNIST':
            (train, _), (test, _) = tf.keras.datasets.mnist.load_data()
        if dataname == 'FASHION_MNIST':
            (train, _), (test, _) = tf.keras.datasets.fashion_mnist.load_data()
        x = np.concatenate([train, test])
        lista  = list(range(infoperm.data_size))
        random.shuffle(lista)
        train = x[lista[0:ntrain],:,:]
        valid = x[lista[ntrain:ntrain+nvalid],:,:]
        test  = x[lista[ntrain+nvalid:],:,:]
#==============================================================================
    train = preprocess_images(train,dataname,prep,infoperm)
    valid = preprocess_images(valid,dataname,prep,infoperm)
    test  = preprocess_images(test,dataname,prep,infoperm)
    return train, valid, test
###############################################################################

###############################################################################
###############################################################################
def preprocess_images(images,dataname,prep,infoperm):
    '''Normalize and reshape the images'''
    poros    = infoperm.porous
    porosity = infoperm.porosity
    if dataname == 'PERM':
        nx  = images.shape[1]
        ny  = images.shape[2]
        nz  = images.shape[3]
        if prep:
            if poros:
#                GfunctionVec = np.vectorize(Gfunction)
#                images = GfunctionVec(images, porosity)
                g = stats.norm.ppf(porosity)
                images = np.where(images > g, 0.0, 1.0).astype('float32')
                return images.astype('float32')
            else:
                maxdata = np.max(images,axis=None)
                mindata = np.min(images,axis=None)
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
def load_PERM(namein,inputshape,datasize,ntrain,nvalid):
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
    random.shuffle(lista)
    train  = data[lista[0:ntrain],:,:,:]
    valid  = data[lista[ntrain:ntrain+nvalid],:,:,:]
    test   = data[lista[ntrain+nvalid:],:,:,:]
    return train, valid, test
###############################################################################

###############################################################################
###############################################################################
class Sampling(layers.Layer):
    '''Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.'''
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), mean = 0.0, stddev= 1.0,
                                   dtype = tf.float32)
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
 #           print(type(reconstruction),type(data))
 #           print(reconstruction.shape, data.shape)
 #           sys.exit()
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.MeanSquaredError(
                        reduction="none")(data, reconstruction), axis=(1, 2)
#                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            beta = 1.0
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - 
                              tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
###############################################################################

###############################################################################
###############################################################################
def plot_latent_space(vae, namefig, n=15, figsize=15):
    '''Display a n*n 2D manifold of digits'''
    # display an n*n 2D manifold of digits
    digit_size = 28
    A = -3.
    B = 3.
    C = -3
    D = 3.0
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
    name = namefig + '_predicted.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
###############################################################################

###############################################################################
###############################################################################
def plot_label_clusters(vae, data, labels, namefig):
    '''Display a 2D plot of the digit classes in the latent space'''
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    name = namefig + '_post_clusters.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
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
def plot_latent_hist(vae, images, latent_dim, namefig, nf):
    '''Display the distribution of latent variables'''
    z_mean, z_log_var, z = vae.encoder.predict(images)
    #==========================================================================
    statZ = np.zeros((latent_dim, 2))
    for i in range(latent_dim):
        statZ[i,0] = np.mean(z[:,i])
        statZ[i,1] = np.var(z[:,i])
    if nf > latent_dim:
        nf = latent_dim
    nf = np.minimum(nf,36)
    seq = np.arange(latent_dim, dtype = 'int')
    nx = np.int_(np.sqrt(nf, dtype = None))
    ny = nx
    nf = nx * ny
    nseq= np.random.choice(seq, nf, replace = False)
    fig= plt.figure(figsize=(10,10))
    a, b = -4.5, 4.5
    c, d = 0., 0.5
    x = np.arange(a, b, 0.01) 
    num_bins = 20
    n = 0
    for i in range(0,nx):
        for j in range(0,ny):
            zn   = z[:, nseq[n]]
            mu_z = np.mean(zn)
            var_z= np.var(zn)
            std_z= np.std(zn)
            print('Z_%d => mean: %5.3f \t\t sigma^2: %5.3f' % (nseq[n], 
                                                                 mu_z, var_z))
            #==================================================================
            fig.add_subplot(nx, ny, n+1)
            nb, bins, patches = plt.hist(zn, num_bins, density=1, alpha=1.0, 
                                         edgecolor='black')
#            x = np.arange(np.min(zn),np.max(zn),0.001) 
            # add a 'best fit' line
            y = ((1. / (np.sqrt(2. * np.pi) * std_z)) *
                 np.exp(-0.5 * (1. / std_z * (x - mu_z))**2))
            plt.plot(x, y, '-',linewidth=3,markersize=6, marker='',
                    markerfacecoloralt='tab:red', fillstyle='none')
            plt.xlim(a,b)
            plt.ylim(c,d)
            plt.xlabel(r'$z_{_{' + str(nseq[n]) + '}}$', fontsize=18,
                       weight='bold', color='k')
            plt.ylabel(r'$f(z_{_{' + str(nseq[n]) + '}})$', fontsize=18,
                       weight='bold', color='k')
            plt.tick_params(labelsize=16)
            n += 1
            # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    name = namefig + '_hist_latent.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
    return statZ
###############################################################################

###############################################################################
###############################################################################
def random_generator(model,latent_dim,inputshape,Z,home,nf):
    '''Display a n*n 2D manifold of digits'''
    zmean = Z[:,0]
    zvar  = Z[:,1]
    cov   = np.diag(zvar)
    nx    = inputshape[0]
    ny    = inputshape[1]
    nz    = inputshape[2]
    # loop ====================================================================
    for i in range(0,nf): 
        z = np.random.multivariate_normal(zmean, cov, 1).T
        z = z.reshape((1, latent_dim))
        X = model.predict(z)
        img = X[0].reshape(nx * ny * nz)
        fname = home + str(i) + '.dat'
        print(fname)
        np.savetxt(fname, img, fmt='%.8e', delimiter=' ', newline='\n', 
                   header='', footer='', comments='# ', encoding=None)
###############################################################################

###############################################################################
###############################################################################
def midpoints(x):
    '''Compute midpoints of a grid'''
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x
###############################################################################

###############################################################################
###############################################################################
def plot_3D(img, infoperm, namefig):
    '''Plot the 3D field'''
    minx = np.min(img)
    maxx = np.max(img)
    img = (img - minx) / (maxx - minx)
    m   = max(max(infoperm.Lx, infoperm.Ly), infoperm.Lz)
    # prepare some coordinates, and attach rgb values to each 
    x = np.linspace(0, infoperm.Lx, infoperm.nx+1) 
    y = np.linspace(0, infoperm.Ly, infoperm.ny+1) 
    z = np.linspace(0, infoperm.Lz, infoperm.nz+1)
    X, Y, Z = np.meshgrid(x, y, z)

    r, g, b = np.indices((infoperm.nx+1, infoperm.ny+1, infoperm.nz+1)) / 28.0
    rc = midpoints(X) 
    sphere = rc > -2
    # combine the color components 
    colors = np.zeros(sphere.shape + (3,)) 
    colors[..., 0] = img
    colors[..., 1] = img 
    colors[..., 2] = img
#    colors = matplotlib.colors.hsv_to_rgb(colors)
    
    # and plot everything 
    fig = plt.figure(constrained_layout=True) 
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((infoperm.Lx / m , infoperm.Ly / m, infoperm.Lz / m))
    ax.voxels(r, g, b, sphere, 
              facecolors=colors, #              edgecolors='k',  # brighter 
              linewidth=0.5) 
    ax.set_xlim(0,infoperm.Lx)
    ax.set_ylim(0,infoperm.Ly)
    ax.set_zlim(0,infoperm.Lz)
    ax.set_xlabel(r'$x$', fontsize=14, weight='bold', color='k')
    ax.set_ylabel(r'$y$', fontsize=14, weight='bold', color='k')
    ax.set_zlabel(r'$z$', fontsize=14, weight='bold', color='k')
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="z", labelsize=12)
    zz = np.arange(0,infoperm.Lz*1.01,infoperm.Lz/2)
    ax.set_zticks(zz[1:])
#    plt.axis('off')
    name = namefig + '_3D.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
###############################################################################

###############################################################################
###############################################################################
def fieldgenerator(model,latent_dim,inputshape,Z,namefig,infoperm,nf):
    '''Plot the 2D or 3D fields generated by the decoder'''
    zmean = Z[:,0]
    zvar  = Z[:,1]
    cov   = np.diag(zvar)
    nz    = inputshape[2]
    #==========================================================================
    if nz > 1:
        # prepare some coordinates, and attach rgb values to each 
        x = np.linspace(0, infoperm.Lx, infoperm.nx+1) 
        y = np.linspace(0, infoperm.Ly, infoperm.ny+1) 
        z = np.linspace(0, infoperm.Lz, infoperm.nz+1)
        X, Y, Z = np.meshgrid(x, y, z)
    
        r, g, b = np.indices((infoperm.nx+1, infoperm.ny+1, infoperm.nz+1)) / 28.0
        rc = midpoints(X) 
        sphere = rc > -2
        # combine the color components 
        colors = np.zeros(sphere.shape + (3,)) 

    nx = np.int_(np.sqrt(nf, dtype = None))
    ny = nx
    nf = nx * ny
    fig= plt.figure(figsize=(10,10))
    n  = 0
    for i in range(0,nx):
        for j in range(0,ny):
            n += 1
            z = np.random.multivariate_normal(zmean, cov, 1).T
            z = z.reshape((1, latent_dim))
            x_decoded  = model.decoder.predict(z)
            if nz == 1:
                fig.add_subplot(nx, ny, n)
                img = x_decoded[0].reshape(inputshape[0],inputshape[1])
                plt.imshow(img, cmap="jet", aspect='equal', 
                           interpolation='none', alpha = 1.0, origin='upper')
                plt.axis('off')
            else:
                img = x_decoded[0].reshape(inputshape[0],inputshape[1],
                                           inputshape[2])
                minx = np.min(img)
                maxx = np.max(img)
                img = (img - minx) / (maxx - minx)
                m   = max(max(infoperm.Lx, infoperm.Ly), infoperm.Lz)
                colors[..., 0] = img 
                colors[..., 1] = img 
                colors[..., 2] = img
                ax = fig.add_subplot(nx, ny, n, projection='3d')
                ax.set_box_aspect((infoperm.Lx / m , infoperm.Ly / m,
                                   infoperm.Lz / m)) 
                ax.voxels(r, g, b, sphere, facecolors=colors, 
                          #              edgecolors='k',  # brighter 
                          linewidth=0.0) 
                plt.axis('off')
    name = namefig + '_predicted_examples.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
    return zmean, zvar, z
###############################################################################

###############################################################################
###############################################################################
def comparison(vae, images, latent_dim, inputshape, namefig, infoperm, nsample):
    '''Display a 2D plot of the digit classes in the latent space'''
    z_mean, z_log_var, z = vae.encoder.predict(images)
    n   = random.randint(0,np.size(images,axis=0))
    zz  = z[n,:]
    zz  = zz.reshape((1, latent_dim))
    prd = vae.decoder.predict(zz)
    #==========================================================================
    fig = plt.figure(figsize=(10,5))
    if infoperm.nz == 1:
        img = images[n,:,:,:]
        img = img.reshape(inputshape[0],inputshape[1])
        fig.add_subplot(1,2,1)
        plt.imshow(img, cmap="jet", aspect='equal', interpolation='none',
                   alpha = 1.0, origin='upper')
        prd = prd.reshape(inputshape[0],inputshape[1])
        #    prd = np.where(prd > .5, 1.0, 0.0).astype('float32')
        fig.add_subplot(1,2,2)
        plt.imshow(prd, cmap="jet", aspect='equal', interpolation='none',
                   alpha = 1.0, origin='upper')
    else:
        img = images[n,:,:,:]
        x = np.linspace(0, infoperm.Lx, infoperm.nx+1) 
        y = np.linspace(0, infoperm.Ly, infoperm.ny+1) 
        z = np.linspace(0, infoperm.Lz, infoperm.nz+1)
        X, Y, Z = np.meshgrid(x, y, z)
        r, g, b = np.indices((infoperm.nx+1, infoperm.ny+1, infoperm.nz+1)) / 28.0
        rc = midpoints(X) 
        sphere = rc > -2
        colors = np.zeros(sphere.shape + (3,)) 
        minx = np.min(img)
        maxx = np.max(img)
        img  = (img - minx) / (maxx - minx)
        m    = max(max(infoperm.Lx, infoperm.Ly), infoperm.Lz)
        colors[..., 0] = img 
        colors[..., 1] = img 
        colors[..., 2] = img
        #======================================================================
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.set_box_aspect((infoperm.Lx / m , infoperm.Ly / m,
                           infoperm.Lz / m)) 
        ax.voxels(r, g, b, sphere, facecolors=colors, linewidth=0.0) 
        plt.axis('off')
        #======================================================================
        #======================================================================
        prd = prd[0].reshape(inputshape[0],inputshape[1],inputshape[2])
        prd = (prd - minx) / (maxx - minx)
        colors[..., 0] = prd 
        colors[..., 1] = prd 
        colors[..., 2] = prd
        ax = fig.add_subplot(1,2,2, projection='3d')
        ax.set_box_aspect((infoperm.Lx / m , infoperm.Ly / m,
                           infoperm.Lz / m)) 
        ax.voxels(r, g, b, sphere, facecolors=colors, linewidth=0.0) 
        plt.axis('off')
    name = namefig + '_data_x_predicted.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
    #==========================================================================
    ndata = nsample#np.size(images, axis = 0)
    rel_error = np.zeros((ndata,1))
    for i in range(ndata):
        zz  = z[i,:]
        zz  = zz.reshape((1, latent_dim))
        prd = vae.decoder.predict(zz)
        prd = prd.reshape(inputshape[0],inputshape[1])
        img = images[i,:,:,:]
        img = img.reshape(inputshape[0],inputshape[1])
        rel_error[i] = np.linalg.norm(img - prd) / np.linalg.norm(img)
    #==========================================================================
    fig= plt.figure(figsize=(10,10))
    a, b = np.min(rel_error), np.max(rel_error)
    c, d = 0., 0.5
    x = np.arange(a, b, 0.01) 
    num_bins = 20
    mu = np.mean(rel_error)
    desv = np.std(rel_error)
    print('Relative Mean Squared Error => mean: %5.3f \t\t sigma: %5.3f' % (mu, desv))
    #==========================================================================
    nb, bins, patches = plt.hist(rel_error, num_bins, density=1, alpha=1.0, 
                                         edgecolor='black')
    d =  np.max(nb) * 1.1
    y = ((1. / (np.sqrt(2. * np.pi) * desv)) * 
         np.exp(-0.5 * (1. / desv * (x - mu))**2)) 
    plt.plot(x, y, '-',linewidth=3,markersize=6, marker='',
             markerfacecoloralt='tab:red', fillstyle='none')
    plt.xlim(a,b)
    plt.ylim(c,d)
    plt.xlabel(r'$RMSE$', fontsize=22,
               weight='bold', color='k')
    plt.ylabel(r'Density', fontsize=22,
               weight='bold', color='k')
    plt.tick_params(labelsize=20)
    fig.tight_layout()
    name = namefig + '_hist_latent.png'
    plt.savefig(name, transparent=True, dpi=300, bbox_inches='tight')
    plt.show()
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
