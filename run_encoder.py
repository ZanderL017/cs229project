#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sc
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.callbacks import Callback
import tensorflow.keras
import os
import pickle5 as pickle
from tensorflow.keras import regularizers
from collections import defaultdict


import pydot
import graphviz
from keras.utils.vis_utils import plot_model
from keras_tqdm import TQDMNotebookCallback
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import tensorflow.compat.v1.keras.backend as K
tf.compat.v1.disable_eager_execution()


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-notebook')


# In[3]:


with open('data/df2_new.pickle', 'rb') as f:
     df2_labels, df2_labels_onehot, df2_data, df2_scaled, df2_trans = pickle.load(f)
with open('data/df3_new.pickle', 'rb') as f:
     df3_labels, df3_labels_onehot, df3_data, df3_scaled, df3_trans = pickle.load(f)
with open('data/df4_new.pickle', 'rb') as f:
     df4_labels, df4_labels_onehot, df4_data, df4_scaled, df4_trans = pickle.load(f)
data_dict ={}
data_dict["COAD"] = {"scaled":df3_scaled, "trans":df3_trans}
data_dict["GSE"] = {"scaled":df4_scaled, "trans":df4_trans}


# In[4]:


beta = K.variable(0)
epsilon_std = 1.0
latent_dim = 1024


# In[5]:


def scheduler(epoch, lr):
    if epoch < 250:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
# Function for reparameterization trick to make model differentiable
def sampling(args):
    epsilon_std = 1.0
    import tensorflow as tf
    # Function with args required for Keras Lambda function
    z_mean, z_log_var = args

    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)
    
    # The latent vector is non-deterministic and differentiable
    # in respect to z_mean and z_log_var
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z


class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training

    """
    def __init__(self, var_layer, mean_layer, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        self.var_layer = var_layer
        self.mean_layer = mean_layer
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = x_input.shape[1] * metrics.binary_crossentropy(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + self.var_layer - K.square(self.mean_layer) - 
                                K.exp(self.var_layer), axis=-1)
        return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


# In[6]:


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


# In[7]:


class Tybalt():
    """
    Facilitates the training and output of tybalt model trained on TCGA RNAseq gene expression data
    """
    def __init__(self, original_dim, latent_dim,
                 batch_size, epochs, learning_rate, kappa, beta, train_data, test_data, mydata, name):
        self.original_dim = original_dim
        #self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kappa = kappa
        self.beta = beta
        self.train_data = train_data
        self.test_data = test_data
        self.data = mydata
        self.name = name

    def build_encoder_layer(self):
        # Input place holder for RNAseq data with specific input size
        self.rnaseq_input = Input(shape=(self.original_dim, ))

        # Input layer is compressed into a mean and log variance vector of size `latent_dim`
        # Each layer is initialized with glorot uniform weights and each step (dense connections, batch norm,
        # and relu activation) are funneled separately
        # Each vector of length `latent_dim` are connected to the rnaseq input tensor
        #hidden_dense_linear = Dense(self.hidden_dim, kernel_initializer='glorot_uniform')(self.rnaseq_input)
        #hidden_dense_batchnorm = BatchNormalization()(hidden_dense_linear)
        #hidden_encoded = Activation('relu')(hidden_dense_batchnorm)

        #z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(hidden_encoded)
        z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(self.rnaseq_input)
        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
        self.z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

        #z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(hidden_encoded)
        z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(self.rnaseq_input)
        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
        self.z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

        # return the encoded and randomly sampled z vector
        # Takes two keras layers as input to the custom sampling function layer with a `latent_dim` output
        self.z = Lambda(sampling, output_shape=(self.latent_dim, ))([self.z_mean_encoded, self.z_log_var_encoded])
    
    def build_decoder_layer(self):
        # The decoding layer is much simpler with a single layer glorot uniform initialized and sigmoid activation
        self.decoder_model = tf.keras.Sequential()
        #self.decoder_model.add(Dense(self.hidden_dim, activation='relu', input_dim=self.latent_dim))
        self.decoder_model.add(Dense(self.original_dim, kernel_initializer='glorot_uniform', activation='sigmoid'))
        self.rnaseq_reconstruct = self.decoder_model(self.z)
        
    def compile_vae(self):
        adam = optimizers.Adam(lr=self.learning_rate)
        vae_layer = CustomVariationalLayer(self.z_log_var_encoded,
                                           self.z_mean_encoded)([self.rnaseq_input, self.rnaseq_reconstruct])
        self.vae = Model(self.rnaseq_input, vae_layer)
        self.vae.compile(optimizer=adam, loss=None, loss_weights=[self.beta])
        self.stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=15,verbose=0,mode="auto",baseline=None,restore_best_weights=True,)
        
    def get_summary(self):
        self.vae.summary()
    
    def visualize_architecture(self, output_file):
        # Visualize the connections of the custom VAE model
        plot_model(self.vae, to_file=output_file)
        SVG(model_to_dot(self.vae).create(prog='dot', format='svg'))
        
    def train_vae(self):
        self.hist = self.vae.fit(np.array(self.train_data),
               shuffle=True,
               epochs=self.epochs,
               batch_size=self.batch_size,
               validation_data=(np.array(self.test_data), None),
               callbacks=[self.stopping, WarmUpCallback(self.beta, self.kappa)])
    
    def visualize_training(self, output_file):
        # Visualize training performance
        history_df = pd.DataFrame(self.hist.history).iloc[3:,:2]
        ax = history_df.plot()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('VAE Loss')
        ax.set_title(self.name + " Loss: " + "{0:.2f}".format(history_df['val_loss'].nsmallest(3).median()))
        fig = ax.get_figure()
        fig.savefig(output_file+"_{0:.2f}".format(history_df['val_loss'].nsmallest(3).median())+".png")
        return history_df['val_loss'].nsmallest(3).median()
        
    def compress(self, df):
        # Model to compress input
        self.encoder = Model(self.rnaseq_input, self.z_mean_encoded)
        
        # Encode rnaseq into the hidden/latent representation - and save output
        encoded_df = self.encoder.predict_on_batch(df)
        encoded_df = pd.DataFrame(encoded_df, columns=range(1, self.latent_dim + 1),
                                  index=self.data.index)
        return encoded_df
    
    def get_decoder_weights(self):
        # build a generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim, ))  # can generate from any sampled z vector
        _x_decoded_mean = self.decoder_model(decoder_input)
        self.decoder = Model(decoder_input, _x_decoded_mean)
        weights = []
        for layer in self.decoder.layers:
            weights.append(layer.get_weights())
        return(weights)
    
    def predict(self, df):
        return self.decoder.predict(np.array(df))
    
    def save_models(self, encoder_file, decoder_file):
        self.encoder.save(encoder_file)
        self.decoder.save(decoder_file)


# In[8]:


def train_model(name, latent_dim, batch_size, epochs, learning_rate, kappa, beta, run_num):
    data = data_dict[name]["trans"]
    test_set_percent = 0.3
    test_df = data.sample(frac=test_set_percent)
    train_df = data.drop(test_df.index)
    original_dim = data.shape[1]
    beta = K.variable(0)
    epsilon_std = 1.0
    model = Tybalt(original_dim=original_dim,
                 latent_dim=latent_dim,
                 batch_size=batch_size,
                 epochs=epochs,
                 learning_rate=learning_rate,
                 kappa=kappa,
                 beta=beta, 
                 test_data=test_df,
                 train_data=train_df, 
                 mydata=data,
                 name=name)
    model.build_encoder_layer()
    model.build_decoder_layer()
    model.compile_vae()
    model.train_vae()
    os.makedirs(os.path.join('images/parametersweep', name+'_onehidden', str(batch_size), str(learning_rate)), exist_ok= True) 
    model_training_file = os.path.join('images/parametersweep', name+'_onehidden', str(batch_size), str(learning_rate), str(run_num))
    loss = model.visualize_training(model_training_file)
    data_compressed = model.compress(data)
    os.makedirs(os.path.join('data/parametersweep', 'encoded_'+name+'_onehidden', str(batch_size), str(learning_rate)), exist_ok= True) 
    data_compressed_file = os.path.join('data/parametersweep', 'encoded_'+name+'_onehidden', str(batch_size), str(learning_rate), str(run_num)+".csv")
    data_compressed.to_csv(data_compressed_file)
    return loss


# In[9]:


loss = train_model(name="GSE", latent_dim=1024, 
                  batch_size=8, 
                  epochs=250, 
                  learning_rate=0.0005, 
                  kappa=1, 
                  beta = K.variable(0),
                  run_num=15)


# In[10]:


loss = train_model(name="COAD", latent_dim=1024, 
                  batch_size=4, 
                  epochs=250, 
                  learning_rate=0.0005, 
                  kappa=1, 
                  beta = K.variable(0),
                  run_num=15)


# In[11]:


2+2


# In[ ]:


# mydict = lambda: defaultdict(mydict)
# loss_dict = mydict()
# names = ["GSE"]
# batch_sizes = [16]
# learning_rates = [0.0005,0.001,0.0015,0.002]
# for name in names:
#     for batch_size in batch_sizes:
#         for learning_rate in learning_rates:
#             for run_num in range(5):
#                 loss = train_model(name=name, 
#                   latent_dim=1024, 
#                   batch_size=batch_size, 
#                   epochs=50, 
#                   learning_rate=learning_rate, 
#                   kappa=1, 
#                   beta = K.variable(0),
#                   run_num=run_num)
#                 loss_dict[name][batch_size][learning_rate][run_num] = loss

