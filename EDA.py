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


def generate_datasets(dataset, num):
    dataset.rename(columns={'Unnamed: 0':'sample'}, inplace=True)
    dataset.set_index("sample", inplace=True)
    dataset_data = dataset.iloc[:,1:]
    dataset_labels = dataset_data[["COVAR_N_status", "COVAR_M"]].copy()
    dataset_data = dataset_data.drop(["COVAR_N_status", "COVAR_M"], axis=1)
    dataset_labels_onehot = pd.DataFrame(dataset_labels["COVAR_N_status"]*2+dataset_labels["COVAR_M"])
    label_map = {0:"No LNM and no DNM", 1:"No LNM but DNM", 2:"LNM but no DM", 3:"LNM and DM"}
    dataset_labels_onehot["label"] = dataset_labels_onehot[0].map(label_map)
    dataset_labels_onehot.rename(columns={'0':'one-hot'}, inplace=True)
    if num == 4:
        dataset_data = dataset_data.apply(np.exp)
    dataset_scaled = pd.DataFrame(data=(MinMaxScaler().fit_transform(dataset_data)), columns=dataset_data.columns, index=dataset.index)
    dataset_trans = MinMaxScaler().fit_transform(dataset_data.apply(np.log))
    dataset_trans = pd.DataFrame(data=dataset_trans, columns=dataset_data.columns, index=dataset.index)
    return dataset_labels, dataset_labels_onehot, dataset_data, dataset_scaled, dataset_trans


# In[4]:


df2 = pd.read_csv("data/READCopyProtein50.csv")
with open('data/df2_new.pickle', 'wb') as f:
    pickle.dump(generate_datasets(df2, 2), f)


# In[12]:


df3 = pd.read_csv("data/COADCopyProtein50.csv")
with open('data/df3_new.pickle', 'wb') as f:
    pickle.dump(generate_datasets(df3, 3), f)


# In[13]:


df4 = pd.read_csv("data/GSE62254CopyConvertedProtein.csv")
with open('data/df4_new.pickle', 'wb') as f:
    pickle.dump(generate_datasets(df4, 4), f)


# In[4]:


with open('data/df2_new.pickle', 'rb') as f:
     df2_labels, df2_labels_onehot, df2_data, df2_scaled, df2_trans = pickle.load(f)
with open('data/df3_new.pickle', 'rb') as f:
     df3_labels, df3_labels_onehot, df3_data, df3_scaled, df3_trans = pickle.load(f)
with open('data/df4_new.pickle', 'rb') as f:
     df4_labels, df4_labels_onehot, df4_data, df4_scaled, df4_trans = pickle.load(f)
data_dict ={}
data_dict["COAD"] = {"scaled":df3_scaled, "trans":df3_trans}
data_dict["GSE"] = {"scaled":df4_scaled, "trans":df4_trans}


# In[5]:


def get_skew(df, df_trans, name):
    skew = sc.stats.skew(df, axis=0)
    median = np.median(df, axis=0)
    skew_trans = sc.stats.skew(df_trans, axis=0)
    median_trans = np.median(df_trans, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(name+' before and after transformation')
    ax1.scatter(skew, median)
    ax2.scatter(skew_trans, median_trans)
    fig.supylabel("median")
    fig.supxlabel("skew")
    plt.savefig("images/"+name+'_skew.png', bbox_inches='tight')
    plt.show()

get_skew(df2_data, df2_trans, "READ")
get_skew(df3_data, df3_trans, "COAD")
get_skew(df4_data, df4_trans, "GSE62254")


# In[19]:


def get_umap(df_scaled, df_trans, data_labels_onehot, name):
    reducer = umap.UMAP()
    reducer_trans = umap.UMAP()
    
    embedding = pd.DataFrame(reducer.fit_transform(df_scaled), columns =['embedding 1', 'embedding 2'], index=data_labels_onehot.index)
    embedding = pd.concat((embedding, data_labels_onehot), axis = 1)
    embedding_trans = pd.DataFrame(reducer_trans.fit_transform(df_trans), columns =['embedding 1', 'embedding 2'], index=data_labels_onehot.index)
    embedding_trans = pd.concat((embedding_trans, data_labels_onehot), axis = 1)
    fig, (ax1, ax2) = plt.subplots(1, 2) 
    fig.suptitle(name+' UMAP before and after transformation')
    sns.scatterplot(x='embedding 1', y='embedding 2', data=embedding, ax=ax1, hue="label", legend = False)
    sns.scatterplot(x='embedding 1', y='embedding 2', data=embedding_trans, ax=ax2, hue="label")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax1.set(xlabel=None)
    ax1.set(ylabel=None)
    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    fig.supylabel("Embedding 1")
    fig.supxlabel("Embedding 2")
    plt.savefig("images/"+name+'_umap.png', bbox_inches='tight')
    plt.show()
    
get_umap(df2_data, df2_trans, df2_labels_onehot, "READ")
get_umap(df3_data, df3_trans, df3_labels_onehot, "COAD")
get_umap(df4_data, df4_trans, df4_labels_onehot, "GSE62254")


# In[21]:


def get_pca(df_scaled, df_trans, data_labels_onehot, name):
    pca = PCA(n_components=2)
    pca_trans = PCA(n_components=2)
    embedding = pd.DataFrame(pca.fit_transform(df_scaled), columns = ['principal component 1', 'principal component 2'], index=data_labels_onehot.index)
    embedding_trans = pd.DataFrame(pca_trans.fit_transform(df_trans), columns = ['principal component 1', 'principal component 2'], index=data_labels_onehot.index)
    embedding = pd.concat((embedding, data_labels_onehot), axis = 1)
    embedding_trans = pd.concat((embedding_trans, data_labels_onehot), axis = 1)
    
    pca_var = PCA(n_components=20)
    pca_var_20 = sum(pca_var.fit(df_scaled).explained_variance_ratio_)
    pca_trans_var = PCA(n_components=20)
    pca_trans_var_20 = sum(pca_trans_var.fit(df_trans).explained_variance_ratio_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2) 
    fig.suptitle(name+' PCA before and after transformation')
    sns.scatterplot(x='principal component 1', y='principal component 2', data=embedding, ax=ax1, hue="label", legend = False)
    sns.scatterplot(x='principal component 1', y='principal component 2', data=embedding_trans, ax=ax2, hue="label")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    #ax1.set_title("Explained Variance: " + "{0:.2f}".format(sum(pca.explained_variance_ratio_[0:20])))
    #ax2.set_title("Explained Variance: " + "{0:.2f}".format(sum(pca_trans.explained_variance_ratio_[0:20])))
    #ax1.set_title("Explained Variance: " + "{0:.2f}".format(pca_var_20))
    #ax2.set_title("Explained Variance: " + "{0:.2f}".format(pca_trans_var_20))
    ax1.set(xlabel=None)
    ax1.set(ylabel=None)
    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    
    fig.supylabel("Principal component 1")
    fig.supxlabel("Principal component 2")
    plt.savefig("images/"+name+'_pca.png', bbox_inches='tight')
    plt.show()
    
get_pca(df2_data, df2_trans, df2_labels_onehot, "READ")
get_pca(df3_data, df3_trans, df3_labels_onehot, "COAD")
get_pca(df4_data, df4_trans, df4_labels_onehot, "GSE62254")


# In[5]:


beta = K.variable(0)
epsilon_std = 1.0
latent_dim = 1024


# In[6]:


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


# In[7]:


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


# In[8]:


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


# In[9]:


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


# In[10]:


mydict = lambda: defaultdict(mydict)
loss_dict = mydict()
names = ["GSE"]
batch_sizes = [8, 16]
learning_rates = [0.0005,0.001,0.0015,0.002]
for name in names:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for run_num in range(5):
                loss = train_model(name=name, 
                  latent_dim=1024, 
                  batch_size=batch_size, 
                  epochs=50, 
                  learning_rate=learning_rate, 
                  kappa=1, 
                  beta = K.variable(0),
                  run_num=run_num)
                loss_dict[name][batch_size][learning_rate][run_num] = loss


# In[ ]:


2+2


# In[ ]:




