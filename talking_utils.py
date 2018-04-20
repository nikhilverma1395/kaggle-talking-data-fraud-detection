from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cross_validation import StratifiedShuffleSplit

import pandas as pd
import gc
import numpy as np
import types
import tempfile
import keras.models


from sklearn.preprocessing import LabelEncoder

def get_keras_layer_model(train_df, test, emb_n, dense_n, drop_out,conv_n,conv_strides,conv_kernel_size):
    emb_layers = []
    inp_layers = []
    for col in train_df.keys():
        inl = Input(shape=[1], name = col)
        inp_layers.append(inl)
        ml = np.max([train_df[col].max(),test[col].max()])+1
        ebl = Embedding(ml, emb_n)(inl)
        emb_layers.append(ebl)
    
    s_dout = SpatialDropout1D(drop_out)(concatenate(emb_layers))    
    
    flsd = Flatten()(s_dout)
    
    conv = Conv1D(conv_n[0], kernel_size=conv_kernel_size[0], strides=conv_strides[0], padding='same')(s_dout)
    flc = Flatten()(conv)
    
    concat = concatenate([(flsd), (flc)])
    x = Dropout(drop_out)(Dense(dense_n,activation='relu')(concat))
    x = Dropout(drop_out)(Dense(dense_n,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    
    model = Model(inputs=inp_layers, outputs=outp)

    return model
    
    
#Exponentially Decaying over len(train)/BATCH_SIZE    
def get_keras_model_compiled(len_train_data, batch_size,epochs, lr_s, lr_f, model):
 
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len_train_data / batch_size) * epochs
    lr_decay = exp_decay(lr_s, lr_f, steps)
    optimizer_adam = Adam(lr=0.001, decay=lr_decay)
    model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
    
    return model

#Categorical Features fitted via Label Encoder    
def get_data_keras_np(dataset,tr_cols):
    inp = {}
    for col in dataset.columns.values:
        if col in tr_cols:
            inp[col] = np.apply_along_axis(LabelEncoder().fit_transform, 0, np.array(dataset[col]))
        else:
            inp[col] = np.array(dataset[col])
    return inp
    
    
def get_evenly_split_train_val(ratio,y,train_np,splits=1):
    sss = StratifiedShuffleSplit(y, splits, test_size=ratio, random_state=56653)
    train_x = {}
    train_y = {}
    for train_index, val_index in sss:
        val_x = y[train_index]
        val_y = y[val_index]
        for key in train_np.keys():
            train_x[key] = train_np[key][train_index]
            train_y[key] = train_np[key][val_index]
            
    del val_index, train_index, sss
    valid = val_y.shape[0] + train_y['app'].shape[0] + val_x.shape[0] + train_x['app'].shape[0] - y.shape[0] - train_np['app'].shape[0]
   
    
    print('Created train-val sets!',valid)
            
    return train_x,train_y,val_x,val_y

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__
