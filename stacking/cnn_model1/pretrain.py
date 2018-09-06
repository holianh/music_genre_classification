import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Dropout, GlobalMaxPooling2D, Reshape, Conv2D, Flatten,Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.metrics import top_k_categorical_accuracy
from sklearn.model_selection import train_test_split
from keras import *
from keras.models import *
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from clr_callback import CyclicLR
from random import randint
from sklearn.utils import class_weight
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, GlobalMaxPooling2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score

NUMBER_OF_FOLD = 5
NUMBER_OF_CLASSES = 10
BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
EPOCH = 150
folds = [2,3,4]
model_sets = ['densenet201','inception_resnet_v2','inception_v3', 'xception']

# ftype = 'mfcc'
ftype = 'melspectrogram'

def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(to_categorical(row['class_id'], num_classes=NUMBER_OF_CLASSES))
    return np.array(y_true)

def accuracy(preds, truths):
    p_idx = np.argmax(preds, axis=1)
    y_idx = np.argmax(truths, axis=1)
    return accuracy_score(y_idx, p_idx)

def Stacking_Model():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(2, 1), activation='relu', input_shape=(len(model_sets), NUMBER_OF_CLASSES, 1)))
    model.add(Conv2D(16, (2, 1), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (2, 1), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    model.summary()
    return model

if __name__ == '__main__':
    pseudo_df = pd.read_csv('../../datasets/pseudo_dataset.csv')
    pseudo_size = pseudo_df.shape[0]
    for fold in folds:
        print('***************  Fold %d  ***************'%(fold))
        train_df = pd.read_csv('../../data/train_set_fold%d_lv2.csv'%fold)
        train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
        valid_df = pd.read_csv('../../data/valid_set_fold%d_lv2.csv'%fold)
        train_size = train_df.shape[0]
        valid_size = valid_df.shape[0]

        y_valid = get_y_true(valid_df)
        y_train = get_y_true(train_df)
        
        x_train = np.array([], dtype=np.float64).reshape(train_size,0)
        x_valid = np.array([], dtype=np.float64).reshape(valid_size,0)
        for mset in model_sets:
            p_train_tmp = np.load('../../%s/data/ptrain_fold%d_%s.npy'%(mset,fold,ftype))
            p_pseudo_tmp = np.load('../../%s/data/ppseudo_fold%d_%s.npy'%(mset,fold,ftype))
            p_train_tmp = np.vstack((p_train_tmp,p_pseudo_tmp))
            p_valid_tmp = np.load('../../%s/data/pvalid_fold%d_%s.npy'%(mset,fold,ftype))

            print('Model: %20s  train_acc: %.3f  valid_acc: %.3f'%(mset, accuracy(p_train_tmp,y_train), accuracy(p_valid_tmp,y_valid)))

            x_train = np.hstack((x_train,p_train_tmp))
            x_valid = np.hstack((x_valid,p_valid_tmp))

        x_train = np.reshape(x_train, (train_size, len(model_sets), NUMBER_OF_CLASSES, 1))
        x_valid = np.reshape(x_valid, (valid_size, len(model_sets), NUMBER_OF_CLASSES, 1))

        print('SHAPE OF DATASETS:')
        print('X VALID:',x_valid.shape)
        print('Y VALID:',y_valid.shape)
        print('X TRAIN:',x_train.shape)
        print('Y TRAIN:',y_train.shape)

        WEIGHTS_BEST = 'weights/pretrained_weights_fold%d_%s.hdf5'%(fold,ftype)
        early_stoping = EarlyStopping(monitor='val_acc', patience=15, verbose=1)
        save_checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 5, min_lr = 1e-9, verbose=1)
        clr = CyclicLR(base_lr=1e-9, max_lr=8e-5, step_size=2000., mode='exp_range', gamma=0.99994)
        callbacks = [early_stoping, save_checkpoint, clr]

        model = Stacking_Model()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=8e-5), metrics=['accuracy'])
        model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, shuffle=True, callbacks = callbacks, validation_data = (x_valid, y_valid))

    K.clear_session()