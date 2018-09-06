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
from sklearn.model_selection import train_test_split
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, GlobalMaxPooling2D, Dense, Dropout, Flatten
NUMBER_OF_FOLD = 5
NUMBER_OF_CLASSES = 10
BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
EPOCH = 150
folds = [2,3,4]
model_sets = ['resnet50','densenet201','inception_resnet_v2','inception_v3', 'xception','densenet169']

# ftype = 'mfcc'
ftype = 'melspectrogram'

def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(to_categorical(row['class_id'], num_classes=NUMBER_OF_CLASSES))
    return np.array(y_true)
def get_y_true1(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(row['class_id'])
    return np.array(y_true)

def Stacking_Model():
    inputs = Input(shape=(len(model_sets), NUMBER_OF_CLASSES, 1))
    cnn1 = Conv2D(8, (2, 1))(inputs)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = Activation('relu')(cnn1)

    cnn1_skip = Conv2D(16, (3, 1))(cnn1)
    cnn1_skip = Activation('relu')(cnn1_skip)
    cnn1_skip = Dropout(0.25)(cnn1_skip)

    cnn2 = Conv2D(16, (2, 1))(cnn1)
    cnn2 = Activation('relu')(cnn2)
    cnn2 = Dropout(0.25)(cnn2)

    cnn2_skip = Conv2D(32, (3, 1))(cnn2)
    cnn2_skip = Activation('relu')(cnn2_skip)
    cnn2_skip = Dropout(0.25)(cnn2_skip)

    cnn3 = Conv2D(32, (2, 1))(cnn2)
    cnn3 = concatenate([cnn3, cnn1_skip], axis=3)
    cnn3 = Activation('relu')(cnn3)
    cnn3 = Dropout(0.25)(cnn3)

    cnn3_skip = Conv2D(64, (3, 1))(cnn3)
    cnn3_skip = Activation('relu')(cnn3_skip)
    cnn3_skip = Dropout(0.25)(cnn3_skip)

    cnn4 = Conv2D(64, (2, 1))(cnn3)
    cnn4 = concatenate([cnn4, cnn2_skip], axis=3)
    cnn4 = Activation('relu')(cnn4)
    cnn4 = Dropout(0.25)(cnn4)

    cnn5 = Conv2D(128, (2, 1))(cnn4)
    cnn5 = concatenate([cnn5, cnn3_skip], axis=3)
    cnn5 = Activation('relu')(cnn5)
    cnn5 = Dropout(0.25)(cnn5)

    x = Flatten()(cnn5)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    classify = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=classify)
    model.summary()
    return model

if __name__ == '__main__':
    test_df = pd.read_csv('../../datasets/test.csv', header=None)
    test_size = test_df.shape[0]
    
    p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)

    for fold in folds:
        print('***************  Fold %d  ***************'%(fold))
        valid_df = pd.read_csv('../../data/valid_set_fold%d_lv2.csv'%fold)
        valid_size = valid_df.shape[0]
        y_valid = get_y_true(valid_df)
        y_valid1 = get_y_true1(valid_df)

        x_valid = np.array([], dtype=np.float64).reshape(valid_size,0)
        x_test = np.array([], dtype=np.float64).reshape(test_size, 0)

        for mset in model_sets:
            p_valid_tmp = np.load('../../%s/data/pvalid_fold%d_%s.npy'%(mset,fold,ftype))
            p_test_tmp = np.load('../../%s/data/ptest_fold%d_%s.npy'%(mset,fold,ftype))
            x_valid = np.hstack((x_valid,p_valid_tmp))
            x_test = np.hstack((x_test,p_test_tmp))

        x_valid = np.reshape(x_valid, (valid_size, len(model_sets), NUMBER_OF_CLASSES, 1))
        x_test = np.reshape(x_test, (test_size, len(model_sets), NUMBER_OF_CLASSES, 1))

        print('SHAPE OF DATASETS:')
        print('X VALID:',x_valid.shape)
        print('Y VALID:',y_valid.shape)
        print('X TEST:',x_test.shape)

        PRETRAINED_WEIGHTS  = 'weights/pretrained_weights_fold%d_%s.hdf5'%(fold,ftype)

        kf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)

        for sub_fold, (train_index, valid_index) in enumerate(kf.split(x_valid, y_valid1)):
            x_train_fold, x_valid_fold = x_valid[train_index], x_valid[valid_index]
            y_train_fold, y_valid_fold = y_valid[train_index], y_valid[valid_index]

            WEIGHTS_BEST = 'weights/best_weights_fold%d_subfold%d_%s.hdf5'%(fold,sub_fold,ftype)

            clr = CyclicLR(base_lr=1e-8, max_lr=8e-5)
            early_stoping = EarlyStopping(monitor='val_acc', patience=20, verbose=1)
            save_checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True, mode='max')
            callbacks = [early_stoping, save_checkpoint, clr]

            model = Stacking_Model()
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=8e-5), metrics=['accuracy'])
            model.load_weights(PRETRAINED_WEIGHTS)
            model.fit(x=x_train_fold, y=y_train_fold, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, shuffle=True, callbacks = callbacks, validation_data = (x_valid_fold, y_valid_fold))

            model.load_weights(WEIGHTS_BEST)
            sub_ptest = model.predict(x_test, batch_size=TEST_BATCH_SIZE, verbose=1)
            p_test += sub_ptest

    p_test /= float(5*len(folds))
    np.save('ptest.npy', np.array(p_test))
