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
    test_df = pd.read_csv('../../datasets/private_test.csv', header=None)
    test_size = test_df.shape[0]

    p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)

    for fold in folds:
        x_test = np.array([], dtype=np.float64).reshape(test_size, 0)

        for mset in model_sets:
            p_test_tmp = np.load('../../%s/data/ptest_fold%d_%s.npy'%(mset,fold,ftype))
            x_test = np.hstack((x_test,p_test_tmp))
        x_test = np.reshape(x_test, (test_size, len(model_sets), NUMBER_OF_CLASSES, 1))

        for sub_fold in range(NUMBER_OF_FOLD):
            WEIGHTS_BEST = 'weights/best_weights_fold%d_subfold%d_%s.hdf5'%(fold,sub_fold,ftype)
            model = Stacking_Model()
            model.load_weights(WEIGHTS_BEST)
            sub_ptest = model.predict(x_test, batch_size=TEST_BATCH_SIZE, verbose=1)
            p_test += sub_ptest
    p_test /= float(len(folds)*NUMBER_OF_FOLD)
    np.save('ptest.npy', np.array(p_test))