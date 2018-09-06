import os
import os.path
import pandas as pd
import numpy as np
import cv2
import threading

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Dense, Input, Dropout, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils.np_utils import to_categorical
from keras import *
from keras.models import *
from keras.preprocessing import image
from keras import backend as K
from random import randint
import random

NUMBER_OF_CLASSES = 10
SIZE = 299
NUMBER_OF_FOLD = 5
BATCH_SIZE = 20
EPOCH = 100

# ftype = 'mfcc'
ftype = 'melspectrogram'

class ThreadSafeIterator:
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			return self.it.__next__()

def threadsafe_generator(f):
	"""
	A decorator that takes a generator function and makes it thread-safe.
	"""

	def g(*args, **kwargs):
		return ThreadSafeIterator(f(*args, **kwargs))

	return g

@threadsafe_generator
def train_generator(df, batch_size):
	while True:
		df = df.sample(frac=1, random_state = randint(11, 99)).reset_index(drop=True)
		for start in range(0, df.shape[0], batch_size):
			end = min(start + batch_size, df.shape[0])
			sub_df = df.iloc[start:end,:]
			x_batch = []
			y_batch = []
			for index, row in sub_df.iterrows():
				img_path = row[ftype]
				img = cv2.imread(img_path)
				img = cv2.resize(img,(SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
				
				if random.random() < 0.5:
					img = np.fliplr(img)
				
				img = image.img_to_array(img)
				img = preprocess_input(img)
				x_batch.append(img)
				y_batch.append(to_categorical(row['class_id'], num_classes=NUMBER_OF_CLASSES))
			yield np.array(x_batch), np.array(y_batch)

@threadsafe_generator
def valid_generator(df, batch_size):
	while True:
		for start in range(0, df.shape[0], batch_size):
			end = min(start + batch_size, df.shape[0])
			sub_df = df.iloc[start:end,:]
			x_batch = []
			y_batch = []
			for index, row in sub_df.iterrows():
				img_path = row[ftype]
				img = cv2.imread(img_path)
				img = cv2.resize(img,(SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
				img = image.img_to_array(img)
				img = preprocess_input(img)
				x_batch.append(img)
				y_batch.append(to_categorical(row['class_id'], num_classes=NUMBER_OF_CLASSES))
			yield np.array(x_batch), np.array(y_batch)

def InceptionResNetV2_Model():
	input_tensor = Input(shape=(SIZE, SIZE, 3))
	base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
	bn = BatchNormalization()(input_tensor)
	x = base_model(bn)
	x = GlobalMaxPooling2D()(x)
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.25)(x)
	output_tensor = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
	model = Model(inputs=input_tensor, outputs=output_tensor)
	return model

if __name__ == '__main__':
	for fold in range(2,5,1):
		print('***************  Fold %d  ***************'%(fold))
		train_df = pd.read_csv('../../data/train_set_fold%d.csv'%(fold))
		pseudo_df = pd.read_csv('../../data/pseudo_trainset.csv')
		train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
		valid_df = pd.read_csv('../../data/valid_set_fold%d.csv'%(fold))
		train_size = train_df.shape[0]
		valid_size = valid_df.shape[0]
		train_steps = np.ceil(float(train_size) / float(BATCH_SIZE))
		valid_steps = np.ceil(float(valid_size) / float(BATCH_SIZE))
		print('TRAIN SIZE: %d VALID SIZE: %d'%(train_size, valid_size))

		PSEUDO_WEIGHTS_BEST = '../weights/pseudo_best_weights_fold%d_%s.hdf5'%(fold,ftype)
		WEIGHTS_BEST = '../weights/best_weights_fold%d_%s.hdf5'%(fold,ftype)
		TRAINING_LOG = '../logs/training_logs_fold%d_%s.csv'%(fold,ftype)
		early_stoping = EarlyStopping(monitor='val_acc', patience=8, verbose=1)
		save_checkpoint = ModelCheckpoint(PSEUDO_WEIGHTS_BEST, monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True, mode='max')
		reduce_lr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 4, min_lr = 1e-8, verbose=1)
		csv_logger = CSVLogger(TRAINING_LOG, append=True)
		callbacks_warmup = [save_checkpoint,csv_logger]
		callbacks = [early_stoping, save_checkpoint, reduce_lr, csv_logger]

		model = InceptionResNetV2_Model()

		# warm up
		for layer in model.layers[0:-3]:
			layer.trainable = False
		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=8e-5), metrics=['accuracy'])
		model.summary()
		model.fit_generator(generator=train_generator(train_df, BATCH_SIZE), steps_per_epoch=train_steps, epochs=1, verbose=1,
							validation_data=valid_generator(valid_df, BATCH_SIZE), validation_steps=valid_steps, callbacks=callbacks_warmup)

		for layer in model.layers:
			layer.trainable = True
		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=4e-5), metrics=['accuracy'])
		model.summary()
		model.fit_generator(generator=train_generator(train_df, BATCH_SIZE), steps_per_epoch=train_steps, epochs=EPOCH, verbose=1,
							validation_data=valid_generator(valid_df, BATCH_SIZE), validation_steps=valid_steps, callbacks=callbacks)

	K.clear_session()
