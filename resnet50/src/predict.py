import os
import os.path
import pandas as pd
import numpy as np
import cv2
import threading

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, Input, Dropout, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.utils.np_utils import to_categorical
from keras import *
from keras.models import *
from keras.preprocessing import image
from keras import backend as K

import imgaug as ia
from imgaug import augmenters as iaa
from random import randint

NUMBER_OF_CLASSES = 10
SIZE = 224
NUMBER_OF_FOLD = 5
BATCH_SIZE = 64

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
def train_generator_tta(df, batch_size, choice, sec):
	for start in range(0, df.shape[0], batch_size):
		end = min(start + batch_size, df.shape[0])
		sub_df = df.iloc[start:end,:]
		x_batch = []
		for index, row in sub_df.iterrows():
			file_id = row['file_id']
			img_path = '../../datasets/images/train/%s_%s_%dsec.jpg'%(file_id, ftype, sec)
			img = cv2.imread(img_path)
			img = cv2.resize(img,(SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
			
			if choice == 1:
				img = np.fliplr(img)

			img = image.img_to_array(img)
			img = preprocess_input(img)
			x_batch.append(img)
		yield np.array(x_batch)

@threadsafe_generator
def test_generator_tta(df, batch_size, choice, sec):
	for start in range(0, df.shape[0], batch_size):
		end = min(start + batch_size, df.shape[0])
		sub_df = df.iloc[start:end,:]
		x_batch = []
		for index, row in sub_df.iterrows():
			file_id = row[0].split('.')[0]
			img_path = '../../datasets/images/test/%s_%s_%dsec.jpg'%(file_id, ftype, sec)
			img = cv2.imread(img_path)
			img = cv2.resize(img,(SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
			
			if choice == 1:
				img = np.fliplr(img)

			img = image.img_to_array(img)
			img = preprocess_input(img)
			x_batch.append(img)
		yield np.array(x_batch)

@threadsafe_generator
def pseudo_generator_tta(df, batch_size, choice, sec):
	for start in range(0, df.shape[0], batch_size):
		end = min(start + batch_size, df.shape[0])
		sub_df = df.iloc[start:end,:]
		x_batch = []
		for index, row in sub_df.iterrows():
			file_id = row['file_id']
			img_path = '../../datasets/images/test/%s_%s_%dsec.jpg'%(file_id, ftype, sec)
			img = cv2.imread(img_path)
			img = cv2.resize(img,(SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
			
			if choice == 1:
				img = np.fliplr(img)

			img = image.img_to_array(img)
			img = preprocess_input(img)
			x_batch.append(img)
		yield np.array(x_batch)

def ResNet50_Model():
	input_tensor = Input(shape=(SIZE, SIZE, 3))
	base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
	bn = BatchNormalization()(input_tensor)
	x = base_model(bn)
	x = GlobalMaxPooling2D()(x)
	x = Dense(256, activation='relu')(x)
	x = Dropout(0.25)(x)
	output_tensor = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
	model = Model(inputs=input_tensor, outputs=output_tensor)
	return model

if __name__ == '__main__':
	test_df = pd.read_csv('../../datasets/test.csv', header=None)
	test_size = test_df.shape[0]
	test_steps = np.ceil(float(test_size) / float(BATCH_SIZE))
	pseudo_df = pd.read_csv('../../datasets/pseudo_dataset.csv')
	pseudo_size = pseudo_df.shape[0]
	pseudo_steps = np.ceil(float(pseudo_size) / float(BATCH_SIZE))

	for fold in range(2,5,1):
		print('***************  Fold %d  ***************'%(fold))
		train_df = pd.read_csv('../../data/train_set_fold%d_lv2.csv'%fold)
		valid_df = pd.read_csv('../../data/valid_set_fold%d_lv2.csv'%fold)
		
		train_size = train_df.shape[0]
		valid_size = valid_df.shape[0]
		train_steps = np.ceil(float(train_size) / float(BATCH_SIZE))
		valid_steps = np.ceil(float(valid_size) / float(BATCH_SIZE))
		print('TRAIN SIZE: %d VALID SIZE: %d'%(train_size, valid_size))

		WEIGHTS_BEST = '../weights/pseudo_best_weights_fold%d_%s.hdf5'%(fold,ftype)

		model = ResNet50_Model()
		model.load_weights(WEIGHTS_BEST)
		
		print('Predict train set...')
		p_train = np.zeros((train_size, NUMBER_OF_CLASSES), dtype = np.float64)
		for choice in range(2):
			for sec in range(0,110,10):
				p_train_tmp = model.predict_generator(generator=train_generator_tta(train_df, BATCH_SIZE, choice, sec), steps=train_steps, verbose=1)
				p_train += p_train_tmp
		p_train = p_train/22.0
		np.save('../data/ptrain_fold%d_%s.npy'%(fold,ftype), np.array(p_train))
		
		print('Predict pseudo set...')
		p_pseudo = np.zeros((pseudo_size, NUMBER_OF_CLASSES), dtype = np.float64)
		for choice in range(2):
			for sec in range(0,110,10):
				p_pseudo_tmp = model.predict_generator(generator=pseudo_generator_tta(pseudo_df, BATCH_SIZE, choice, sec), steps=pseudo_steps, verbose=1)
				p_pseudo += p_pseudo_tmp
		p_pseudo = p_pseudo/22.0
		np.save('../data/ppseudo_fold%d_%s.npy'%(fold,ftype), np.array(p_pseudo))

		print('Predict valid set...')
		p_valid = np.zeros((valid_size, NUMBER_OF_CLASSES), dtype = np.float64)
		for choice in range(2):
			for sec in range(0,110,10):
				p_valid_tmp = model.predict_generator(generator=train_generator_tta(valid_df, BATCH_SIZE, choice, sec), steps=valid_steps, verbose=1)
				p_valid += p_valid_tmp
		p_valid = p_valid/22.0
		np.save('../data/pvalid_fold%d_%s.npy'%(fold,ftype), np.array(p_valid))
		
		# print('Predict test set...')
		# p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)
		# for choice in range(2):
		# 	for sec in range(0,110,10):
		# 		p_test_tmp = model.predict_generator(generator=test_generator_tta(test_df, BATCH_SIZE, choice, sec), steps=test_steps, verbose=1)
		# 		p_test += p_test_tmp
		# p_test = p_test/22.0
		# np.save('../data/ptest_fold%d_%s.npy'%(fold,ftype), np.array(p_test))
	K.clear_session()
