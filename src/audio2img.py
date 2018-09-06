import librosa
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import *
import numpy as np
from multiprocessing import Pool, cpu_count
import os.path
from os import walk
from os.path import join

SIZE = 512

class my_element:
	def __init__(self, filename_, class_id_):
		self.filename = filename_
		self.class_id = class_id_

def audio2image_trainset(element):
    audio_path = '../datasets/train/%s'%element.file_name
    samples, sample_rate = librosa.load(audio_path)
    if element.class_id == 1 or element.class_id == 9 or element.class_id == 10:
        for i in range(0,105,5):
            sub_sample = samples[sample_rate*i:sample_rate*(i+20)]

            img_path_melspectrogram = '../datasets/images/train/'+element.file_name.split('.')[0]+ '_melspectrogram_%dsec.jpg'%i
            img_path_mfcc = '../datasets/images/train/'+element.file_name.split('.')[0]+ '_mfcc_%dsec.jpg'%i

            spectrogram = librosa.feature.melspectrogram(sub_sample, sr=sample_rate, n_mels=SIZE)
            log_S = librosa.power_to_db(spectrogram, ref=np.max)
            fig1 = plt.figure(frameon=False)
            fig1.set_size_inches(1,1)
            ax1 = plt.Axes(fig1,[0,0,1,1],frame_on = False)
            ax1.set_axis_off()
            fig1.add_axes(ax1)
            ax1.imshow(log_S,origin='lower',aspect="auto")
            fig1.savefig(img_path_melspectrogram,dpi=SIZE)
            
            # mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=40)
            # delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            # fig2 = plt.figure(frameon=False)
            # fig2.set_size_inches(1,1)
            # ax2 = plt.Axes(fig2,[0,0,1,1],frame_on = False)
            # ax2.set_axis_off()
            # fig2.add_axes(ax2)
            # ax2.imshow(delta2_mfcc,origin='lower',aspect="auto")
            # fig2.savefig(img_path_mfcc,dpi=SIZE)

            plt.close()
    else:
        for i in range(0,110, 10):
            sub_sample = samples[sample_rate*i:sample_rate*(i+20)]
            img_path_melspectrogram = '../datasets/images/train/'+element.file_name.split('.')[0]+ '_melspectrogram_%dsec.jpg'%i
            img_path_mfcc = '../datasets/images/train/'+element.file_name.split('.')[0]+ '_mfcc_%dsec.jpg'%i

            spectrogram = librosa.feature.melspectrogram(sub_sample, sr=sample_rate, n_mels=SIZE)
            log_S = librosa.power_to_db(spectrogram, ref=np.max)
            fig1 = plt.figure(frameon=False)
            fig1.set_size_inches(1,1)
            ax1 = plt.Axes(fig1,[0,0,1,1],frame_on = False)
            ax1.set_axis_off()
            fig1.add_axes(ax1)
            ax1.imshow(log_S,origin='lower',aspect="auto")
            fig1.savefig(img_path_melspectrogram,dpi=SIZE)
            
            # mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=40)
            # delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            # fig2 = plt.figure(frameon=False)
            # fig2.set_size_inches(1,1)
            # ax2 = plt.Axes(fig2,[0,0,1,1],frame_on = False)
            # ax2.set_axis_off()
            # fig2.add_axes(ax2)
            # ax2.imshow(delta2_mfcc,origin='lower',aspect="auto")
            # fig2.savefig(img_path_mfcc,dpi=SIZE)

            plt.close()

def audio2image_testset(file_name):
    audio_path = '../datasets/test/%s'%element.file_name
    samples, sample_rate = librosa.load(audio_path)
    for i in range(0,105, 5):
        sub_sample = samples[sample_rate*i:sample_rate*(i+20)]
        img_path_melspectrogram = '../datasets/images/test/'+element.file_name.split('.')[0]+ '_melspectrogram_%dsec.jpg'%i
        img_path_mfcc = '../datasets/images/test/'+element.file_name.split('.')[0]+ '_mfcc_%dsec.jpg'%i

        spectrogram = librosa.feature.melspectrogram(sub_sample, sr=sample_rate, n_mels=SIZE)
        log_S = librosa.power_to_db(spectrogram, ref=np.max)
        fig1 = plt.figure(frameon=False)
        fig1.set_size_inches(1,1)
        ax1 = plt.Axes(fig1,[0,0,1,1],frame_on = False)
        ax1.set_axis_off()
        fig1.add_axes(ax1)
        ax1.imshow(log_S,origin='lower',aspect="auto")
        fig1.savefig(img_path_melspectrogram,dpi=SIZE)
        
        # mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=40)
        # delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # fig2 = plt.figure(frameon=False)
        # fig2.set_size_inches(1,1)
        # ax2 = plt.Axes(fig2,[0,0,1,1],frame_on = False)
        # ax2.set_axis_off()
        # fig2.add_axes(ax2)
        # ax2.imshow(delta2_mfcc,origin='lower',aspect="auto")
        # fig2.savefig(img_path_mfcc,dpi=SIZE)

        plt.close()

if __name__ == '__main__':
    df1 = pd.read_csv('../datasets/train.csv', header=None)
    my_elements = []
    for idx, row in tqdm(df1.iterrows(), total=df1.shape[0]):
        file_name = row[0]
        class_id = row[1]
        tmp = my_element(file_name, class_id)
        my_elements.append(tmp)
    p1 = Pool(12)
    p1.map(func=audio2image_trainset, iterable = my_elements)
    p1.close()
    df2 = pd.read_csv('../datasets/test.csv', header=None)
    p2 = Pool(12)
    p2.map(func=audio2image_testset, iterable = df2[0].values.tolist())
    p2.close()
