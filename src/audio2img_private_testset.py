import librosa
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import *
import numpy as np
from multiprocessing import Pool, cpu_count
import os.path
from os import walk
from os.path import join
from pydub.utils import mediainfo

SIZE = 512
audio_type = ['.mp3', '.wav', '.amr']

def audio2image(file_name):
    audio_path = '../datasets/private_test/%s'%file_name
    samples, sample_rate = librosa.load(audio_path)
    duration = librosa.get_duration(samples, sr = sample_rate)
    if duration < 120:
        m = int(120.0/float(duration))
        for i in range(m):
            samples = np.concatenate((samples, samples), axis=None)
        samples = samples[0:120*sample_rate]
    for i in range(0,110, 10):
        sub_sample = samples[sample_rate*i:sample_rate*(i+20)]
        img_path_melspectrogram = '../datasets/images/private_test/'+file_name.split('.')[0]+ '_melspectrogram_%dsec.jpg'%i
        img_path_mfcc = '../datasets/images/private_test/'+file_name.split('.')[0]+ '_mfcc_%dsec.jpg'%i

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
    afiles = []
    for (path, subdirs, files) in walk('../datasets/private_test'):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension in audio_type:
                afiles.append(file)
    df = pd.DataFrame()
    df['file'] = np.array(afiles)
    df.to_csv('../datasets/private_test.csv', header=False, index = False)

    p = Pool(12)
    p.map(func=audio2image, iterable = afiles)
    p.close()