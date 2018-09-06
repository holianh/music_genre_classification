import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import os
import os.path
from os import walk
from os.path import join

NUMBER_OF_FOLD = 5
if __name__ == '__main__':
    df = pd.read_csv('../datasets/train.csv', header=None)
    file_ids = []
    class_ids = []

    for idx, row in df.iterrows():
        file_ids.append(row[0].split('.')[0])
        class_ids.append(row[1] - 1)
    new_df = pd.DataFrame()
    new_df['file_id'] = np.array(file_ids)
    new_df['class_id'] = np.array(class_ids)

    new_df.to_csv('../datasets/new_train.csv', index=False)
    
    kf = StratifiedKFold(n_splits= NUMBER_OF_FOLD, shuffle=True, random_state=8)
    for fold, (train_index, valid_index) in enumerate(kf.split(new_df, new_df['class_id'].values)):
        train_df = new_df.iloc[train_index]
        valid_df = new_df.iloc[valid_index]
        train_df = shuffle(train_df)
        valid_df = shuffle(valid_df)

        img_melspectrogram_paths = []
        img_mfcc_paths = []
        class_ids = []
        for idx, row in train_df.iterrows():
            file_id = row['file_id']
            class_id = row['class_id']
            if class_id == 0 or class_id == 8 or class_id == 9:
                for i in range(0,105,5):
                    img_path = '../../datasets/images/train/%s_melspectrogram_%dsec.jpg'%(file_id, i)
                    img_melspectrogram_paths.append(img_path)
                    img_mfcc_paths.append(img_path.replace('melspectrogram', 'mfcc'))
                    class_ids.append(class_id)
            else:
                for i in range(0,110,10):
                    img_path = '../../datasets/images/train/%s_melspectrogram_%dsec.jpg'%(file_id, i)
                    img_melspectrogram_paths.append(img_path)
                    img_mfcc_paths.append(img_path.replace('melspectrogram', 'mfcc'))
                    class_ids.append(class_id)

        new_train_df = pd.DataFrame()
        new_train_df['melspectrogram'] = np.array(img_melspectrogram_paths)
        new_train_df['mfcc'] = np.array(img_mfcc_paths)
        new_train_df['class_id'] = np.array(class_ids)

        new_train_df = shuffle(new_train_df)
        new_train_df.to_csv('../data/train_set_fold%d.csv'%fold, index=False)

        img_melspectrogram_paths = []
        img_mfcc_paths = []
        class_ids = []
        for idx, row in valid_df.iterrows():
            file_id = row['file_id']
            class_id = row['class_id']
            if class_id == 0 or class_id == 8 or class_id == 9:
                for i in range(0,105,5):
                    img_path = '../../datasets/images/train/%s_melspectrogram_%dsec.jpg'%(file_id, i)
                    img_melspectrogram_paths.append(img_path)
                    img_mfcc_paths.append(img_path.replace('melspectrogram', 'mfcc'))
                    class_ids.append(class_id)
            else:
                for i in range(0,110,10):
                    img_path = '../../datasets/images/train/%s_melspectrogram_%dsec.jpg'%(file_id, i)
                    img_melspectrogram_paths.append(img_path)
                    img_mfcc_paths.append(img_path.replace('melspectrogram', 'mfcc'))
                    class_ids.append(class_id)

        new_valid_df = pd.DataFrame()
        new_valid_df['melspectrogram'] = np.array(img_melspectrogram_paths)
        new_valid_df['mfcc'] = np.array(img_mfcc_paths)
        new_valid_df['class_id'] = np.array(class_ids)

        new_valid_df = shuffle(new_valid_df)

        new_valid_df.to_csv('../data/valid_set_fold%d.csv'%fold, index=False)
