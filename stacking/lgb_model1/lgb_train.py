import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

NUMBER_OF_FOLD = 5
NUMBER_OF_CLASSES = 10

folds = [2,3,4]
model_sets = ['resnet50','densenet201','inception_resnet_v2','inception_v3', 'xception','densenet169']

# ftype = 'mfcc'
ftype = 'melspectrogram'

def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(row['class_id'])
    return np.array(y_true)

if __name__ == '__main__':
    test_df = pd.read_csv('../../datasets/test.csv', header=None)
    test_size = test_df.shape[0]
    
    p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)

    for fold in folds:
        print('***************  Fold %d  ***************'%(fold))
        valid_df = pd.read_csv('../../data/valid_set_fold%d_lv2.csv'%fold)
        valid_size = valid_df.shape[0]
        y_valid = get_y_true(valid_df)

        x_valid = np.array([], dtype=np.float64).reshape(valid_size,0)
        x_test = np.array([], dtype=np.float64).reshape(test_size, 0)

        for mset in model_sets:
            p_valid_tmp = np.load('../../%s/data/pvalid_fold%d_%s.npy'%(mset,fold,ftype))
            p_test_tmp = np.load('../../%s/data/ptest_fold%d_%s.npy'%(mset,fold,ftype))
            x_valid = np.hstack((x_valid,p_valid_tmp))
            x_test = np.hstack((x_test,p_test_tmp))

        np.save('data/x_test_fold%d.npy'%fold, np.array(x_test))

        print('SHAPE OF DATASETS:')
        print('X VALID:',x_valid.shape)
        print('Y VALID:',y_valid.shape)
        print('X TEST:',x_test.shape)

        kf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)

        for sub_fold, (train_index, valid_index) in enumerate(kf.split(x_valid, y_valid)):
            x_train_fold, x_valid_fold = x_valid[train_index], x_valid[valid_index]
            y_train_fold, y_valid_fold = y_valid[train_index], y_valid[valid_index]

            WEIGHTS_BEST = 'weights/best_weights_fold%d_subfold%d.txt'%(fold,sub_fold)

            model = lgb.LGBMClassifier(
                    nthread=12,
                    n_estimators=10000,
                    learning_rate=0.05,
                    num_leaves=128,
                    class_weight = 'balanced',
                    # boosting_type = "gbdt",
                    boosting_type = "dart",
                    feature_fraction = 0.7,
                    bagging_fraction= 0.7,
                    max_depth=-1,
                    reg_alpha=0.041545473,
                    reg_lambda=0.0735294,
                    min_split_gain=0.0222415,
                    min_child_weight=39.3259775,
                    silent=-1,
                    verbose=-1,
                    num_class=NUMBER_OF_CLASSES,
                    random_state = 8)

            model.fit(x_train_fold, y_train_fold, eval_set=[(x_valid_fold, y_valid_fold)], eval_metric= 'softmax', verbose= 50, early_stopping_rounds= 150)
            model.booster_.save_model(WEIGHTS_BEST)