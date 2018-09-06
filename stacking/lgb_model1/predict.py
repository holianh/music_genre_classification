import pandas as pd
import numpy as np
import lightgbm as lgb

NUMBER_OF_FOLD = 5
NUMBER_OF_CLASSES = 10

folds = [2,3,4]
model_sets = ['resnet50','densenet201','inception_resnet_v2','inception_v3', 'xception','densenet169']
# ftype = 'mfcc'
ftype = 'melspectrogram'
if __name__ == '__main__':
    test_df = pd.read_csv('../../datasets/private_test.csv', header=None)
    test_size = test_df.shape[0]

    p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)

    for fold in folds:
        x_test = np.array([], dtype=np.float64).reshape(test_size, 0)

        for mset in model_sets:
            p_test_tmp = np.load('../../%s/data/ptest_fold%d_%s.npy'%(mset,fold,ftype))
            x_test = np.hstack((x_test,p_test_tmp))

        for sub_fold in range(NUMBER_OF_FOLD):
            WEIGHTS_BEST = 'weights/best_weights_fold%d_subfold%d.txt'%(fold,sub_fold)
            model = lgb.Booster(model_file=WEIGHTS_BEST)

            sub_ptest = model.predict(x_test)
            p_test += sub_ptest
    p_test /= float(len(folds)*NUMBER_OF_FOLD)
    np.save('ptest.npy', np.array(p_test))