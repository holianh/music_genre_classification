import pandas as pd
import numpy as np

NUMBER_OF_CLASSES = 10
NUMBER_OF_FOLD = 5
sets =   ['cnn_model1' ,'cnn_model2', 'lgb_model1']
weighs = [0.45      ,     0.45,            0.1]

if __name__ == '__main__':
    test_df = pd.read_csv('../datasets/private_test.csv', header=None)
    test_size = test_df.shape[0]
    p_test = np.zeros((test_size, NUMBER_OF_CLASSES), dtype = np.float64)

    for idx, mset in enumerate(sets):
        p_test_tmp = np.load('%s/ptest.npy'%(mset))
        p_test += weighs[idx] * p_test_tmp

    ids = [] 
    genres = []
    best_idx = np.argmax(p_test, axis=1)

    print(test_df.shape, p_test.shape, best_idx.shape)

    for idx, row in test_df.iterrows():
        ids.append(row[0])
        genres.append(best_idx[idx] + 1)
    sub = pd.DataFrame()
    sub['Id'] = np.array(ids)
    sub['Genre'] = np.array(genres)
    sub.to_csv('../submission.csv', index=False)