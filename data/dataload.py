# -*- coding: utf-8 -*-
# @Time    : 2021/4/25 11:51
# @File    : dataload.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import sklearn

from config import args_parser
args = args_parser()

np.random.seed(120)


####  时间窗函数  ####
def multivariate_data(dataset, target, start_index,
                      end_index, history_size, step, time_step):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) + 1

    for i in range(start_index, end_index, time_step):
        indices = range(i - history_size, i, step)

        data.append(dataset[indices])
        labels.append(np.isin(1, target[indices]))

    return np.array(data, dtype=float), np.array(labels, dtype=int)


##时间窗大小
seq_len = args.seq_len

def loading(data_name):

    if data_name == 'swat':
        # pass
        all_data = pd.read_csv('data/swatfenbu.csv')
        all_data = all_data.drop('Timestamp', 1)  # 删除timestamp列

        label = all_data.iloc[:, -1]
        print(pd.DataFrame(label).apply(pd.value_counts))

        dataset_train = all_data[0: int(0.8 * len(all_data))]
        dataset_test = all_data[int(0.8 * len(all_data)): -1]

        # Normalization
        dataset_normalizer = StandardScaler().fit(dataset_train.iloc[:, :-1])
        dataset_train_ = dataset_normalizer.transform(dataset_train.iloc[:, :-1])
        dataset_test_ = dataset_normalizer.transform(dataset_test.iloc[:, :-1])

        ### 调用时间窗
        TRAIN_SPLIT = int(0.8 * len(dataset_train_))
        train_X, train_y = multivariate_data(dataset_train_, np.array(dataset_train.iloc[:, -1]),
                        0, TRAIN_SPLIT, history_size=seq_len, step=1, time_step=1)

        val_X, val_y = multivariate_data(dataset_train_, np.array(dataset_train.iloc[:, -1]),
                        TRAIN_SPLIT, None, history_size=seq_len, step=1, time_step=seq_len)

        test_X, test_y = multivariate_data(dataset_test_, np.array(dataset_test.iloc[:, -1]),
                        0, None, history_size=seq_len, step=1, time_step=seq_len)

        x_train, y_train = sklearn.utils.shuffle(train_X, train_y)
        val_X, val_y = sklearn.utils.shuffle(val_X, val_y)
        test_X, test_y = sklearn.utils.shuffle(test_X, test_y)
        # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
        print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)
        print(pd.DataFrame(train_y).apply(pd.value_counts))
        print(pd.DataFrame(val_y).apply(pd.value_counts))
        print(pd.DataFrame(test_y).apply(pd.value_counts))

        return train_X, train_y, val_X, val_y, test_X, test_y


    elif data_name == 'wind':
        all_data = pd.read_csv('data/allData.csv', nrows=1000)
        all_data = all_data.sort_values('time')
        all_data = all_data.drop('time', 1)  # 删除timestamp列


        label = all_data.iloc[:, -1]
        print(pd.DataFrame(label).apply(pd.value_counts))

        dataset_train = all_data[0: int(0.8 * len(all_data))]
        dataset_test = all_data[int(0.8 * len(all_data)): -1]

        # Normalization
        dataset_normalizer = StandardScaler().fit(dataset_train.iloc[:, :-1])
        dataset_train_ = dataset_normalizer.transform(dataset_train.iloc[:, :-1])
        dataset_test_ = dataset_normalizer.transform(dataset_test.iloc[:, :-1])

        ### 调用时间窗
        TRAIN_SPLIT = int(0.8 * len(dataset_train_))
        train_X, train_y = multivariate_data(dataset_train_, np.array(dataset_train.iloc[:, -1]),
                                             0, TRAIN_SPLIT, history_size=seq_len, step=1)

        val_X, val_y = multivariate_data(dataset_train_, np.array(dataset_train.iloc[:, -1]),
                                         TRAIN_SPLIT, None, history_size=seq_len, step=1)

        test_X, test_y = multivariate_data(dataset_test_, np.array(dataset_test.iloc[:, -1]),
                                           0, None, history_size=seq_len, step=1)

        x_train, y_train = sklearn.utils.shuffle(train_X, train_y)
        # x_val, y_val = sklearn.utils.shuffle(val_X, val_y)
        # x_test, y_test = sklearn.utils.shuffle(test_X, test_y)
        # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)
        print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)
        print(pd.DataFrame(train_y).apply(pd.value_counts))
        print(pd.DataFrame(val_y).apply(pd.value_counts))
        print(pd.DataFrame(test_y).apply(pd.value_counts))

        return train_X, train_y, val_X, val_y, test_X, test_y

    elif data_name == 'swat_duan':
        # X = np.load('data/swat_shan.npy')
        # y = np.load('data/swat_y.npy')

        # x_train = X[0: int(0.8 * len(X)),:,:]
        # x_test = X[int(0.8 * len(X)): -1,:,:]
        # y_train = y[0: int(0.8 * len(y))]
        # y_test = y[int(0.8 * len(y)): -1]

        x_train = np.load('data/swat_train_x.npy')
        y_train = np.load('data/swat_train_y.npy')

        x_test = np.load('data/swat_test_x.npy')
        y_test = np.load('data/swat_test_y.npy')

        # x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        # x_val, y_val = sklearn.utils.shuffle(val_X, val_y)
        x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

        return x_train, y_train, x_test, y_test

    elif data_name == 'act':

        x_train = np.load('data/Activity/X_train.npy', allow_pickle=True)
        y_train = np.load('data/Activity/y_train_2.npy', allow_pickle=True)

        x_test = np.load('data/Activity/X_test.npy', allow_pickle=True)
        y_test = np.load('data/Activity/y_test_2.npy', allow_pickle=True)

        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        # x_val, y_val = sklearn.utils.shuffle(val_X, val_y)
        x_test, y_test = sklearn.utils.shuffle(x_test, y_test)


        return x_train, y_train, x_test, y_test

    elif data_name == 'HT':
        # X = np.load('data/swat_shan.npy')
        # y = np.load('data/swat_y.npy')

        # x_train = X[0: int(0.8 * len(X)),:,:]
        # x_test = X[int(0.8 * len(X)): -1,:,:]
        # y_train = y[0: int(0.8 * len(y))]
        # y_test = y[int(0.8 * len(y)): -1]

        x_train = np.load('data/HT_Sensor/X_train.npy', allow_pickle=True)
        y_train = np.load('data/HT_Sensor/y_train_2.npy', allow_pickle=True)

        x_test = np.load('data/HT_Sensor/X_test.npy', allow_pickle=True)
        y_test = np.load('data/HT_Sensor/y_test_2.npy', allow_pickle=True)

        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        # x_val, y_val = sklearn.utils.shuffle(val_X, val_y)
        x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

        return x_train, y_train, x_test, y_test

    elif data_name == 'har':
        x_train = np.load('data/uar/trainX_down.npy')
        y_train = np.load('data/uar/trainy_down.npy')
        x_test = np.load('data/uar/testX_down.npy')
        y_test = np.load('data/uar/testy_down.npy')

        return x_train, y_train, x_test, y_test

    elif data_name == 'eeg':
        x_train = np.load('data/eeg2/X_train.npy')
        y_train = np.load('data/eeg2/y_train.npy')
        x_test = np.load('data/eeg2/X_test.npy')
        y_test = np.load('data/eeg2/y_test.npy')

        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        # x_val, y_val = sklearn.utils.shuffle(val_X, val_y)
        x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

        return x_train, y_train, x_test, y_test

    elif data_name == 'wafer':
        x_train = np.load('data/Wafer/X_train.npy')
        y_train = np.load('data/Wafer/y_train_2.npy')
        x_test = np.load('data/Wafer/X_test.npy')
        y_test = np.load('data/Wafer/y_test_2.npy')

        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        # x_val, y_val = sklearn.utils.shuffle(val_X, val_y)
        x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

        return x_train, y_train, x_test, y_test
    elif data_name == 'occupation':
        x_train = np.load('data/occupation/X_train.npy')
        y_train = np.load('data/occupation/y_train.npy')
        x_test = np.load('data/occupation/X_test.npy')
        y_test = np.load('data/occupation/y_test.npy')

        # x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        # # x_val, y_val = sklearn.utils.shuffle(val_X, val_y)
        x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

        return x_train, y_train, x_test, y_test

    elif data_name == 'earth':
        train_data = pd.read_csv('data/Earthquakes/Earthquakes_TRAIN.tsv', sep='\t', header=None)
        test_data = pd.read_csv('data/Earthquakes/Earthquakes_TEST.tsv', sep='\t', header=None)

        x_train = train_data.iloc[:,1:].values
        y_train = train_data.iloc[:,0].values

        x_test = test_data.iloc[:, 1:].values
        y_test = test_data.iloc[:, 0].values

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        return x_train, y_train, x_test, y_test

    elif data_name == 'walkvsrun':
        x_train = np.load('data/walkvsrun/X_train.npy')
        y_train = np.load('data/walkvsrun/y_train_2.npy')
        x_test = np.load('data/walkvsrun/X_test.npy')
        y_test = np.load('data/walkvsrun/y_test_2.npy')

        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        # x_val, y_val = sklearn.utils.shuffle(val_X, val_y)
        x_test, y_test = sklearn.utils.shuffle(x_test, y_test)

        return x_train, y_train, x_test, y_test
 
    else:
        pass

    # sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=0)
    # (train_idx, val_idx) = next(sss.split(X, y))
    #
    # x_train, x_test = X[train_idx], X[val_idx]
    # y_train, y_test = y[train_idx], y[val_idx]
    # print("x_train:", x_train.shape, "x_test:", x_test.shape)

    # ### 统计正负样本
    # from collections import Counter
    # print(Counter(y_test))  ## Counter({0.0: 143875, 1.0: 6013})
    # print(Counter(y_train))  ## Counter({0.0: 250276, 1.0: 49693})

    # return x_train, y_train, x_test, y_test
