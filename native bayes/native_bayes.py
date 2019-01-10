import random
import numpy as np
import pandas as pd

from data_read import get_raw_data

# print(data_set)


def create_training_test(dataset, fraction_training, msg=False):
    traning_dataset = dataset.copy()
    test_dataset = dataset.copy()
    total_number = len(dataset)
    train_number = round(total_number * fraction_training)
    test_number = total_number - train_number
    total_list_idx = list(dataset.index.values)
    print(total_list_idx)
    train_list_idx = random.sample(list(dataset.index.values), train_number)
    print(train_list_idx)
    test_list_idx = list(set(total_list_idx) - set(train_list_idx))
    train_list_idx.sort()
    test_list_idx.sort()
    print(traning_dataset.index[test_list_idx])
    traning_dataset.drop(
        traning_dataset.index[test_list_idx], inplace=True)
    test_dataset.drop(
        test_dataset.index[train_list_idx], inplace=True)
    return traning_dataset, test_dataset


def get_parameters(dataset, msg=False):
    features = dataset.columns.values
    nbins = 10
    dic_parameters = {}
    for i in range(len(features) - 1):
        aux_df = pd.DataFrame(dataset[features[i]])
        aux_df['bin'] = pd.cut(aux_df[features[i]], nbins)
        counts = pd.value_counts(aux_df['bin'])
        print(counts)
        points_X = np.zeros(nbins)
        points_Y = np.zeros(nbins)

        for j in range(nbins):
            points_X[j] = counts.index[j].mid
            points_Y[j] = counts.iloc[j]
            print(counts.index[j].mid)
            print(counts.iloc[j])
        total_Y = np.sum(points_Y)
        mu = np.sum(points_X * points_Y) / total_Y
        sigma2 = np.sum((points_X - mu) ** 2 * points_Y)/(total_Y - 1)
        sigma = np.sqrt(sigma2)
        dic_parameters[features[i]] = (mu, sigma)

    return dic_parameters


data_set = get_raw_data()
# create_training_test(data_set, 0.7)
get_parameters(data_set)
