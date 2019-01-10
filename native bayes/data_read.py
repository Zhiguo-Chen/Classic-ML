import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from numpy import *
import math
from IPython.display import display

pi = math.pi

data_path = './diabetes.csv'

raw_dataset = pd.read_csv(filepath_or_buffer=data_path)
# print(raw_dataset)
# print(raw_dataset.columns)
# print(len(raw_dataset))
# print(raw_dataset.head())
positive = raw_dataset[raw_dataset['Outcome'] == 1]
negative = raw_dataset[raw_dataset['Outcome'] == 0]
# print(raw_dataset['Outcome'] == 1)
# print(len(positive))
# print(len(negative))

print('=====')


def show_data():
    df = pd.DataFrame(raw_dataset, columns=raw_dataset.columns.drop('Outcome'))
    pd.plotting.scatter_matrix(df, c=raw_dataset['Outcome'].values, figsize=(15, 15), marker='o', hist_kwds={
        'bins': 10, 'color': 'green'}, s=10, alpha=.2, cmap=plt.get_cmap('bwr'))
    plt.show()

    df = pd.DataFrame(positive, columns=positive.columns.drop('Outcome'))
    pd.plotting.scatter_matrix(df, c='red', figsize=(15, 15), marker='o', hist_kwds={
        'bins': 10, 'color': 'red'}, s=10, alpha=.2)
    plt.show()

    df = pd.DataFrame(negative, columns=positive.columns.drop('Outcome'))
    pd.plotting.scatter_matrix(df, c='blue', figsize=(15, 15), marker='o', hist_kwds={
        'bins': 10, 'color': 'blue'}, s=10, alpha=.2)
    plt.show()


def get_raw_data():
    return raw_dataset


def main():
    # print(list(raw_dataset.index.values))
    df = pd.DataFrame(raw_dataset, columns=raw_dataset.columns.drop('Outcome'))
    # print(raw_dataset.columns.values)
    # print(pd.DataFrame(raw_dataset['BMI']))
    print(raw_dataset.index[0, 3, 7, 8])


if __name__ == "__main__":
    main()
