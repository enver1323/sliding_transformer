import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def compute_path_increments(inputs) -> torch.Tensor:
    return inputs[:, 1:, :] - inputs[:, :-1, :]


def scale_df(input_df, scalable_params):
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_df[scalable_params] = scaler.fit_transform(input_df[scalable_params])

    return scaler


def encode_labels_df(input_df, encoding_params):
    labels_map = {}
    for param in encoding_params:
        labels = input_df[param].unique()
        labels_map[param] = dict(zip(labels, range(1, len(labels) + 1)))
        input_df[param] = input_df[param].apply(lambda item: labels_map[param][item])

    return labels_map


def encode_date_df(input_df, param, date_format='%Y.%m.%d %H:%M:%S'):
    input_df[param] = pd.to_datetime(input_df[param], format=date_format)
    input_df[f"{param}_year"] = input_df[param].apply(lambda time: time.month)
    input_df[f"{param}_month"] = input_df[param].apply(lambda time: time.month)
    input_df[f"{param}_day"] = input_df[param].apply(lambda time: time.day)
    input_df[f"{param}_hour"] = input_df[param].apply(lambda time: time.hour)


def df_to_dataset(input_df, x_params, y_params, lookback, horizon):
    x = []
    y = []
    x_df = input_df[x_params].to_numpy()
    y_df = input_df[y_params].to_numpy()

    for i in range(len(x_df) - lookback - horizon + 1):
        x.append([datum for datum in x_df[i:i + lookback]])
        y.append(y_df[i + lookback - 1: i + lookback + horizon - 1])

    return np.array(x), np.array(y)
