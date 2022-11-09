import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import optim, nn
from sklearn.preprocessing import MinMaxScaler
from model import SlidingTransformer, SlidingAttention, PollutionPredictor
from utils import compute_path_increments

SCALABLE_PARAMS = ["dew", "temp", "wnd_spd"]
ENCODING_PARAMS = ['wnd_dir']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def scale_df(input_df, scalable_params):
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_df[SCALABLE_PARAMS] = scaler.fit_transform(input_df[scalable_params])

    return scaler


def encode_labels_df(input_df, encoding_params):
    labels_map = {}
    for param in encoding_params:
        labels = input_df[param].unique()
        labels_map[param] = dict(zip(labels, range(1, len(labels) + 1)))
        input_df[param] = input_df[param].apply(lambda item: labels_map[param][item])

    return labels_map


def encode_date_df(input_df, param, format='%Y.%m.%d %H:%M:%S'):
    input_df[param] = pd.to_datetime(input_df['date'], format=format)
    input_df[f"{param}_month"] = input_df[param].apply(lambda time: time.month)
    input_df[f"{param}_day"] = input_df[param].apply(lambda time: time.day)
    input_df[f"{param}_hour"] = input_df[param].apply(lambda time: time.hour)


def df_to_dataset(input_df, x_params, y_params, lookback, horizon):
    x = []
    y = []
    x_df = input_df[x_params].to_numpy()
    y_df = input_df[y_params].to_numpy()

    for i in range(len(x_df) - lookback - horizon):
        x.append([datum for datum in x_df[i:i + lookback]])
        y.append(y_df[i + lookback - 1: i + lookback + horizon])

    return np.array(x), np.array(y)


def train(model, x_train, y_train, epochs: int, clip_value: float = None):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    progress_bar = tqdm(range(epochs))

    for epoch in progress_bar:
        total_loss = 0
        for idx, (x, y) in enumerate(zip(x_train, y_train)):
            optimizer.zero_grad()

            x = compute_path_increments(x)
            y = compute_path_increments(y)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            loss.backward()

            if clip_value is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            progress_bar.set_description_str(
                f"[Epoch {epoch} | Step {idx + 1}] Loss: {loss.item():.2f}, Cum loss: {(total_loss / (idx + 1)):.2f}")

    return model


def app():
    print("Data import ...")
    input_df = pd.read_csv('data/pollution/train.csv')

    print("Data preprocessing ...")
    scaler = scale_df(input_df, SCALABLE_PARAMS)
    label_encoder = encode_labels_df(input_df, ENCODING_PARAMS)
    encode_date_df(input_df, 'date')

    lookback_window = 7 * 24
    horizon = 4

    in_features = [
        'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain', 'date_month', 'date_hour', 'pollution'
    ]
    out_features = ['pollution']

    x_train, y_train = df_to_dataset(input_df, in_features, out_features, lookback_window, horizon)

    x_train = torch.tensor(x_train, dtype=torch.float).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float).to(DEVICE)

    print(x_train.shape, y_train.shape)
    print("Training ...")
    kernel = 24
    sliding_transformer = SlidingTransformer(
        d_model=len(in_features),
        kernel=kernel,
        stride=kernel,
        out_dim=8,
        num_heads=2,
        dropout=0.2
    ).to(DEVICE)
    transformer_model = PollutionPredictor(sliding_transformer, horizon).to(DEVICE)
    train(model=transformer_model, x_train=x_train, y_train=y_train, epochs=10)


if __name__ == '__main__':
    app()
