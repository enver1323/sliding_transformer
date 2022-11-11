import pandas as pd
import torch

from utils import scale_df, encode_labels_df, encode_date_df, df_to_dataset


class PollutionConfig:
    def __init__(self, lookback_window=7 * 24, horizon=4, device=torch.device('cpu')):
        self.device = device

        self.scalable_params = ["dew", "temp", "wnd_spd"]
        self.encoding_params = ['wnd_dir']

        self.in_features = [
            'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain', 'pollution'
        ]
        self.out_features = ['pollution']

        self.lookback_window = lookback_window
        self.horizon = horizon

        self.scaler = None
        self.label_encoder = None

    def preprocess_dataset(self, path):
        print("Data import ...")
        input_df = pd.read_csv(path)

        print("Data preprocessing ...")
        self.scaler = scale_df(input_df, self.scalable_params)
        self.label_encoder = encode_labels_df(input_df, self.encoding_params)
        encode_date_df(input_df, 'date')

        x_train, y_train = df_to_dataset(
            input_df,
            self.in_features,
            self.out_features,
            self.lookback_window,
            self.horizon
        )

        x_train = torch.tensor(x_train, dtype=torch.float).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)

        return x_train, y_train


class StockConfig:
    def __init__(self, lookback_window=64, horizon=2, device=torch.device('cpu')):
        self.device = device

        self.in_features = [
            'Open', 'High', 'Low', 'Close'
        ]
        self.out_features = ['Close']

        self.lookback_window = lookback_window
        self.horizon = horizon

    def preprocess_dataset(self, path, date_format='%Y.%m.%d %H:%M:%S'):
        print("Data import ...")
        input_df = pd.read_csv(path)

        print("Data preprocessing ...")
        encode_date_df(input_df, 'Date', date_format)

        x_train, y_train = df_to_dataset(
            input_df,
            self.in_features,
            self.out_features,
            self.lookback_window,
            self.horizon
        )

        x_train = torch.tensor(x_train, dtype=torch.float).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)

        return x_train, y_train


class ETTConfig:
    def __init__(self, lookback_window=64, horizon=2, model_params=None, device=torch.device('cpu')):
        self.device = device

        self.in_features = [
            'Open', 'High', 'Low', 'Close'
        ]
        self.out_features = ['Close']

        self.lookback_window = lookback_window
        self.horizon = horizon

        kernel = 32
        self.model_params = {
            'd_model': len(self.in_features),
            'kernel': kernel,
            'stride': 1,
            'out_dim': 1,
            'num_heads': 4,
            'dropout': 0.2,
            **(model_params or {})
        }

    def preprocess_dataset(self, path, date_format='%Y.%m.%d %H:%M:%S'):
        print("Data import ...")
        input_df = pd.read_csv(path)

        print("Data preprocessing ...")
        encode_date_df(input_df, 'Date', date_format)

        x_train, y_train = df_to_dataset(
            input_df,
            self.in_features,
            self.out_features,
            self.lookback_window,
            self.horizon
        )

        x_train = torch.tensor(x_train, dtype=torch.float).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)

        return x_train, y_train
