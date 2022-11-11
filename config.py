from abc import abstractmethod, ABC

import pandas as pd
import torch

from utils import scale_df, encode_labels_df, encode_date_df, df_to_dataset


class Config(ABC):
    """
    Generic configuration for dataset
    """

    def __init__(self, in_features, out_features, lookback_window, horizon, device=torch.device('cpu')):
        self.in_features = in_features
        self.out_features = out_features
        self.lookback_window = lookback_window
        self.horizon = horizon
        self.device = device

    @abstractmethod
    def preprocess_dataset(self, path):
        """
        Preprocesses dataset
        :param path: to the dataset file
        :return: preprocessed dataset to train
        """
        pass

    def split_dataset(self, input_df):
        """
        Splits the given dataset into train input and target output tensor values
        :param input_df: full dataset to split
        :return: x_train , y_train
        """
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


class PollutionConfig(Config):
    def __init__(self, lookback_window=7 * 24, horizon=4, device=torch.device('cpu')):
        super().__init__(
            in_features=[
                'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain', 'pollution'
            ],
            out_features=['pollution'],
            lookback_window=lookback_window,
            horizon=horizon,
            device=device,
        )
        self.scalable_params = ["dew", "temp", "wnd_spd"]
        self.encoding_params = ['wnd_dir']
        self.scaler = None
        self.label_encoder = None

    def preprocess_dataset(self, path):
        print("Data import ...")
        input_df = pd.read_csv(path)

        print("Data preprocessing ...")
        self.scaler = scale_df(input_df, self.scalable_params)
        self.label_encoder = encode_labels_df(input_df, self.encoding_params)
        encode_date_df(input_df, 'date')

        return self.split_dataset(input_df)


class StockConfig(Config):
    def __init__(self, lookback_window=64, horizon=2, device=torch.device('cpu')):
        in_features = ['Open', 'High', 'Low', 'Close']

        super().__init__(
            in_features=in_features,
            out_features=['Close'],
            lookback_window=lookback_window,
            horizon=horizon,
            device=device,
        )

    def preprocess_dataset(self, path, date_format='%Y.%m.%d %H:%M:%S'):
        print("Data import ...")
        input_df = pd.read_csv(path)

        print("Data preprocessing ...")
        encode_date_df(input_df, 'Date', date_format)

        return self.split_dataset(input_df)


class ETTConfig(Config):
    """
    ETT dataset configuration
    """

    def __init__(self, lookback_window=64, horizon=2, device=torch.device('cpu')):
        super().__init__(
            in_features=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
            out_features=['OT'],
            lookback_window=lookback_window,
            horizon=horizon,
            device=device,
        )

    def preprocess_dataset(self, path):
        print("Data import ...")
        input_df = pd.read_csv(path)

        print("Data preprocessing ...")
        encode_date_df(input_df, 'date')

        return self.split_dataset(input_df)
