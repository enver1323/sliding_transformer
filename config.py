import os
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.preprocessing import scale_df, encode_labels_df, encode_date_df, df_to_dataset
from utils.timefeatures import time_features
from utils.tools import StandardScaler


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

    def __init__(
        self,
        lookback_window=48,
        horizon=24,
        in_features=None,
        device=torch.device('cpu')
    ):
        if in_features is None:
            in_features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

        super().__init__(
            in_features=in_features,
            out_features=['OT'],
            lookback_window=lookback_window,
            horizon=horizon,
            device=device,
        )

        self.scaler = None

    def preprocess_dataset(self, path):
        print("Data import ...")
        input_df = pd.read_csv(path)

        print("Data preprocessing ...")
        self.scaler = scale_df(input_df, self.in_features)

        encode_date_df(input_df, 'date')

        return self.split_dataset(input_df)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
