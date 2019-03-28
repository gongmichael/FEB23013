# %%
import numpy as np
import pandas as pd
from os.path import join as pjoin

class DataField(object):
    def __init__(self, features, labels):
        if features.shape[0] != labels.shape[0]:
            raise IndexError(
                "feature and labels must have same first dimensions")
        self.x = features
        self.y = labels

    @property
    def input_dims(self):
        input_dims = (None, ) + self.x.shape[1:]
        return input_dims

    @property
    def output_dims(self):
        output_dims = (None, ) + self.y.shape[1:]
        return output_dims


class DataSet(object):
    def __init__(self,
                 features,
                 labels,
                 sequence=False,
                 sequence_size=5,
                 sequence_mode="ManyToOne",
                 rolling=False,
                 expand=True,
                 train_days=504,
                 test_days=132,
                 train_split=0.8,
                 valid_split=0.8,
                 **args):

        self._check_data_dims(features, labels)
        if sequence_mode not in ['ManyToOne', 'ManyToMany']:
            raise TypeError("sequence_mode can only be ManyToOne, ManyToMany")

        self._features = features
        self._labels = labels

        self.rolling = rolling
        self.expand = expand
        self.train_days = train_days
        self.test_days = test_days
        self.valid_split = valid_split
        self.train_split = train_split
        self.sequence = sequence
        self.sequence_size = sequence_size
        self.sequence_mode = sequence_mode
        self.args = args

        self.features, self.labels = self._combine_features_labels(
            features, labels)

        if sequence:
            self.features = self._seqHelper(self.features, sequence_size)
            if sequence_mode == 'ManyToMany':
                self.labels = self._seqHelper(self.labels, sequence_size)

        self.input_dims = (None, ) + self.features.shape[1:]
        self.output_dims = (None, ) + self.labels.shape[1:]
        self.nsamples = self.features.shape[0]
        self.feature_size = int(self.features.size / self.features.shape[0])
        self.label_size = int(self.labels.size / self.labels.shape[0])

        self.num_splits = int(
            (self.nsamples - self.train_days) / self.test_days) + 1

        if self.rolling:
            self._makeRollingSplit()
        else:
            self._makeTrainSplit()

    def _combine_features_labels(self, features, labels):
        if isinstance(features, dict):
            features = np.hstack(features.values())
        if isinstance(labels, dict):
            labels = np.hstack(labels.values())
        return features, labels

    def _check_data_dims(self, features, labels):
        features_dim = self._check_dims_helper(features, 'features')
        labels_dim = self._check_dims_helper(labels, 'labels')

        if features_dim != labels_dim:
            err = "the first dimension of labels and features must align!"
            err += ", Found features {} != labels {}"
            raise ValueError(err.format(features_dim, labels_dim))

    def _check_dims_helper(self, data, name):
        if isinstance(data, dict):
            data_dims = []
            for key, item in data.items():
                if not isinstance(item, np.ndarray):
                    raise TypeError(
                        "the [{}] {} is not a instance of numpy array".format(
                            key, name))
                data_dims.append(item.shape[0])
            if not data_dims.count(data_dims[0]) == len(data_dims):
                err = "{} first dimensions does not align! Found ".format(name)
                for key, item in a.items():
                    err += "{}:{}, ".format(key, item.shape[0])
                raise TypeError(err)
            return data_dims[0]
        elif isinstance(data, np.ndarray):
            return data.shape[0]
        else:
            raise TypeError(
                "the {} must be dictionary or numpy ndarray!".format(name))

    def _makeTrainSplit(self):
        valid_e = int(self.nsamples * self.train_split)
        train_e = int(valid_e * self.valid_split)
        self._buildTensor(self.features, self.labels, train_e, valid_e)

    def _makeRollingSplit(self):
        self.get_split(0)

    def _buildTensor(self, features, labels, train_e, valid_e):
        self.train = DataField(features[:train_e], labels[:train_e])
        self.valid = DataField(features[train_e:valid_e],
                               labels[train_e:valid_e])
        self.test = DataField(features[valid_e:], labels[valid_e:])

    def _get_split_index(self, i):
        start = 0 if self.expand else self.test_days * i
        end = self.test_days * (i + 1) + self.train_days
        return start, end

    def get_split(self, i):
        if i >= self.num_splits:
            raise RuntimeError(
                "Total number of splits is {}, you can't query split {}".
                format(self.num_splits, i + 1))

        start, end = self._get_split_index(i)
        valid_e = end - start - self.test_days
        train_e = int(valid_e * self.valid_split)
        features = self.features[start:end]
        labels = self.labels[start:end]
        self.split_features = features
        self.split_labels = labels
        self._buildTensor(features, labels, train_e, valid_e)
        return self

    def _seqHelper(self, x, window):
        samples = max(x.shape[0] - window + 1, 0)
        _, feature_size = x.shape
        tensor5D = np.zeros([samples, window, feature_size], dtype=np.float32)

        for t in range(samples):
            tensor5D[t] = x[t:t + window]
        padded = np.zeros([window - 1, window, feature_size], dtype=np.float32)
        tensor5D = np.vstack([padded, tensor5D]).astype(np.float32)
        return tensor5D


class IMVOL(DataSet):
    def __init__(self, ticker, ahead=1, otm=False, data_path='data/database/', **args):

        self.ticker = ticker
        self.ahead = ahead
        self.otm = otm
        self.start = args.get('start', None)
        self.end = args.get('end', None)
        self.data_path = data_path

        self.data = self._load_data(ticker)
        self.tensor = self._get_tensor(self.data)

        features = self.tensor[:-ahead]
        labels = self.tensor[ahead:]
        super(IMVOL, self).__init__(features, labels, **args)

    def _load_data(self, ticker):
        data = pd.read_csv(pjoin(f'{self.data_path}',f'{ticker}.csv'))
        data['d'] = data['delta'] - 100*data['cp_flag'].map({'C': 0, 'P': 1})
        if self.otm:
            data = data[data['d'].abs() <= 50].copy()
        if self.start is not None:
            data = data[data['date'] >= str(self.start)].copy()
        if self.end is not None:
            data = data[data['date'] <= str(self.end)].copy()

        return data

    def _get_tensor(self, data):
        num_dates = data.date.unique().shape[0]
        num_delta = data.d.unique().shape[0]
        num_matur = data.maturity.unique().shape[0]
        tensor = data.imvol.astype(np.float32).values.reshape(
            [num_dates, num_matur * num_delta])
        return tensor

    @property
    def options_type(self):
        df = self.data.iloc[:self.feature_size][[
            'maturity', 'cp_flag', 'delta']]
        cols = pd.MultiIndex.from_frame(df)
        return cols


if __name__ == "__main__":
    imvol = IMVOL(
        'AAPL',
        rolling=False,
        otm=False,
        train_split=0.75,
        start=2010)
