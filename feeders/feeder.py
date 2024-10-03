import numpy as np
import pickle
import torch
import torch.utils.data


import feeders.tools as tools

class Feeder(torch.utils.data.Dataset):

    def __init__(self, data_path, label_path, normalization, random_shift, random_choose, random_move, window_size=-1):

        self.data_path = data_path
        self.label_path = label_path
        self.normalization = normalization
        self.random_shift = random_shift
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data()

        if self.normalization:
            self.get_mean_map()


    def load_data(self):
        if self.label_path.endswith('.pkl'):
            try:
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f, encoding='latin1')
            except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
                raise ValueError(f"Error loading pickle file: {e}")
        else:
            raise ValueError("Unsupported file format. Only .pkl files are supported.")

        self.data = np.load(self.data_path)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data = np.array(self.data[index])
        label = self.label[index]
        name = self.sample_name[index]

        if self.normalization:
            data = (data - self.mean_map) / self.std_map
        if self.random_shift:
            data = tools.random_shift(data)
        if self.random_choose:
            data = tools.random_choose(data, self.window_size)
        elif self.window_size > 0:
            data = tools.auto_pading(data, self.window_size)
        if self.random_move:
            data = tools.random_move(data)

        return data, label, name

