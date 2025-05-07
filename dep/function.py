from scipy.linalg import sqrtm
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import torch

def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

class CustomDatasets(Dataset):
    # initialization: data and label
    def __init__(self, seq_data, event_data):
        self.seq_data = seq_data
        self.label = event_data

    # get the size of data
    def __len__(self):
        return len(self.label)

    # get the data and label
    def __getitem__(self, index):
        seq_data = torch.Tensor(self.seq_data[index])
        label = torch.LongTensor(self.label[index])
        return seq_data, label

# ========================= data aglin =====================================
def data_norm(data):
    """
    :param data:   ndarray ,shape[N,channel,samples]
    :return:
    """
    data_copy = np.copy(data)
    for i in range(len(data)):
        data_copy[i] = data_copy[i] / np.max(abs(data[i]))
    return data_copy

#  Euclidean Alignment
def preprocess_ea(data):
    R_bar = np.einsum('ijk,ilk->jl', data, data)
    R_bar_mean = R_bar / len(data)
    inv_sqrt_R_bar_mean = np.linalg.inv(sqrtm(R_bar_mean))
    data = np.einsum('ij,kjm->kim', inv_sqrt_R_bar_mean, data)
    return data

#  Trial normalization
def prepare_data(data):
    # [-1,1]
    data_preprocss = data_norm(data)
    data_ea = preprocess_ea(data_preprocss)
    return data_ea

def preprocessEA(data): 
    data_preprocessed = prepare_data(data) 
    return data_preprocessed

# ========================= model =====================================
def save_model(args, subject_name, best_valid, val_loss, model, epoch):
    print(f'Validation loss decreased ({best_valid:.6f} --> {val_loss:.6f}) in epoch ({epoch}).  Saving model ...')
    model_save_path = args.model_save_path + subject_name + ".pt"
    makePath(args.model_save_path)
    torch.save(model, model_save_path)

def load_model(path, subject_name):
    '''Saves model when validation acc increase.''' 
    model_save_path = path + subject_name + ".pt"
    model = torch.load(model_save_path) 
    return model 