import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.linalg import sqrtm

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


def getData(args,  dataset="DTU"):
    seq_alldata = []
    alllabel = []
    alll_ckabel = []

    if dataset == 'DTU':
        for id in range(1, args.subject_number + 1):
            onedata, onelabel = get_DTU_data(args,id)
            onedata, onelabel, check_label = sliding_window(args, onedata, onelabel, id)
            onedata = onedata.transpose(0,2,1)
            seq_alldata.append(onedata)
            alllabel.append(onelabel)
            alll_ckabel.append(check_label)

    elif dataset == 'KUL':
        for id in range(1, args.subject_number + 1):
            onedata, onelabel = get_KUL_data(args, id)
            onedata, onelabel, check_label = sliding_window(args, onedata, onelabel, id) 
            onedata = onedata.transpose(0,2,1)
            seq_alldata.append(onedata)
            alllabel.append(onelabel)
            alll_ckabel.append(check_label)

    elif dataset == 'AVED':
        for id in range(1, args.subject_number + 1):
            onedata, onelabel = get_AHU20_data(args, id)
            onedata = onedata.reshape([args.trail_number, -1, args.eeg_channel])
            onedata, onelabel, check_label = sliding_window(args, onedata, onelabel, id) 
            onedata = onedata.transpose(0,2,1)
            seq_alldata.append(onedata)
            alllabel.append(onelabel)
            alll_ckabel.append(check_label)
    return seq_alldata,  alllabel, alll_ckabel
  

# ========================= kul data =====================================
def get_KUL_data(args, sub_id):
    '''description: get all the data from one dataset
    param {type} 
    return {type}:
        data: list  16(subjects), each data is x * 
        label: '''
    alldata = []
    all_data_dir = os.listdir(args.data_path)
    all_data_dir.sort( )
    #for s_data in range(len(all_data_dir)):
    sub_path = args.data_path  + str(sub_id)
    sublabel_path = args.label_path  + "S" +  str(sub_id) + "No.csv"
    sub_data_dir = os.listdir(sub_path)
    sub_data_dir.sort()
    for k in range(len(sub_data_dir)):
        filename = sub_path + '/' + sub_data_dir[k]
        data_pf = pd.read_csv(filename, header=None)
        eeg_data = data_pf.iloc[:, 2:].values # （46080，64）
        
        alldata.append(eeg_data)
    label_pf = pd.read_csv(sublabel_path, header=None)
    all_label = label_pf.iloc[1:, 0].values 
    print('Finish get the data from: ', args.data_path + str(sub_id))
    return alldata, all_label

# ========================= dtu data =====================================
def get_DTU_data(args, sub_id):
    '''description: get all the data from one dataset
    param {type} 
    return {type}:
        data: list  16(subjects), each data is x * 
        label: '''
    alldata = []
    sub_path = args.data_path  + "s" + str(sub_id) + "_data.npy"
    sublabel_path = args.label_path  + "s" +  str(sub_id) + "_label.npy"
    sub_data = np.load(sub_path)
    sub_label = np.load(sublabel_path)
    print('Finish get the data from: ', args.data_path + str(sub_id))
    return sub_data, sub_label

# ========================= ahu data ====================================
def get_AHU20_data(args, test_id):
    '''description: get all the data from one dataset
    param {type} 
    return {type}:
        data: list  16(subjects), each data is x * 
        label: '''

    #for s_data in range(len(all_data_dir)):
    filename = args.data_path  + "sub" +  str(test_id) + ".csv"
    data_pf = pd.read_csv(filename, header=None)
    eeg_data = data_pf.iloc[:, :].values # （46080，64）
    all_label = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2]
    print('Finish get the data from: ', args.data_path + str(test_id))
    return eeg_data, all_label


# ========================= window split =====================================
def sliding_window(args ,eeg_datas, labels, sub_id):
    stride = int(args.window_lap)
    train_eeg = []
    train_label = []
    train_cheak_label = []

    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        cheak_label = []
        for i in range(0, eeg.shape[0] - args.win_len + 1, stride):
            window = eeg[i:i+args.win_len, :]
            windows.append(window)
            new_label.append(label-1)
            cheak_label.append(sub_id)
        train_eeg.append(np.array(windows))
        train_label.append(np.array(new_label))
        train_cheak_label.append(np.array(cheak_label))

    eeg = np.stack(train_eeg, axis=0).reshape(-1, args.win_len, args.eeg_channel)
    label = np.stack(train_label, axis=0).reshape(-1, 1)
    train_cheak_label = np.stack(train_cheak_label, axis=0).reshape(-1, 1)
    # train_cheak_label = np.stack(train_cheak_label, axis=0).reshape(-1, 3)
    return eeg, label, train_cheak_label


