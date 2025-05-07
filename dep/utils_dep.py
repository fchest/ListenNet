import os
import pandas as pd
import numpy as np



def getData(args, sub_id, dataset="DTU"):

    if dataset == 'DTU':
        onedata, onelabel = get_DTU_data(args, sub_id)
        train_eeg, test_eeg, train_label, test_label = sliding_window(args, onedata, onelabel, sub_id)
        train_eeg = train_eeg.transpose(0,2,1) 
        test_eeg = test_eeg.transpose(0,2,1) 

    
    elif dataset == 'KUL':
        onedata, onelabel = get_KUL_data(args, sub_id)
        eeg_data = np.vstack(onedata)
        eeg_data = np.array(eeg_data)
        onedata = eeg_data.reshape([args.trail_number, -1, args.eeg_channel])
        train_eeg, test_eeg, train_label, test_label = sliding_window(args, onedata, onelabel, sub_id)
        train_eeg = train_eeg.transpose(0,2,1) 
        test_eeg = test_eeg.transpose(0,2,1) 
    
    elif dataset == 'AVED':
        onedata, onelabel = get_AVED_data(args, sub_id)
        eeg_data = np.vstack(onedata)
        eeg_data = np.array(eeg_data)
        onedata = eeg_data.reshape([args.trail_number, -1, args.eeg_channel])
        train_eeg, test_eeg, train_label, test_label = sliding_window(args, onedata, onelabel, sub_id)
        train_eeg = train_eeg.transpose(0,2,1) 
        test_eeg = test_eeg.transpose(0,2,1) 
    return train_eeg, test_eeg, train_label, test_label
  

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
    sub_path = args.data_path  + "s" + str(sub_id) + "_data.npy"
    sublabel_path = args.label_path  + "s" +  str(sub_id) + "_label.npy"
    sub_data = np.load(sub_path)
    sub_label = np.load(sublabel_path)
    print('Finish get the data from: ', args.data_path + str(sub_id))
    return sub_data, sub_label


# ========================= AVED data =====================================
def get_AVED_data(args, sub_id):
    '''description: get all the data from one dataset
    param {type} 
    return {type}:
        data: list  16(subjects), each data is x * 
        label: '''
    filename = args.data_path + "sub" +  str(sub_id) + ".csv"
    data_pf = pd.read_csv(filename, header=None)
    alldata = data_pf.iloc[:, :].values 
    all_label = np.array([[1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2]])
    print('Finish get the data from: ', args.data_path + str(sub_id))
    return alldata, all_label


# ========================= window split =====================================
def sliding_window(args ,eeg_datas, labels, sub_id):
    stride = int(args.window_lap)

    train_eeg = []
    train_label = []

    test_eeg = []
    test_label =[]

    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        
        for i in range(0, eeg.shape[0] - args.win_len + 1, stride):
            window = eeg[i:i+args.win_len, :]
            windows.append(window)
            new_label.append(label-1)
        
        train_eeg.append(np.array(windows)[:int(len(windows)*0.9)])
        test_eeg.append(np.array(windows)[int(len(windows)*0.9):])
        train_label.append(np.array(new_label)[:int(len(windows)*0.9)])
        test_label.append(np.array(new_label)[int(len(windows)*0.9):])

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, args.win_len, args.eeg_channel)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, args.win_len, args.eeg_channel)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg,test_eeg, train_label,test_label



