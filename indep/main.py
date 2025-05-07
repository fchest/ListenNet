from __future__ import division
from __future__ import print_function
import os
import math

import numpy as np
import pandas as pd

import logging
import argparse
import copy

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from typing import Optional
import torchinfo
import matplotlib.pyplot as plt

from collections import OrderedDict
from importlib import reload
from ptflops import get_model_complexity_info

from utils_indep import *
from function import *
from model import *


# writer = SummaryWriter()
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device is", device)

def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_logger(name, log_path, length):
    reload(logging)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = makePath(log_path) + "/Train"+str(length)+"s_sub" + str(name) + ".log"
    with open(logfile, 'w') as f:
        f.write("epoch, acc\n")

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    if log_path == "./result/test":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    return logger

class StepwiseLR_GRL: 
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.001,
                 gamma: Optional[float] = 0.01, decay_rate: Optional[float] = 0.1,max_iter: Optional[float] = 100):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter=max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
        if lr <= 1e-8:
            lr = 1e-8
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']
        self.iter_num += 1

class Trynetwork():
    def __init__(self, model, train_loader, valid_loader, test_loader, batch_size, lr, weight_decay, model_constraint):
        self.model = model
        self.datasets = OrderedDict((("train", train_loader), ("valid", valid_loader), ("test", test_loader)))
        if valid_loader is None:
            self.datasets.pop("valid")
        if test_loader is None:
            self.datasets.pop("test")
        self.best_test = 0
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_constraint = model_constraint
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 35], gamma=0.5)
        self.scheduler_down = StepwiseLR_GRL(self.optimizer, init_lr= args.lr, gamma= 10, decay_rate=args.lr_decayrate,max_iter=args.max_epoch)
        self.criterion = nn.CrossEntropyLoss()
        
        # initialize epoch dataFrame instead of loss and acc for train and test
        self.val_df = pd.DataFrame()  
        self.train_df = pd.DataFrame()  
        self.epoch_df = pd.DataFrame()  
    

    def __getModel__(self):
        return self.model

    def train_step(self, args, epoch):
        self.model.train()
        train_dicts_per_epoch = OrderedDict()
        Batch_size, Cls_loss, Train_acc = [], [], []
        for i_batch, batch_data  in enumerate(self.datasets['train']):
            seq_data,  train_label = batch_data
            train_label = train_label.squeeze(-1)
            seq_data,  train_label = seq_data.cuda().float(), train_label.cuda().long()
            _, source_softmax = self.model(seq_data)
            nll_loss = self.criterion(source_softmax, train_label)  

            Batch_size.append(len(train_label))
            _, predicted = torch.max(source_softmax.data, 1)
            batch_acc = np.equal(predicted.cpu().detach().numpy(), train_label.cpu().detach().numpy()).sum() / len(
                train_label)
            # Forward pass
            Train_acc.append(batch_acc)
            cls_loss_np = nll_loss.cpu().detach().numpy()
            Cls_loss.append(cls_loss_np)

            # Backward and optimize
            self.optimizer.zero_grad()
            nll_loss.backward()
            self.optimizer.step()

            if self.model_constraint is not None:
                self.model_constraint.apply(self.model)

        epoch_acc = sum(Train_acc) / len(Train_acc) * 100
        epoch_loss = sum(Cls_loss) / len(Cls_loss)

        cls_loss = {'train_loss': epoch_loss}
        train_acc = {'train_acc': epoch_acc}
        train_dicts_per_epoch.update(cls_loss)
        train_dicts_per_epoch.update(train_acc)
        train_dicts_per_epoch = {k: [v] for k, v in train_dicts_per_epoch.items()}
        self.train_df = pd.concat([self.train_df, pd.DataFrame(train_dicts_per_epoch)], ignore_index=True)
        self.train_df = self.train_df[list(train_dicts_per_epoch.keys())]  # 让epoch_df中的顺序和row_dict中的一致
        return epoch_loss , epoch_acc
    
    def test_batch(self, seq_input, target):
        self.model.eval()
        with torch.no_grad():
            val_seqinput = seq_input.cuda().float()
            val_target = target.cuda().long()
            _, val_fc1 = self.model(val_seqinput)
            loss = self.criterion(val_fc1, val_target)
            _, preds = torch.max(val_fc1.data, 1)  
            preds = preds.cpu().detach().numpy()
            loss = loss.cpu().detach().numpy()
        return preds, loss

    def evaluate_step(self,  datasets):
        datasets['test'] = self.datasets['test']
        result_dicts_per_monitor = OrderedDict()  
        with torch.no_grad():
            for setname in datasets:  
                assert setname != 'train', 'dataset without train set'
                assert setname in ["test"]  
            Batch_size, Epochs_loss, Epochs_acc = [], [], []
            for i_batch, batch_data in enumerate(self.datasets['test']):
                seq_input,  target = batch_data
                target = target.squeeze(-1)
                pred, loss = self.test_batch(seq_input,  target)  
                Epochs_loss.append(loss)
                Batch_size.append(len(target))
                Epochs_acc.append(np.equal(pred, target.numpy()).sum())  
        epoch_acc = sum(Epochs_acc) / sum(Batch_size) * 100
        epoch_loss = sum(Epochs_loss) / len(Epochs_loss)
        key_loss = setname + '_loss'
        key_acc = setname + '_acc'
        loss = {key_loss: epoch_loss}
        acc = {key_acc: epoch_acc}
        result_dicts_per_monitor.update(loss)
        result_dicts_per_monitor.update(acc)
        result_dicts_per_monitor = {k: [v] for k, v in result_dicts_per_monitor.items()}
        self.val_df = pd.concat([self.val_df, pd.DataFrame(result_dicts_per_monitor)], ignore_index=True)
        self.val_df = self.val_df[list(result_dicts_per_monitor.keys())]  # 让epoch_df中的顺序和row_dict中的一致
        return epoch_loss, epoch_acc


    def train(self, args, testsub_id):
        # 打印模型参数量

        torchinfo.summary(self.model)

        '''macs, params = get_model_complexity_info(self.model, 
                                                (1, args.eeg_channel, args.win_len), 
                                                 print_per_layer_stat=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))'''

        testsub_name = 'S' + str(testsub_id)
        
        best_epoch = 0
        best_acc = 0
        for epoch in range(1, args.max_epoch + 1):
            train_loss,train_acc = self.train_step(args, epoch)
            val_loss, val_acc = self.evaluate_step({})
            # self.scheduler_down.step()
            print('TestSub:', testsub_name,
                  'Epoch {:2d} Finsh | Now_lr {:2.4f}/{:2.4f}|Train Loss {:2.4f} | Valid Loss {:2.4f} | Train Acc {:5.4f}| Valid Acc {:5.4f}'.format(epoch,
                                                                                                                                                self.optimizer.param_groups[0]["lr"], args.lr,
                                                                                                                                                train_loss,
                                                                                                                                                val_loss,
                                                                                                                                                train_acc,
                                                                                                                                                val_acc))
            if val_acc > best_acc:
                save_model(args, testsub_name, best_acc, val_acc, self.model, epoch)
                best_acc = val_acc
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale > args.patience:
                    print(f"Early stopping at epoch {epoch}!")
                    break
            self.epoch_df = pd.concat([self.train_df, self.val_df], axis=1)
        model = load_model(args.model_save_path, testsub_name)
        self.model = model
        test_loss, model_test_acc = self.evaluate_step({})
        print("-" * 50)
        print('Test_Subject :{:s} |Best epoch:{:d} | Test Loss:{:2.4f} | Savemodel Acc {:2.4f}'.format(testsub_name,
                                                                                                    best_epoch,
                                                                                                    test_loss,
                                                                                                    model_test_acc))
        print("-" * 50)
        return best_epoch, model_test_acc

def cross_subject(args, test_id, sub_ids, alldata, alllabel):
    ## LOSO
    if test_id in sub_ids:
        test_index = sub_ids.index(test_id)
    tempt_data,  tempt_label = copy.deepcopy(alldata), copy.deepcopy(alllabel)
    test_data,  test_label = copy.deepcopy(tempt_data.pop(test_index)), copy.deepcopy(tempt_label.pop(test_index)) 
    train_data, train_label = copy.deepcopy(tempt_data),  copy.deepcopy(tempt_label)
    
    
    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    
    train_data =  preprocessEA(train_data)
    test_data =  preprocessEA(test_data)
    
    train_data = np.expand_dims(train_data, axis=1)
    test_data = np.expand_dims(test_data, axis=1)
    

    train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                                  batch_size=args.batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                                 batch_size=args.batch_size, drop_last=True)
    valid_loader  = None
    print(f"train_seqdata_shape{train_data.shape},test_seqdata_shape{test_data.shape}")
    
    #####################################################################################
    #2.define model
    #####################################################################################
    model_constraint = MaxNormDefaultConstraint()
    MyNet =Trynetwork(
        model = ListenNet(num_classes=args.class_num, chans=args.eeg_channel, samples=args.win_len,
                   depth = 16,
                   kernel = 8,
                   avepool = args.fs// 10,
                   ).cuda(),
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        test_loader =test_loader,
        batch_size = args.batch_size, 
        lr = args.lr,
        weight_decay = args.weight_decay,
        model_constraint = model_constraint)
    onesub_test_epoch, onesub_test_acc  = MyNet.train(args, test_id)
    return onesub_test_epoch, onesub_test_acc 

    
if __name__ == '__main__':
    # Training settings
    args = argparse.ArgumentParser()
    args.seed = 42

    # data
    args.dataset = 'KUL'
    options = {'KUL':[16, 64, 8, 128, "/media/jiangwencong/yangxiaoke/Dataset/KUL/pre_data/", "/media/jiangwencong/yangxiaoke/Dataset/KUL/label/"], 
               'DTU':[18, 64, 60, 128, "/media/jiangwencong/yangxiaoke/Dataset/DTU/128/data/", "/media/jiangwencong/yangxiaoke/Dataset/DTU/128/label/"], 
               #'AVED':[10 ,32, 16, 128, "/media/jiangwencong/yangxiaoke/Dataset/AHU_20/audio-only/","/media/jiangwencong/yangxiaoke/Dataset/AHU_20/audio-only/"]}
               'AVED':[10 ,32, 16, 128, "/media/jiangwencong/yangxiaoke/Dataset/AHU_20/audio-video/","/media/jiangwencong/yangxiaoke/Dataset/AHU_20/audio-video/"]}

    args.subject_number = options[args.dataset][0]
    args.eeg_channel = options[args.dataset][1]
    args.trail_number = options[args.dataset][2]
    args.fs  = options[args.dataset][3]
    args.data_path = options[args.dataset][4]
    args.label_path = options[args.dataset][5]

    args.win_time = 1
    args.win_len = math.ceil(args.fs * args.win_time)
    args.overlap = 0.5
    args.window_lap = args.win_len * (1 - args.overlap)
    
    # basic info of the model
    args.class_num = 2
    args.batch_size = 128
    args.lr = 1e-3
    args.lam = 0.2
    args.lr_decayrate = 0.5
    args.weight_decay = 3e-4
    args.max_epoch = 100
    args.patience = 20
    args.log_interval = 10
    

    # save to
    filename = "./writer/Subject_independent_AAD/%s/" % f"{args.dataset}"
    args.model_save_path = f'{filename}/{args.win_time}s/savemodel/'
    makePath(args.model_save_path)
    args.log_path = f'{filename}{args.win_time}s/result/' 
    makePath(args.log_path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
  
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)
   

    sub_ids =  list(range(1, args.subject_number+1)) # KUL 16
    # load win data 和 label
    alldata,  alllabel, alll_ckabel = getData(args, args.dataset)

    for test_id in sub_ids:
        print('Test sub_id: ', test_id) 
        logger = get_logger(test_id, args.log_path, args.win_time)
        best_epoch, test_acc = cross_subject(args, test_id, sub_ids, alldata,  alllabel)
        logger.info(f"{best_epoch}, {test_acc}")
        

