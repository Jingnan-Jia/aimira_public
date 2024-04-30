# -*- coding: utf-8 -*-
# @Time    : 4/5/22 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
import os

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的文件夹路径
current_folder_path = os.path.dirname(current_file_path)
# 获取当前文件夹的父文件夹路径
parent_folder_path = os.path.dirname(current_folder_path)

sys.path.append(current_folder_path)
sys.path.append(parent_folder_path)


import random
import threading
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from medutils import medutils
from medutils.medutils import count_parameters
# from mlflow import log_metric, log_metrics, log_param, log_params
from mlflow.tracking import MlflowClient
from monai.utils import set_determinism
from typing import List, Sequence
from argparse import Namespace
import functools
import thop
import os
import copy
import pandas as pd
from glob import glob
from torchsummary import summary
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, average_precision_score

from aimira.modules.compute_metrics import icc, metrics
from aimira.modules.datasets import all_loaders
from aimira.modules.loss import get_loss
from aimira.modules.networks import get_net_3d
from aimira.modules.path import aimira_Path
from aimira.modules.set_args import get_args
from aimira.modules.tools import record_1st, dec_record_cgpu, retrive_run, try_func, int2str, txtprocess, log_all_metrics, process_dict

args = get_args()
global_lock = threading.Lock()


# log_metric = try_func(log_metric)
# log_metrics = try_func(log_metrics)


def log_metric(k, v, idx):
    print(k, ' : ', v, ' at ', idx)
    

def log_param(k, v):
    print(k, ' : ', v)
    
def log_params(dt):
    print(dt)
    
class Run_aimira:
    """A class which has its dataloader and step_iteration. It is like Lighting. 
    """

    def __init__(self, args: Namespace, dataloader_flag=True):
        self.args = args
        
        self.device = torch.device("cuda")  # 'cuda'
        self.target = [i.lstrip() for i in args.target.split('-')]
        self.mypath = aimira_Path(args.id, check_id_dir=False, space=args.ct_sp)
        self.net = get_net_3d(name=args.net, nb_cls=len(self.target))  # receive ct and pcd as input
        print('net:', self.net)

        if dataloader_flag:
            self.data_dt_all = all_loaders(self.mypath.data_dir, self.mypath.label_fpath, args, nb=1000)
            
        self.fold = args.fold
        self.flops_done = False


        net_parameters = count_parameters(self.net)
        net_parameters = str(round(net_parameters / 1e6, 2))
        log_param('net_parameters_M', net_parameters)

        self.loss_fun = get_loss(args.loss)
        
        self.opt = torch.optim.Adam( self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        validMAEEpoch_AllBest = 1000
        args.pretrained_id = str(args.pretrained_id)

        self.BestMetricDt = {'trainLossEpochBest': 1000,
                             # 'trainnoaugLossEpochBest': 1000,
                             'validLossEpochBest': 1000,
                             'testLossEpochBest': 1000,

                             'trainMAEEpoch_AllBest': 1000,
                             # 'trainnoaugMAEEpoch_AllBest': 1000,
                             'validMAEEpoch_AllBest': validMAEEpoch_AllBest,
                             'testMAEEpoch_AllBest': 1000,
                             }
        self.net = self.net.to(self.device)

    def step(self, mode, epoch_idx, save_pred=False):
        dataloader_all = self.data_dt_all[mode]
        loss_fun_mae = nn.L1Loss()
        scaler = torch.cuda.amp.GradScaler()
        print(mode + "ing ......")
        if mode == 'train':
            self.net.train()
        else:
            self.net.eval()

        # t0 = time.time()
        monitor_dt = {'loss_accu': 0,
                      'mae_accu_ls': [0 for _ in self.target],
                      'mae_accu_all': 0}
        # loss_accu = 0
        # mae_accu_ls = [0 for _ in self.target]
        # mae_accu_all = 0 
        for batch_dt_ori in dataloader_all:
            torch.cuda.empty_cache()  # avoid memory leak

            # label
            batch_y = batch_dt_ori['label'].to(self.device)
            batch_x = batch_dt_ori['input'].to(self.device)
            
            if not self.flops_done:  # only calculate macs and params once
                macs, params = thop.profile(self.net, inputs=(batch_x, ))
                self.flops_done = True
                log_param('macs_G', str(round(macs/1e9, 2)))
                log_param('net_params_M', str(round(params/1e6, 2)))
                
            tt0 = time.time()
            with torch.cuda.amp.autocast():
                if mode != 'train' or save_pred:  # save pred for inference
                    with torch.no_grad():
                        pred = self.net(batch_x)
                else:
                    pred = self.net(batch_x)
                print('pred', pred)
                
                tt1 = time.time()
                print(f'time forward: , {tt1-tt0: .2f}')

                if save_pred:
                    for k in batch_dt_ori:
                        if 'pat_id' in k:
                            key_pat_id = k    
                            break                        
                   
                    head = ['pat_id']
                    head.extend(self.target)

                    batch_pat_id = batch_dt_ori[key_pat_id].cpu( ).detach().numpy()  # shape (N,1)
                    batch_pat_id = int2str(batch_pat_id)  # shape (N,1)

                    batch_y_np = batch_y.cpu().detach().numpy()  # shape (N, out_nb)
                    pred_np = pred.cpu().detach().numpy()  # shape (N, out_nb)
                    saved_label = np.hstack((batch_pat_id, batch_y_np))
                    saved_pred = np.hstack((batch_pat_id, pred_np))

                    pred_fpath = self.mypath.save_pred_fpath(mode)
                    label_fpath = self.mypath.save_label_fpath(mode)
                        
                    medutils.appendrows_to(label_fpath, saved_label, head=head)
                    medutils.appendrows_to(pred_fpath, saved_pred, head=head)
    
                loss = self.loss_fun(pred, batch_y)          
                
                with torch.no_grad():                
                    mae_ls = [loss_fun_mae(pred[:, i], batch_y[:, i]).item() for i in range(len(self.target))]
                    mae_all = loss_fun_mae(pred, batch_y).item()
                    
            if mode == 'train' and save_pred is not True:  # update gradients only when training
                self.opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
            tt2 = time.time()
            print(f'time backward: , {tt2-tt1: .2f}')
            loss_cpu = loss.item()
            print('loss:', loss_cpu)

            monitor_dt['loss_accu'] += loss_cpu
            for i, mae in enumerate(mae_ls):
                monitor_dt['mae_accu_ls'][i] += mae
            monitor_dt['mae_accu_all'] += mae_all


        data_len = len(dataloader_all)
        log_metric(mode+'LossEpoch', monitor_dt['loss_accu']/data_len, epoch_idx)
        log_metric(mode+'MAEEpoch_All', monitor_dt['mae_accu_all'] / data_len, epoch_idx)
        for t, i in zip(self.target, monitor_dt['mae_accu_ls']):
            log_metric(mode + 'MAEEpoch_' + t, i / data_len, epoch_idx)

        self.BestMetricDt[mode + 'LossEpochBest'] = min( self.BestMetricDt[mode+'LossEpochBest'], monitor_dt['loss_accu']/data_len)
        tmp = self.BestMetricDt[mode+'MAEEpoch_AllBest']
        self.BestMetricDt[mode + 'MAEEpoch_AllBest'] = min( self.BestMetricDt[mode+'MAEEpoch_AllBest'], monitor_dt['mae_accu_all']/data_len)

        log_metric(mode+'LossEpochBest', self.BestMetricDt[mode + 'LossEpochBest'], epoch_idx)
        log_metric(mode+'MAEEpoch_AllBest', self.BestMetricDt[mode + 'MAEEpoch_AllBest'], epoch_idx)

        if self.BestMetricDt[mode+'MAEEpoch_AllBest'] == monitor_dt['mae_accu_all']/data_len:
            for t, i in zip(self.target,  monitor_dt['mae_accu_ls']):
                log_metric(mode + 'MAEEpoch_' + t + 'Best', i / data_len, epoch_idx)

            if mode == 'valid':
                print(
                    f"Current mae is {self.BestMetricDt[mode+'MAEEpoch_AllBest']}, better than the previous mae: {tmp}, save model to {self.mypath.model_fpath}.")
                ckpt = {'model': self.net.state_dict(),
                        'metric_name': mode+'MAEEpoch_AllBest',
                        'current_metric_value': self.BestMetricDt[mode+'MAEEpoch_AllBest']}
                torch.save(ckpt, self.mypath.model_fpath)

def remove_ops(state_dict):
    new_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
    return new_state_dict
    


# def metrics_cls(label_fpath, pred_fpath, threshold_baseline=-3.07)
#     # 划分四个区域的数据点
#     TN_data = (y_pred_test > threshold_baseline) & (y_test > threshold_baseline)
#     FP_data = (y_pred_test < threshold_baseline) & (y_test > threshold_baseline)
#     FN_data = (y_pred_test > threshold_baseline) & (y_test < threshold_baseline)
#     TP_data = (y_pred_test < threshold_baseline) & (y_test < threshold_baseline)

#     # 统计每个区域的数据点个数
#     TN = sum(TN_data)
#     FN = sum(FN_data)
#     FP = sum(FP_data)
#     TP = sum(TP_data)
    
#     sensitivity = TP / (TP + FN)
#     precision = TP / (TP + FP)

#     f1 = 2 * TP / (2 * TP + FP + FN)
#     specificity = TN / (FP + TN)
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     # 计算 ROC 曲线和 AUC 值
#     fpr, tpr, thresholds = roc_curve(test_data[diff_bin_name], y_pred_test)
#     roc_auc = auc(fpr, tpr)
#     pr_auc = average_precision_score(test_data[diff_bin_name], y_pred_test)
#     metrics_dt = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': sensitivity,
#         'specificity': specificity,
#         'f1': f1,
#         'roc_auc': roc_auc,
#         'pr_auc': pr_auc  # Area Under the Precision-Recall curve
#         }
#     return metrics_dt
            
            
@dec_record_cgpu(args.outfile)
def run(args: Namespace, all_data_dt: dict):
    """
    Run the whole  experiment using this args.
    """
    myrun = Run_aimira(args)

    modes = ['valid', 'test', 'train'] if args.mode != 'infer' else ['valid', 'test']
    if args.mode == 'infer':
        for mode in modes:
            for i in range(1):
                myrun.step(mode,  0,  save_pred=True)
    else:  # 'train' or 'continue_train'
        for i in range(args.epochs):  # 20000 epochs
            myrun.step('train', i)
            if i % args.valid_period == 0:  # run the validation             
                myrun.step('valid',  i)
                myrun.step('test',  i)
            if i == args.epochs - 1:  # load best model and do inference
                print('start inference')
                if os.path.exists(myrun.mypath.model_fpath):  # load the best model
                    ckpt = torch.load(myrun.mypath.model_fpath, map_location=myrun.device)
                    if isinstance(ckpt, dict) and 'model' in ckpt:
                        model = ckpt['model']
                    else:
                        model = ckpt
                    # model_fpath need to exist
                    # model = remove_ops(model)
                    myrun.net.load_state_dict(model, strict=False)
                    print(f"load net from {myrun.mypath.model_fpath}")
                else:
                    print(
                        f"no model found at {myrun.mypath.model_fpath}, let me save the current model to this lace")
                    ckpt = {'model': myrun.net.state_dict()}
                    torch.save(ckpt, myrun.mypath.model_fpath)
                for mode in modes:
                    myrun.step(mode, i, save_pred=True)

        mypath = aimira_Path(args.id, check_id_dir=False, space=args.ct_sp)
        label_ls = [mypath.save_label_fpath(mode) for mode in modes]
        pred_ls = [mypath.save_pred_fpath(mode) for mode in modes]

        for pred_fpath, label_fpath in zip(pred_ls, label_ls):
            r_p_value = metrics(pred_fpath, label_fpath, ignore_1st_column=True)
            r_p_value = txtprocess(r_p_value)
            log_params(r_p_value)
            print('r_p_value:', r_p_value)

            icc_value = icc(label_fpath, pred_fpath, ignore_1st_column=True)
            log_params(icc_value)
            print('icc:', icc_value)
            if os.path.exists(os.path.dirname(pred_fpath) + '/valid_scatter.png'):
                os.rename(os.path.dirname(pred_fpath) + '/valid_scatter.png', os.path.dirname(pred_fpath) + f'/valid_scatter_{i}.png')
                
            if os.path.exists(os.path.dirname(pred_fpath) + '/test_scatter.png'):
                os.rename(os.path.dirname(pred_fpath) + '/test_scatter.png', os.path.dirname(pred_fpath) + f'/test_scatter_{i}.png')
                
            # add classification metrics
            # metrics_cls_dt = metrics_cls(label_fpath, pred_fpath, threshold_baseline=-3.07)


    print('Finish all things!')

def fixed_seed(SEED):
    set_determinism(SEED)  # set seed for this run
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
        

def main():
    fixed_seed(SEED=4)

    # mlflow.set_tracking_uri("http://nodelogin02:5000")
    # experiment = mlflow.set_experiment('AIMIRA')
    
    RECORD_FPATH = f"{Path(__file__).absolute().parent}/results/record.log"
    # write super parameters from set_args.py to record file.
    id = record_1st(RECORD_FPATH)
 
    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        args.id = id  # do not need to pass id seperately to the latter function
        args.fold = 'all'
        tmp_args_dt = vars(args)
        current_id = id
        # log_params(tmp_args_dt)

        all_folds_id_ls = []
        for fold in [4,3,2,1]:
            id = record_1st(RECORD_FPATH)
            all_folds_id_ls.append(id)
            with mlflow.start_run(run_name=str(id) + '_fold_' + str(fold), tags={"mlflow.note.content": f"fold: {fold}"}, nested=True):
                args.fold = fold
                args.id = id  
                tmp_args_dt = vars(args)
                log_params(tmp_args_dt)
                run(args)
                
        # log_all_metrics(all_folds_id_ls, current_id, experiment)

if __name__ == "__main__":
    main()
