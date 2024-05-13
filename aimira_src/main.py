import os
import random
import threading
from pathlib import Path
from medutils.medutils import save_itk

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
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from aimira_modules.main_func import train, pretrained, predict
# from models.model import scratch_nn
from aimira_modules.model import ModelClip
from aimira_generators.aimira_generator import AIMIRA_generator

from aimira_src.aimira_modules.compute_metrics import icc, metrics
from aimira_src.aimira_modules.path import aimira_Path
from aimira_src.aimira_modules.set_args import get_args
from aimira_src.aimira_modules.tools import record_1st, dec_record_cgpu, retrive_run, try_func, int2str, txtprocess, log_all_metrics, process_dict

args = get_args()
global_lock = threading.Lock()

def log_metric(k, v, idx):
    print(k, ' : ', v, ' at ', idx)
    

def log_param(k, v):
    print(k, ' : ', v)
    
def log_params(dt):
    print(dt)
    
# basic workflow of pytorch:
# 1. define dataset:
#       dataset = Dataset(...) torch.utils.data.Dataset object -- including the data and label -- find examples in the dataset.datasets.py
#       dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4) -- a iterator and decide the batch_size and shuffle, just a function to organize the dataset
# 2. define model:
#       model = scratch_nn()  # you can pass any attributes into it if there is a required attribute.
# 3. Train the model / OR load the weights
#       train(model, dataset, val_dataset, lr, num_epoch, num_classes)
# 4. Inference


def fixed_seed(SEED):
    set_determinism(SEED)  # set seed for this run
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
        

def main_process(args):

    fixed_seed(SEED=4)

    # mlflow.set_tracking_uri("http://nodelogin02:5000")
    experiment = mlflow.set_experiment('AIMIRA')
    
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
        for fold in range(args.total_folds):
            id = record_1st(RECORD_FPATH)
            all_folds_id_ls.append(id)
            with mlflow.start_run(run_name=str(id) + '_fold_' + str(fold), tags={"mlflow.note.content": f"fold: {fold}"}, nested=True):
                args.fold = fold
                args.id = id
                tmp_args_dt = vars(args)
                log_params(tmp_args_dt)
                
                mypath = aimira_Path(args.id, check_id_dir=False)

                # Step. 1 get the dataset: (highly recommand a function to generate the dataset, to make the code clean)
                # train_dataset, val_dataset = dataset_generator(data_dir=data_dir)
                generator= AIMIRA_generator(data_root=mypath.img_dir, 
                                            target_category=['TRT'], 
                                            target_site=[args.site],
                                            target_dirc=['TRA', 'COR'], 
                                            target_reader=['Reader1', 'Reader2'], 
                                            target_timepoint=['1'], 
                                            target_biomarker=['SYN', 'TSY', 'BME'],
                                            task_mode='clip', 
                                            score_sum=True,
                                            working_dir=mypath.project_dir, 
                                            print_flag=True, 
                                            max_fold=args.total_folds)
                
                run(args, mypath, generator)
       
        log_all_metrics(all_folds_id_ls, current_id, experiment, modes=['valid'], parent_dir = mypath.ex_dir + '/')

def evaluate(modes, mypath):

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
            os.rename(os.path.dirname(pred_fpath) + '/valid_scatter.png', os.path.dirname(pred_fpath) + f'/valid_scatter.png')
            
        if os.path.exists(os.path.dirname(pred_fpath) + '/test_scatter.png'):
            os.rename(os.path.dirname(pred_fpath) + '/test_scatter.png', os.path.dirname(pred_fpath) + f'/test_scatter.png')
            
         
def run(args, mypath, generator):

    
    train_dataset, val_dataset = generator.returner(fold_order=args.fold, material='img',
                 monai=True, full_img=7, dimension=2,
                 contrast=args.contrast, data_balance=False,
                 path_flag=True)
    
    # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
    model = ModelClip(group_num=2, group_cap=7, out_ch=1, width=2, dimension=2, extra_fc=False)

    # Step. 4 Load the weights and predict
    model = pretrained(model=model, 
                       model_file_name=mypath.data_dir_root + f"/SYN_TSY_BME__{args.site}_2dirc_fold0Sum.model")
    
    
    # freeze part model
    # if 'encoder' in args.freeze:                
    #     for param in model.encoder_class.parameters():
    #         param.requires_grad = False
    # if 'decoder' in args.freeze:                
    #     for param in model.decoder.parameters():
    #         param.requires_grad = False
            
            
    # for module in model.decoder.modules():
    #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #         torch.nn.init.kaiming_normal_(module.weight)
    #         if module.bias is not None:
    #             torch.nn.init.constant_(module.bias, 0)
                
    model.to(device)


    # Step. 3 Train the model /OR load the weights
    # train(model=model, dataset=train_dataset, val_dataset=val_dataset, lr=args.lr, num_epoch=args.epochs)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    G,P, id = predict(model, val_dataloader, 'valid', mypath=mypath, save_results = True)

    modes = ['valid']

    evaluate(modes, mypath)
    
    # print(classification_report(G,P))   


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.site = 'Foot'
    main_process(args)
    
    


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
            
          

            

 
        # add classification metrics
        # metrics_cls_dt = metrics_cls(label_fpath, pred_fpath, threshold_baseline=-3.07)


    print('Finish all things!')
