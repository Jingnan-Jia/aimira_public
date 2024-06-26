# -*- coding: utf-8 -*-
# @Time    : 7/11/21 2:31 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import itertools
import json
import os
from aimira.modules.trans import LoadDatad
from sklearn.model_selection import KFold
import monai
from torch.utils.data import Dataset
import pandas as pd
from monai.transforms import RandSpatialCropd, RandGaussianNoised, ToTensord, \
    CenterSpatialCropd, ScaleIntensityRanged, SpatialPadd
from monai.data import DataLoader
from tqdm import tqdm
import torch
from mlflow import log_metric, log_param, log_params
import numpy as np
from pathlib import Path
from medutils.medutils import load_itk
import random
import glob
import sys
sys.path.append("../..")
from torch.utils.data import WeightedRandomSampler
from monai.transforms import RandomizableTransform
from aimira.modules.trans import Label_fusiond, Input_fusiond, CastToTyped, ToTensord
from monai import transforms
# import streamlit as st
from aimira.modules.central_slice import central_selector


PAD_DONE = False



def sampler_by_disext(data, ref = 'DLCOc_SB') -> WeightedRandomSampler:
    """Balanced sampler according to score distribution of disext.

    Args:
        tr_y: Training labels.
            - Three scores per image: [[score1_disext, score1_gg, score1_ret], [score2_disext, score2_gg, score3_ret],
             ...]
            - One score per image: [score1_disext, score2_disext, ...]
        sys_ratio:

    Returns:
        WeightedRandomSampler

    Examples:
        :func:`ssc_scoring.mymodules.mydata.LoadScore.load`
    """

    disext_np = np.array([i[ref] for i in data])
    disext_np = (disext_np + 0.5)//1

    disext_unique = np.unique(disext_np)
    disext_unique_list = list(disext_unique)

    class_sample_count = np.array([len(np.where(disext_np == t)[0]) for t in disext_unique])
    weight = 1. / class_sample_count

    print("class_sample_count", class_sample_count)
    print("unique_disext", disext_unique_list)
    print("weight: ", weight)

    samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])

    # weight = [nb_nonzero/len(data_y_list) if e[0] == 0 else nb_zero/len(data_y_list) for e in data_y_list]
    samples_weight = samples_weight.astype(np.float32)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    print(list(sampler))
    return sampler



def rearrange_transforms(xforms):
    # 分离不是 RandomizableTransform 的元素和是 RandomizableTransform 的元素
    non_randomizable = [x for x in xforms if not isinstance(x, RandomizableTransform)]
    randomizable = [x for x in xforms if isinstance(x, RandomizableTransform)]

    # 重新组合列表
    rearranged = non_randomizable + randomizable

    return rearranged


def xformd(mode, args):     

    # all_keys = 
    # date_keys = 
    # ori_img_keys = 
    
    position_codes = args.input_position_code.split('-')  # split 'ct-pcd_vessel' to ['ct', 'pcd_vessel]
    xforms = []
    

    xforms.extend([LoadDatad(position_codes=position_codes, data_image_dir = args.data_image_dir), 
                            # AddChanneld(keys=keys),
                            ])

    # if len(inputmodes):
    xforms.extend([
                   ToTensord(dtype=torch.float)])
    # xforms.append()
    xforms.extend([   # it is moved to collect_fun because graph cannot be converted to tensor
                    Input_fusiond(args.input_position_code, args.nb_slices),
                    Label_fusiond(args.target, args.input_position_code),  # label_fusiond should be in front of Inut_fusiond
                    # RemoveTextd(keys=None), 
                    CastToTyped(dtype=torch.float)
                    # RemoveDuplicatedd(keys = ['label', 'pat_id']
                                      ])
    if mode == 'train':
        img_keys = ['input']
        xforms.extend([
                        # transforms.RandGaussianNoise(0.2, 0, 0.1),
                        transforms.RandFlipd(keys=img_keys, prob=0.5, spatial_axis=0),
                        transforms.RandRotated(keys=img_keys,range_x=(10), prob=0.5),
                        transforms.RandAffined(keys=img_keys,prob=1.0, translate_range=(20, 20)),
                        # transforms.RandShiftIntensity(offsets=0.1, safe=True, prob=0.2),
                        # transforms.RandStdShiftIntensity(factors=0.1, prob=0.2),
                        # transforms.RandBiasField(degree=2, coeff_range=(0, 0.1), prob=0.2),
                        # transforms.RandAdjustContrast(prob=0.5, gamma=(0.9, 1.1)),
                        transforms.RandHistogramShiftd(keys=img_keys,num_control_points=10, prob=0.2),
                        transforms.RandZoomd(keys=img_keys,prob=0.3, min_zoom=0.9, max_zoom=1.0, keep_size=True)
                        ]),
           
    xforms = rearrange_transforms(xforms)
    transform = monai.transforms.Compose(xforms)
    return transform


def filter_data(data_dir, df):

    # df = df[df['hoeveelste_MRI']==1]  # baseline visit
    
    # calculate inflammation score 
    # 提取所有以 'MT'等 开头且不包含 'ERO' 的列名，求和得到IFM
    for position_code in ['MT', 'MC', 'WR']:  
        mt_columns = [col for col in df.columns if col.startswith(f'{position_code}') and 'ERO' not in col]

        # 计算符合条件的列的值的和，然后存放到新的列 MT_IFM 中,这里千万记得除以2！因为是2个医生的评分
        df.loc[:, f'{position_code}_IFM'] = df[mt_columns].sum(axis=1) / 2
    ifm_columns = ['MT_IFM', 'MC_IFM', 'WR_IFM']  # total inflammation
    df.loc[:, 'IFM'] = df.loc[:, ifm_columns].sum(axis=1)
    
    
    # extract all rows with 'treatment' equals to 1
    df_trt = df[df['treatment'] == 1]
    # in the new dataframe, df_trt, select all the rows with VISNUMMER equal to 1
    df_trt_bl = df_trt[df_trt['VISNUMMER_aangepast'] == 1]
    # the remaining rows build a new dataframe called 'df_trt_fu'.
    df_trt_fu = df_trt[df_trt['VISNUMMER_aangepast'] != 1]
    # In df_trt_fu, remove the rows whose 'TENR' value is not in the df_trt_bl.
    df_trt_fu = df_trt_fu[df_trt_fu['TENR'].isin(df_trt_bl['TENR'])]
    # use df_trt_fu minus df_trt_bl, get a new dataframe, called df_trt_change.
    df_trt_change = df_trt_fu.set_index('TENR') - df_trt_bl.set_index('TENR')
    # Update column names
    df_trt_change.columns = [str(col) + '_change' for col in df_trt_change.columns]

    # concatenate df_trt_change to df_trt_bl to make sure their 'TENR' alligned
    result = pd.concat([df_trt_bl.set_index('TENR'), df_trt_change], axis=1).reset_index()

    # get availabel files
    filenames = glob.glob(data_dir + "/*PostCOR*.mha")  
    patient_numbers = [int(filename.split('Treat')[-1][:4]) for filename in filenames]
    exclude_ls = [72, 105, 59, 145, 365, 458] # 72 and 105 lack the baseline scans, the others lack follou-up scans/scores
    patient_numbers = [i for i in patient_numbers if i not in exclude_ls]
    # 将患者编号与 DataFrame 中的 TENR 列进行比较，找出重合的文件名和 DataFrame 行
    # matched_filenames = [filename for filename, patient_number in zip(filenames, patient_numbers) if patient_number in df['TENR'].values]
    matched_df = result[result['TENR'].isin(patient_numbers)]

    # get the central slices for each scan


    return matched_df


def pat_fromo_csv(mode: str, data, fold=1) -> np.ndarray:
    tmp_ls = []
    ex_fold_dt = {1: '905', 2: '914', 3: '919', 4: '924'}
    df = pd.read_csv(
        f"/data1/jjia/lung_function/lung_function/scripts/results/experiments/{ex_fold_dt[fold]}/{mode}_label.csv")
    pat_ls = [patid for patid in df['pat_id']]
    for d in data:
        if int(d['subjectID'].split('_')[-1]) in pat_ls:
            tmp_ls.append(d)
    tmp_np = np.array(tmp_ls)
    return tmp_np


def pat_from_json(json_fpath, fold, total_folds) -> np.ndarray:
    with open(json_fpath, "r") as f:
        data_split = json.load(f)

    valid = data_split[f'valid_fold{fold}']
    test = data_split[f'test']
    # train = []
    train = list(itertools.chain(
        *[data_split[f'valid_fold{i}'] for i in list(range(1, total_folds+1)) if i != fold]))
    # for i in tmp_ls:
    #     train.extend(i)

    # def avail_data(pat_ls, data) -> np.ndarray:
    #     tmp_ls = []
    #     for d in data:
    #         tmp_id = d['subjectID'].split('_')[-1]
    #         if tmp_id in pat_ls:
    #             tmp_ls.append(d)
    #     return np.array(tmp_ls)

    # train = avail_data(train, data)
    # valid = avail_data(valid, data)
    # test = avail_data(test, data)
    return train, valid, test


def collate_fn(batch):  # batch_dt is a list of dicts
    # 初始化一个字典来存储每种类型的数据的批次
    batched_data = {k: [] for k in batch[0].keys()}
    

    # 遍历batch中的每个字典
    for item in batch:
        # 将相应的数据添加到batched_data中
        for key in item.keys():
            batched_data[key].append(item[key])

    # 对于numpy.ndarray类型的数据，将其转换为torch.tensor并堆叠
    for key in batched_data.keys():
        if type(batched_data[key][0]) != str:
            batched_data[key] = torch.tensor(np.stack(batched_data[key]))

    return batched_data

# def organize_labels(df):
#     data_ls = []
#     dt = {}
    
#         data_ls.append(dt)
#     return data_ls  clean_AIMIRA-LUMC-Treat0002_TRT-*WR_PostCOR*.mha
def six_path_with_slices(id, data_image_dir):
    
    out_ls = []
    for site in ['Wrist', 'MCP', 'Foot']:
        for view in ['COR', 'TRA']:
            fpath = glob.glob(data_image_dir + f"/clean_AIMIRA-LUMC-Treat{id:04d}_TRT-*{site}_Post{view}*.mha")[0]
            targent_slices = central_selector(fpath)
            fpath += targent_slices
            out_ls.append(fpath)
    return out_ls
            
def add_central_slices(label_excel, data_image_dir):
    # 应用自定义函数到'TENR'列的每个值
    new_column_names = [i+'_'+j+'_fpath_slice' for i in ['WR', 'MC', 'MT'] for j in ['COR', 'TRA']]
    label_excel[new_column_names] = label_excel.apply(lambda row: pd.Series(six_path_with_slices(row['TENR'], data_image_dir)), axis=1)

    # new_columns_values = label_excel['TENR'].apply(six_path_with_slices)
    # 将得到的列表拆分成6列，并将它们添加到label_excel中
    # label_excel[new_column_names] = pd.DataFrame(new_columns_values.tolist(), index=label_excel.index)

    return label_excel

def all_loaders(data_dir, label_fpath, filename, args, datasetmode=('train', 'valid', 'test'), nb=None):

    # 检查文件是否存在
    if not os.path.exists(filename):
        # 如果文件不存在，创建一个新文件并写入内容
        label_excel = pd.read_excel(label_fpath, engine='openpyxl')  # read the patient ids from the score excel 
        # label_excel = label_excel[:15]
        label_excel = filter_data(data_dir, label_excel)  # select baseline and 12-month follow up (with some 4-month follow up)
        label_excel = add_central_slices(label_excel, data_dir)
        label_excel.to_csv(filename, index=False)       
        print(f"文件 '{filename}' 已创建。")
    else:
        # 如果文件存在，直接读取文件内容
        label_excel = pd.read_csv(filename)
        print(f"文件 '{filename}' 已存在")
        

    # data_ls = organize_labels(label_excel)
    # data_ls =  {'id': , } # a list of dicts, in each dict, id and 12 specific scores are stored as baseline scores, 12 scores are follow up scores
    
    # 选择所有非序列化的列
    # non_serializable_columns = label_excel.select_dtypes(exclude=['number', 'bool']).columns

    # # 将非序列化的列转换为字符串格式
    # label_excel[non_serializable_columns] = label_excel[non_serializable_columns].astype(str)
    # for i in label_excel:
    for i in label_excel.columns:
        if pd.api.types.is_datetime64_any_dtype(label_excel[i]) or pd.api.types.is_timedelta64_dtype(label_excel[i]):
            label_excel.drop(columns=[i], inplace=True)

            # label_excel[i] = label_excel.loc[:, i].astype(str)
            
    # nparray is easy for kfold split
    data = np.array(label_excel.to_dict('records'))

    json_fpath = os.path.dirname(label_fpath) + f"/data_split_{args.total_folds}.{args.fold}.json"
    if not os.path.exists(json_fpath):
        kf = KFold(n_splits=args.total_folds, shuffle=True, random_state=args.kfold_seed)  # for future reproduction
   
        ts_nb = 27  # 27 testing patients, 88 train&valid patients
        tr_vd_data, ts_data = data[:-ts_nb], data[-ts_nb:]
        kf_list = list(kf.split(tr_vd_data))
        
        json_data = {}
        for fold in range(args.total_folds):
            tr_pt_idx, vd_pt_idx = kf_list[fold]
            json_data.update({f'valid_fold{fold+1}': tr_vd_data[vd_pt_idx].tolist()})
            json_data.update({'test': ts_data.tolist()})
        print(f"length of training data: {len(tr_pt_idx)}")
        
    
        with open(json_fpath,'w') as f:

            json.dump(json_data,f) 
            
    tr_data, vd_data, ts_data = pat_from_json(json_fpath, args.fold, args.total_folds)

        
    if nb:
        tr_data, vd_data, ts_data = tr_data[:nb], vd_data[:nb], ts_data[:nb]

    args.data_image_dir = os.path.dirname(label_fpath) + '/images'  # in xformd we need it
    data_dt = {}
    if 'train' in datasetmode:
        tr_dataset = monai.data.CacheDataset(data=tr_data, transform=xformd('train', args), num_workers=0, cache_rate=1)
        train_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.workers, persistent_workers=True,
                                    pin_memory=True, collate_fn=collate_fn)  
        data_dt['train'] = train_dataloader

    if 'valid' in datasetmode:
        vd_dataset = monai.data.CacheDataset(data=vd_data, transform=xformd('valid', args), num_workers=0, cache_rate=1)
        valid_dataloader = DataLoader(vd_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers, persistent_workers=True, collate_fn=collate_fn)
        data_dt['valid'] = valid_dataloader

    if 'test' in datasetmode:
        ts_dataset = monai.data.CacheDataset(data=ts_data, transform=xformd('test', args), num_workers=0, cache_rate=1)
        test_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers, persistent_workers=True, collate_fn=collate_fn)
        data_dt['test'] = test_dataloader


    return data_dt
