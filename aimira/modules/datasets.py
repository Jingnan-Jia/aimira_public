# # -*- coding: utf-8 -*-
# # @Time    : 7/11/21 2:31 PM
# # @Author  : Jingnan
# # @Email   : jiajingnan2222@gmail.com
# import itertools
# import json
# import os
# from aimira.modules.trans import LoadDatad, SaveDatad, RandomCropForegroundd, RemoveTextd
# from sklearn.model_selection import KFold
# import monai
# from torch.utils.data import Dataset
# import pandas as pd
# from monai.transforms import RandSpatialCropd, RandGaussianNoised, CastToTyped, ToTensord, \
#     CenterSpatialCropd, AddChanneld, ScaleIntensityRanged, SpatialPadd
# from monai.data import DataLoader
# from tqdm import tqdm
# import torch
# from mlflow import log_metric, log_param, log_params
# import numpy as np
# from pathlib import Path
# from medutils.medutils import load_itk
# import random
# import glob
# import sys
# sys.path.append("../..")
# from torch.utils.data import WeightedRandomSampler
# import torch_geometric
# from monai.transforms import RandomizableTransform
# from torch_geometric.datasets import TUDataset

# # import streamlit as st


# PAD_DONE = False



# def sampler_by_disext(data, ref = 'DLCOc_SB') -> WeightedRandomSampler:
#     """Balanced sampler according to score distribution of disext.

#     Args:
#         tr_y: Training labels.
#             - Three scores per image: [[score1_disext, score1_gg, score1_ret], [score2_disext, score2_gg, score3_ret],
#              ...]
#             - One score per image: [score1_disext, score2_disext, ...]
#         sys_ratio:

#     Returns:
#         WeightedRandomSampler

#     Examples:
#         :func:`ssc_scoring.mymodules.mydata.LoadScore.load`
#     """

#     disext_np = np.array([i[ref] for i in data])
#     disext_np = (disext_np + 0.5)//1

#     disext_unique = np.unique(disext_np)
#     disext_unique_list = list(disext_unique)

#     class_sample_count = np.array([len(np.where(disext_np == t)[0]) for t in disext_unique])
#     weight = 1. / class_sample_count

#     print("class_sample_count", class_sample_count)
#     print("unique_disext", disext_unique_list)
#     print("weight: ", weight)

#     samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])

#     # weight = [nb_nonzero/len(data_y_list) if e[0] == 0 else nb_zero/len(data_y_list) for e in data_y_list]
#     samples_weight = samples_weight.astype(np.float32)
#     samples_weight = torch.from_numpy(samples_weight)
#     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
#     print(list(sampler))
#     return sampler


# def build_dataset(file_ls, PNB=140000):
#     points_ls = []
#     for i in file_ls:
#         a = pd.read_pickle(i)
#         # convert voxel location to physical mm
#         b = a['data'][:PNB, :3] * a['spacing']
#         c = np.concatenate((b, a['data'][:PNB, -1].reshape(-1, 1)), axis=1)
#         points_ls.append(c)
#     points_np = np.array(points_ls)
#     return points_np

# def rearrange_transforms(xforms):
#     # 分离不是 RandomizableTransform 的元素和是 RandomizableTransform 的元素
#     non_randomizable = [x for x in xforms if not isinstance(x, RandomizableTransform)]
#     randomizable = [x for x in xforms if isinstance(x, RandomizableTransform)]

#     # 重新组合列表
#     rearranged = non_randomizable + randomizable

#     return rearranged


# def xformd(mode, args, pad_truncated_dir='tmp'):
#     z_size = args.z_size
#     y_size = args.y_size
#     x_size = args.x_size
#     target = args.target
#     crop_foreground = args.crop_foreground
#     pad_ratio = args.pad_ratio
#     # inputmode = args.input_mode
#     PNB = args.PNB

#     post_pad_size = [int(i * pad_ratio) for i in [z_size, y_size, x_size]]
#     inputmodes = args.input_mode.split('-')  # split 'ct-pcd_vessel' to ['ct', 'pcd_vessel]
#     xforms = []
#     for inputmode in inputmodes:
#         keys = (inputmode, )
#         if inputmode in ('vessel_skeleton_graph'):
#             xforms.append(LoadGraphd(keys=keys, target=target))
#             # xforms.append(SamplePointsd(keys=keys, num=args.PNB, train_mode=(mode=='train')))
#             # xforms.append(CenterNormd(keys=keys))
            
            
#         elif inputmode in ['vessel_skeleton_pcd', 'lung_mask_pcd']:
            
#             xforms.extend([LoadPointCloud(keys=keys, target=target, position_center_norm=args.position_center_norm,
#             repeated_sample=args.repeated_sample, FPS_input=args.FPS_input, set_all_r_to_1=args.set_all_r_to_1, 
#             set_all_xyz_to_1=args.set_all_xyz_to_1, in_channel=args.in_channel, scale_r=args.scale_r),
#                     SampleShuffled( keys=keys, PNB=PNB, train_mode=(mode=='train'),repeated_sample=args.repeated_sample),
#                     # ShiftCoordinated(keys=keys, position_center_norm=args.position_center_norm),
#                     ])

#         else:
#             if inputmode == 'vessel':
#                 min_value, max_value = 0, 1
#             elif 'ct' in inputmode:
#                 min_value, max_value = -1500, 1500
#             elif inputmode == 'lung_mask':
#                 min_value, max_value = 0, 1
#             else:
#                 raise Exception(f"wrong input mode: {inputmode}")
#             if crop_foreground:
#                 keys = keys + ('lung_mask',)

#             if not os.path.isdir(pad_truncated_dir):
#                 os.makedirs(pad_truncated_dir)
#             xforms.extend([LoadDatad(keys=keys[0], target=target, crop_foreground=crop_foreground, inputmode=inputmode), 
#                             AddChanneld(keys=keys),
#                             SpatialPadd( keys=keys[0], spatial_size=post_pad_size, mode='constant', constant_values=min_value)
#                             ])
#             if crop_foreground:
#                 xforms.append(SpatialPadd( keys=keys[1], spatial_size=post_pad_size, mode='constant', constant_values=0))
#             xforms.append(ScaleIntensityRanged( keys=keys[0], a_min=min_value, a_max=max_value, b_min=-1, b_max=1, clip=True))

#             # xforms.append()
#             if mode == 'train':
#                 if crop_foreground:
#                     xforms.extend([RandomCropForegroundd(keys=keys, roi_size=[ z_size, y_size, x_size], source_key='lung_mask')])
#                 else:
#                     xforms.extend([RandSpatialCropd(keys=keys, roi_size=[ z_size, y_size, x_size], random_center=True, random_size=False)])
#                 # xforms.extend([RandGaussianNoised(keys=keys, prob=0.5, mean=0, std=0.01)])
#             else:
#                 xforms.extend( [CenterSpatialCropd(keys=keys, roi_size=[z_size, y_size, x_size])])

#             # xforms.append(SaveDatad(pad_truncated_dir+"/patches_examples/" + mode))

#             # ('pat_id', 'image', 'lung_mask', 'origin', 'spacing', 'label')
#     ct_pnn_inputmodes = [i for i in inputmodes if i!='vessel_skeleton_graph']
#     if len(ct_pnn_inputmodes):
#         xforms.append(CastToTyped(keys=ct_pnn_inputmodes, dtype=np.float32))
#     xforms.extend([# ToTensord(keys=inputmodes),  # it is moved to collect_fun because graph cannot be converted to tensor
#                     RemoveTextd(),
#                     RemoveDuplicatedd(keys = ['label', 'pat_id'])])
    
#     xforms = rearrange_transforms(xforms)
#     transform = monai.transforms.Compose(xforms)
#     return transform


# def filter_data(data_dir, pft_df):
#     pft_df.drop(pft_df[np.isnan(pft_df.DLCO_SB)].index, inplace=True)
#     pft_df.drop(pft_df[np.isnan(pft_df.DateDF_abs)].index, inplace=True)
#     pft_df.drop(pft_df[pft_df.DateDF_abs > 10].index, inplace=True)


#     # get availabel files
#     scans = glob.glob(data_dir + "/*.mha")

#     availabel_id_set = set([Path(id).stem[:19] for id in scans])  

#     pft_df.drop(pft_df.loc[~pft_df['subjectID'].isin(availabel_id_set)].index, inplace=True)

#     # pft_df = pft_df.drop(pft_df[pft_df['subjectID'] not in availabel_id_set].index)
#     # print(f"length of scans: {len(scans)}, length of labels: {len(pft_df)}")
#     assert len(scans) >= len(pft_df)

#     return pft_df


# def pat_fromo_csv(mode: str, data, fold=1) -> np.ndarray:
#     tmp_ls = []
#     ex_fold_dt = {1: '905', 2: '914', 3: '919', 4: '924'}
#     df = pd.read_csv(
#         f"/data1/jjia/lung_function/lung_function/scripts/results/experiments/{ex_fold_dt[fold]}/{mode}_label.csv")
#     pat_ls = [patid for patid in df['pat_id']]
#     for d in data:
#         if int(d['subjectID'].split('_')[-1]) in pat_ls:
#             tmp_ls.append(d)
#     tmp_np = np.array(tmp_ls)
#     return tmp_np


# def pat_from_json(data, fold=1) -> np.ndarray:
#     with open('/home/jjia/data/lung_function/lung_function/modules/data_split.json', "r") as f:
#         data_split = json.load(f)

#     valid = data_split[f'valid_fold{fold}']
#     test = data_split[f'test']
#     # train = []
#     train = list(itertools.chain(
#         *[data_split[f'valid_fold{i}'] for i in [1, 2, 3, 4] if i != fold]))
#     # for i in tmp_ls:
#     #     train.extend(i)

#     def avail_data(pat_ls, data) -> np.ndarray:
#         tmp_ls = []
#         for d in data:
#             tmp_id = d['subjectID'].split('_')[-1]
#             if tmp_id in pat_ls:
#                 tmp_ls.append(d)
#         return np.array(tmp_ls)

#     train = avail_data(train, data)
#     valid = avail_data(valid, data)
#     test = avail_data(test, data)
#     return train, valid, test


# def collate_fn(batch):  # batch_dt is a list of dicts
#     # 初始化一个字典来存储每种类型的数据的批次
#     batched_data = {k: [] for k in batch[0].keys()}
    

#     # 遍历batch中的每个字典
#     for item in batch:
#         # 将相应的数据添加到batched_data中
#         for key in item.keys():
#             batched_data[key].append(item[key])

#     # 对于numpy.ndarray类型的数据，将其转换为torch.tensor并堆叠
#     for key in batched_data.keys():
#         if 'graph' in key:
#             batched_data[key] = torch_geometric.data.Batch.from_data_list(batched_data[key])
#         else:
#             batched_data[key] = torch.tensor(np.stack(batched_data[key]))

#     return batched_data

# def organize_labels(df):
#     data_ls = []
#     dt = {}
    
#         data_ls.append(dt)
#     return data_ls

# def all_loaders(data_dir, label_fpath, args, datasetmode=('train', 'valid', 'test'), nb=None):

#     label_excel = pd.read_excel(label_fpath, engine='openpyxl')  # read the patient ids from the score excel 
#     label_excel = filter_data(data_dir, label_excel)  # select baseline and 12-month follow up (with some 4-month follow up)
#     data_ls = organize_labels(label_excel)
#     data_ls =  {'id': , } # a list of dicts, in each dict, id and 12 specific scores are stored as baseline scores, 12 scores are follow up scores
    

#     # nparray is easy for kfold split
#     data = np.array(label_excel.to_dict('records'))     

#     if args.test_pat == 'random_as_ori':
#         tr_data, vd_data, ts_data = pat_from_json(data, args.fold)
#     else:  # set a shuffle seed !
#         kf = KFold(n_splits=args.total_folds, shuffle=True,
#                 random_state=args.kfold_seed)  # for future reproduction
   
#         ts_nb = 27  # 27 testing patients, 88 train&valid patients
#         tr_vd_data, ts_data = data[:-ts_nb], data[-ts_nb:]
#         kf_list = list(kf.split(tr_vd_data))
#         tr_pt_idx, vd_pt_idx = kf_list[args.fold - 1]
#         tr_data = tr_vd_data[tr_pt_idx]
#         vd_data = tr_vd_data[vd_pt_idx]
#         print(f"length of training data: {len(tr_data)}")
#     if nb:
#         tr_data, vd_data, ts_data = tr_data[:nb], vd_data[:nb], ts_data[:nb]



                    
                    
                    
#     data_dt = {}
#     if 'train' in datasetmode:
#         tr_dataset = monai.data.CacheDataset(data=tr_data, transform=xformd('train', args), num_workers=0, cache_rate=1)
#         train_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
#                                     shuffle=True, num_workers=args.workers, persistent_workers=True,
#                                     pin_memory=True, collate_fn=collate_fn)
#         data_dt['train'] = train_dataloader

#     if 'valid' in datasetmode:
#         vd_dataset = monai.data.CacheDataset(data=vd_data, transform=xformd('valid', args), num_workers=0, cache_rate=1)
#         valid_dataloader = DataLoader(vd_dataset, batch_size=args.batch_size,
#                                     shuffle=False, num_workers=args.workers, persistent_workers=True, collate_fn=collate_fn)
#         data_dt['valid'] = valid_dataloader

#     if 'test' in datasetmode:
#         ts_dataset = monai.data.CacheDataset(data=ts_data, transform=xformd('test', args), num_workers=0, cache_rate=1)
#         test_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size,
#                                     shuffle=False, num_workers=args.workers, persistent_workers=True, collate_fn=collate_fn)
#         data_dt['test'] = test_dataloader


#     return data_dt
