from torch.nn import MSELoss, HuberLoss
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from aimira_generators.aimira_generator import AIMIRA_generator
from main_func import train, pretrained, predictplus
from models.clip_model import ModelClip
# from utils.output_finder import output_finder

# from utils.log import Record
from torchsummary import summary
import os
import numpy as np
from typing import Union
from aimira_generators.aimira_scanner.filter.filters import point_filter

import os


def output_finder(target_category:list, target_site:list, target_dirc:list, target_biomarker, fold_order:int, sumscore:bool=False)->str:
    cate_name = ''
    if target_category is not None:
            for cate in target_category:
                cate_name = cate_name + cate + '_'
    else:
        cate_name = cate_name + 'All_'
    if len(target_site) > 1:
        site_name = str(len(target_site)) + 'site'
    else:
        site_name = target_site[0]
    
    if len(target_dirc) <2:
        dirc_name = target_dirc[0]
    else:
        if 'TRA' in target_dirc or 'COR' in target_dirc:
            dirc_name = '2dirc'
        else:
            dirc_name = '2read'
    sumscore_flag = 'Sum' if sumscore else ''
    if target_biomarker:  
        if len(target_biomarker)>1:
            replace = 'ALLBIO'
        else:
            replace = f'ALL{target_biomarker[0]}'
        output_name = "./models/weights/{}/{}_{}_{}_fold{}{}.model".format(replace, cate_name, site_name, dirc_name, fold_order, sumscore_flag)
    else:
        output_name = "./models/weights/{}_{}_{}_fold{}{}.model".format(cate_name, site_name, dirc_name, fold_order, sumscore_flag)
    output_dir = os.path.dirname(output_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_name


def main_process(data_dir='R:\\AIMIRA\\AIMIRA_Database\\LUMC', target_category=['TRT'], 
                 target_site=['Wrist'], target_dirc=['TRA', 'COR'],
                 target_biomarker=['SYN'], target_reader=['Reader1', 'Reader2'], 
                 target_timepoint=['1'], task_mode='clip',
                 distri_flag:bool=False, pretrain:bool=False, full_img:Union[bool, int]=True,
                 dimension:int=2, train_flag:bool=True, score_sum:bool=False,
                 model_csv:bool=False, extension:int=0):
    if target_biomarker:
        for item in target_biomarker:
            assert (item in ['ERO', 'BME', 'SYN', 'TSY'])
    dataset_generator = AIMIRA_generator(data_dir, target_category, target_site, target_dirc, target_reader,
                                          target_biomarker, target_timepoint, task_mode, 
                                          train_flag=train_flag, print_flag=True, filter=[point_filter])
    # fold_log = Record()
    for fold_order in range(0, 1):
        # in_fold_log = Record('gt', 'pred', 'diff', 'abs_path')
        # save records
        save_bio = target_biomarker[0] if len(target_biomarker)==1 else 'all'
        save_site = target_site[0] if len(target_site)==1 else 'multiple'
        save_father_dir = os.path.join('./aimira_generators/aimira_fig', f'{save_site}_{save_bio}')
        if not os.path.exists(save_father_dir):
            os.makedirs(save_father_dir)
        save_dir = os.path.join(save_father_dir, f'fold_{fold_order}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # get dataset
        train_dataset, val_dataset = dataset_generator.returner(fold_order=fold_order,
                                                                material='img', monai=True, full_img=full_img,
                                                                dimension=dimension, contrast=False, 
                                                                data_balance=False, score_sum=score_sum,
                                                                path_flag=True)
        # input: [N*5, 512, 512] + int(label) / dimension=3: [N, 5, 512, 512]

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        if full_img is True:
            depth = 20
        elif isinstance(full_img, int):
            depth = full_img
        else:
            depth = 5
        group_num = len(target_site) * len(target_dirc)   # input is a (5*site*dirc) * 512 * 512 img
        out_ch = 0
        out_ch = 1
 
        model = ModelClip(group_num=group_num, group_cap=depth, out_ch=out_ch, width=2, dimension=dimension) 

        batch_size = 6
   
        output_name = output_finder(target_biomarker, target_site, target_dirc, None, fold_order)
        if pretrain:
            model = pretrained(model=model, output_name=output_name)
            l_rate = 0.00001
        else:
            # l_rate = 0.00004 # 3D
            l_rate = 0.00004 # 2D
        # Step. 2.2: criterion
        criterion = MSELoss() # HuberLoss() # 

        # Step. 3 Train the model /OR load the weights  
        if train_dataset is not None and train_flag:
            train(model=model, dataset=train_dataset, val_dataset=val_dataset, 
                  lr=l_rate, num_epoch=60, batch_size=batch_size, output_name=output_name,
                  extra_aug_flag=False, criterion=criterion, optim_ada=True, distri_flag=distri_flag, out_ch=out_ch,
                  save_dir=save_dir)

        # Step. 4 Load the weights and predict
        model = pretrained(model=model, output_name=output_name)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        G,P,_,abs_path = predictplus(model, val_dataloader, arg_flag=False)  # must be false -- it's not a classification -- need more output
        # G,P -- [batch * (scores(43))]
        



if __name__ == '__main__':
    biomakrer_zoo = [['SYN', 'TSY', 'BME']]# , ['SYN'], ['TSY'], ['BME']]
    site_zoo = [['Wrist'], ['MCP'], ['Foot']]#['Wrist', 'MCP', 'Foot']]#, ['Wrist'], ['MCP'], ['Foot']]  # , 
    for site in site_zoo:
        for bio in biomakrer_zoo:
            main_process(data_dir='R:\\AIMIRA\\AIMIRA_Database\\LUMC', target_category=['TRT'], 
                         # data_dir='R:\\AIMIRA\\AIMIRA_Database\\LUMC', target_category=['TRT', 'PLA'], 
                        target_site=site, target_dirc=['TRA', 'COR'],
                        target_biomarker=bio, target_reader=['Reader1', 'Reader2'], 
                        target_timepoint=['1'], task_mode='clip',
                        distri_flag=True, pretrain=False, full_img=7,
                        dimension=2, train_flag=True, score_sum=True,
                        model_csv=False, extension=0)  # TODO
