
import os
from typing import Union
from abc import ABC

class aimira_Path():
    """    
    Common path values are initialized.
    """
    project_dir = "/exports/lkeb-hpc/jjia/project/project/aimira/aimira_src"  
    results_dir: str = project_dir + '/scripts/results'
    record_file = os.path.join(results_dir, "records.csv")
    log_dir = os.path.join(results_dir, 'logs')
    ex_dir = os.path.join(results_dir, 'experiments')
    data_dir_root = project_dir.replace('aimira_src', 'aimira') + '/data'
    img_dir = data_dir_root + '/TRT_ori'
    label_ori_fpath = data_dir_root + "/TE_scores_MRI_serieel_nieuw.xlsx"    
    # label_all_fpath =  data_dir_root + "/pat_id_with_scores2_yanli_gt_slice.csv"  # pat_id_with_scores1


    for directory in [results_dir, log_dir, ex_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            print('successfully create directory:', directory)

    def __init__(self, id: Union[int, str], check_id_dir: bool = False):
        self.data_dir = self.data_dir_root 
  
        if isinstance(id, (int, float)):
            self.id = str(int(id))
        else:
            self.id = id
        self.id_dir = os.path.join(self.ex_dir, str(id))  # +'_fold_' + str(args.fold)
        if check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for directory in [self.data_dir, self.id_dir]:
            if not os.path.isdir(directory):
                os.makedirs(directory)
                print('successfully create directory:', directory)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')

    def save_pred_fpath(self, mode):
        return os.path.join(self.id_dir, mode + '_pred.csv')

    def save_label_fpath(self, mode):
        return os.path.join(self.id_dir, mode + '_label.csv')