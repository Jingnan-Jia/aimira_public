
import os
from typing import Union
from abc import ABC

class aimira_Path():
    """
    Common path values are initialized.
    """
    results_dir: str = '/home/jjia/data/aimira/scripts/results'
    record_file = os.path.join(results_dir, "records.csv")
    log_dir = os.path.join(results_dir, 'logs')
    ex_dir = os.path.join(results_dir, 'experiments')
    data_dir_root = '/home/jjia/data/aimira/data'
    label_fpath = "/home/jjia/data/aimira/data/TE_scores_MRI_serieel_nieuw.xlsx"    

    for directory in [results_dir, log_dir, ex_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            print('successfully create directory:', directory)

    def __init__(self, id: Union[int, str], check_id_dir: bool = False, space='1.0'):
        if space in ('1.0', '1.5', 1, 1.5):
            # if space == 1:
            #     space = '1.0'
            # else:
            #     space = '1.5'
            self.data_dir = self.data_dir_root + '/iso' + str(space)
        elif space == 'ori':
            self.data_dir = self.data_dir_root + '/ori_resolution'
        else:
            raise Exception(f"wrong space: {space}")
        

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