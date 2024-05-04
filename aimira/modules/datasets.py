import os
import SimpleITK as sitk
import numpy as np
from torch.utils import data
from skimage.transform import resize
import torch
from typing import Union
# from aimira_generators.dataset.cleanup.cleanup import clean_main


class AIMIRACLIP(data.Dataset):  # contrast means -- compare different time points
    def __init__(self, img_list:list, ):
        # self.target_img_split, [5 * split[id[tp1[path1:cs, ...] tp2[...], ...], ...]
        # self.target_ramris_split, [5 * split[id1[tp1[site1_array, site2_array], tp2[], ...], id2[tp[]], ...]

            
        self.slices = 7
        self.full_img = False

        self.img_list = []
        for id_img in img_list:  # img_list [ID[TP[PATHS OF IMG, ...], ...], ...], id_img [ID[TP[PATHS OF IMG, ...]]
            for tp_img in id_img:  # tp_img TP[PATHS OF IMG, ...]
                self.img_list.append(tp_img)  # self.img_list [ID_TP[PATHS OF IMG], ...] 

    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self._flatten_getitem(idx)

    def _flatten_getitem(self, idx):
        data = self._load_file(idx)  # data list [scan-tra, scan-cor]
        for i in range(len(data)):
            data[i] = torch.from_numpy(data[i])
            if self.transform is not None:
                data[i] = self.transform(data[i])
        # data list [tensors]
        data = torch.vstack(data)

        return data     


    def _load_file(self, idx):
        data_matrix = []
        paths = self.img_list[idx]
        for indiv_path in paths:
            # indiv_path: 'subdir\names.mha:cs'
            try:
                disk, path, cs = indiv_path.split(':')  # 'subdir\names.mha', 'cs'
            # updated: 'subdir\names.mha:1to6plus1to11'
            except:
                disk = None
                path, cs = indiv_path.split(':')  # 'subdir\names.mha', 'cs'
            five, ten = cs.split('plus')
            fivelower, fiveupper = five.split('to')
            tenlower, tenupper = ten.split('to')
            if self.slices == 5:
                lower, upper = fivelower, fiveupper
            else:
                lower, upper = tenlower, tenupper
            lower, upper = int(lower), int(upper)
            abs_path = disk + ':' + path if disk else path
            data_mha = sitk.ReadImage(abs_path)
            data_array = sitk.GetArrayFromImage(data_mha)
            data_array = self._itensity_normalize(data_array[lower:upper])  # [5, 512, 512]
            if data_array.shape != (self.slices, 512, 512):
                if data_array.shape == (self.slices, 256, 256):
                    data_array = resize(data_array, (self.slices, 512, 512), preserve_range=True)  # preserve_range: no normalization
                else:
                    raise ValueError('the shape of input:{}, the id: {}, central_slice: {}'.format(data_array.shape, path, lower))
            # data_array = clean_main(data_array)
            data_matrix.append(data_array.astype(np.float32))
        return data_matrix, abs_path  # [N*5, 512, 512]

  
    def _itensity_normalize(self, volume: np.array):
        """
        normalize the itensity of a volume based on the mean and std of nonzeor region
        inputs:
            volume: the input volume
        outputs:
            out: the normalized volume
        """
        min_value = volume.min()
        max_value = volume.max()
        if max_value > min_value:
            out = (volume - min_value) / (max_value - min_value)
        else:
            out = volume
        # out_random = np.random.normal(0, 1, size=volume.shape)
        # out[volume == 0] = out_random[volume == 0]
        return out