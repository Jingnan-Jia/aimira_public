from ctypes import Union
import os
from typing import Optional
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing   # 填充
from skimage.transform import resize

class PreClean:
    def __init__(self, path, target_shape=[20, 512, 512]):
        self.dir_path = path
        self.filename = os.listdir(self.dir_path)  # list
        self.length = []
        self.width = []
        self.depth = []
        self.len_wid_ratio = []
        self.target_shape = target_shape

    def runner(self):
        names = [os.path.join(self.dir_path, item) for item in self.filename]
        
        pool = Pool()
        Crop = list(pool.map(self.clean_main, names))
        print('success')

    def create_nonzero_mask_for_clean(self, data):  # data: [20, 512, 512]
        
        assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
        nonzero_mask = np.zeros(data.shape[:], dtype=bool)  # [512, 512] 因为20层应该是要全用，所以说完全可以把z轴作为channel
        for c in range(data.shape[0]):  # 读取单个sample
            max_value = np.max(data[c])
            this_mask = data[c] >= 0.1 * max_value # threshold = 0.1,  out of range(0, 1) after normalization
            nonzero_mask[c] = nonzero_mask[c] | this_mask  # 有一个为1则为1， 在循环后相当于取各个slice的最大值
        for c in range(len(nonzero_mask)):
            nonzero_mask[c] = binary_fill_holes(nonzero_mask[c])
            # 此后进行开闭操作去除artifacts： 开操作先腐蚀去除奇异点，再扩张还原。 闭操作使边缘均匀化，未必需要。
            nonzero_mask[c] = binary_opening(nonzero_mask[c], iterations=20)
            nonzero_mask[c] = binary_closing(nonzero_mask[c], iterations=1)
        return nonzero_mask

    def place_air_zero(self, data, nonzero_mask_clean):
        assert (data[0].shape == nonzero_mask_clean[0].shape)
        data[nonzero_mask_clean == 0] = 0
        return data

    def savefile(self, data, datapath: str, spacing, data_origin, data_direction, size_flag):
        data_dir = os.path.dirname(datapath)
        if size_flag:
            data_revise_dir = data_dir + '_clean' + size_flag
        else:
            data_revise_dir = data_dir + '_clean'
        if not os.path.exists(data_revise_dir):
            os.makedirs(data_revise_dir)
        data_name = os.path.split(datapath)[-1]
        if '.nii' in data_name:
            if size_flag:
                data_revise_name = 'clean_' + size_flag + data_name + '.gz'
            else:
                data_revise_name = 'clean_' + data_name + '.gz'
        elif '.mha' in data_name:
            if size_flag:
                data_revise_name = 'clean_' + size_flag + data_name
            else:
                data_revise_name = 'clean_' + data_name
        elif '.dcm' in data_name:
            if size_flag:
                data_revise_name = 'clean_' + size_flag + data_name
            else:
                data_revise_name = 'clean_' + data_name
        else:
            raise TypeError('the original file type seems to be out of dicom, mha or nii, please contact the code author.')
        dataname = os.path.join(data_revise_dir, data_revise_name)
        data_img = sitk.GetImageFromArray(data)
        data_img.SetSpacing(spacing)
        data_img.SetDirection(data_direction)
        data_img.SetOrigin(data_origin)
        sitk.WriteImage(data_img, dataname)

    def resize_cleaned(self, data: np.array, target_shape: list) -> np.array:
        
        # target_shape - list [depth, length, width] - TODO here the depth should be the max depth
        # ideal shape [20, 512, 512]
        # check the ratio of target_shape
        reshaped_data = resize(data, (target_shape[0], target_shape[1], target_shape[2]))
        reshaped_final_data = np.asarray(reshaped_data, dtype=np.int16)
        return reshaped_final_data  # array [target_depth, target_length, target_width]

    def crop_cleaned(self, data: np.array, target_shape: list) -> np.array:
        # target_shape - list [depth, length, width] - TODO here the depth should be the max depth
        # ideal shape [20, 512, 512]
        if data.shape[0] == target_shape[0]:
            # which means data [20, 512, 512]
            return data
        elif data.shape[0] > target_shape[0]:
            # which means data [x>20, 512, 512]
            croped_Data = data[:target_shape[0], :]
            return croped_Data
        else: # [x<20, 512, 512]
            target_data = np.zeros(target_shape, dtype=np.uint16)  # [20, 512, 512] * 0
            target_data[:data.shape[0], :] = data
            return target_data

    def clean_main(self, datapath: str, resize_flag: bool =False, depth_crop_flag: bool =False) -> None:
        assert datapath
        data = sitk.ReadImage(datapath)
        data_spacing = data.GetSpacing()
        data_origin = data.GetOrigin()
        data_direction = data.GetDirection()
        data_array = sitk.GetArrayFromImage(data)  # output [slice, 512, 512]
        # self.savefile(data_array, datapath, data_spacing, data_origin, data_direction, size_flag=None)
        assert len(data_array.shape) == 3
        data_after_clean = self.create_nonzero_mask_for_clean(data_array)  # [slice, 512, 512] the mask for each slice
        cleaned_data = self.place_air_zero(data_array, data_after_clean)  # [depth, 512-, 512-]
        # self.savefile(cleaned_data, datapath, data_spacing, data_origin, data_direction, size_flag=None)
        target_shape = self.target_shape
        if resize_flag==True:
            resize_data = self.resize_cleaned(cleaned_data, target_shape)
            size_flag = '_resize'
        elif depth_crop_flag==True:
            resize_data = self.crop_cleaned(cleaned_data, target_shape)
            size_flag = '_depthcrop'
        else:
            resize_data = cleaned_data
            size_flag = None
        self.savefile(resize_data, datapath, data_spacing, data_origin, data_direction, size_flag)


if __name__ == "__main__":
    # png_test()
    # path = 'R:\\AIMIRA\\AIMIRA_Database\\LUMC' # old dataset
    # target_cat_list = ['ATL', 'CSA', 'EAC']
    # for target_cat in target_cat_list:
        # path = 'D:\\ESMIRA\\ESMIRA_RApredictionUltra\\Wrist\\{}_Wrist_TRA'.format(target_cat)
    path = 'E:\\jjia\\data\\aimira\\TRT' 
    clean_code = PreClean(path=path, target_shape=[20, 512, 512])  # input the directory, 输入保存mha文件的路径
    clean_code.runner()
    print('finished')

