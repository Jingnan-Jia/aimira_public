import os
import random
from typing import Dict, Optional, Union, Hashable, Sequence

from medutils.medutils import load_itk, save_itk
import glob
import numpy as np
import pandas as pd
import torch
from monai.transforms import MapTransform
# from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomAffine
import SimpleITK as sitk



import re
from collections.abc import Callable, Hashable, Mapping
from copy import deepcopy
from typing import Any, Sequence, cast

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaObj, MetaTensor
from monai.data.utils import no_collation
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.traits import MultiSampleTrait, RandomizableTrait
from monai.transforms.transform import MapTransform, Randomizable, RandomizableTransform
from monai.transforms.utility.array import (
    AddCoordinateChannels,
    AddExtremePointsChannel,
    AsChannelLast,
    CastToType,
    ClassesToIndices,
    ConvertToMultiChannelBasedOnBratsClasses,
    CuCIM,
    DataStats,
    EnsureChannelFirst,
    EnsureType,
    FgBgToIndices,
    Identity,
    ImageFilter,
    IntensityStats,
    LabelToMask,
    Lambda,
    MapLabelValue,
    RemoveRepeatedChannel,
    RepeatChannel,
    SimulateDelay,
    SplitDim,
    SqueezeDim,
    ToCupy,
    ToDevice,
    ToNumpy,
    ToPIL,
    TorchVision,
    ToTensor,
    Transpose,
)
from monai.transforms.utils import extreme_points_to_image, get_extreme_points
from monai.transforms.utils_pytorch_numpy_unification import concatenate
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix, TraceKeys, TransformBackends
from monai.utils.type_conversion import convert_to_dst_type


TransInOut = Dict[Hashable, Optional[Union[np.ndarray, torch.Tensor, str, int]]]
# Note: all transforms here must inheritage Transform, Transform, or RandomTransform.



class Label_fusiond(MapTransform):
    """Load data. The output image values range from -1500 to 1500.

        #. Load data from `data['fpath_key']`;
        #. truncate data image to [-1500, 1500];
        #. Get origin, spacing;
        #. Calculate relative slice number;
        #. Build a data dict.

    Examples:
        :func:`ssc_scoring.mymodules.composed_trans.xformd_pos2score` and
        :func:`ssc_scoring.mymodules.composed_trans.xformd_pos`

    """
    def __init__(self, target, position, *args, **kwargs):
        self.position = position 
        self.target = target

    def __call__(self, data: TransInOut) -> TransInOut:
        data['label'] = []
        for targ in self.target.split('-'):
            data['label'].append(data[f"{self.position}_{targ}"])
                
        data['label'] = np.stack(data['label'], axis=0)

        return data


class Input_fusiond(MapTransform):
    """Load data. The output image values range from -1500 to 1500.
    """
    def __init__(self, input_position_code, nb_slices, *args, **kwargs):
        self.nb_slices = nb_slices
        self.input_position_code = input_position_code

    def __call__(self, data: TransInOut) -> TransInOut:
        data['input'] = []
        for position_name in self.input_position_code.split('-'):
            for view in [ 'TRA', 'COR']:
                start2end = data[f"{position_name}_{view}_fpath_slice"].split(':')[-1].split('plus')[-1]  # WR_TRA_fpath_slice
                start, end = map(int, start2end.split('to'))
                data['input'].append(data[f"img_{position_name}_{view}"][start: end])
                
                print(f"id: {data['TENR']}, ori_shape: {data[f'img_{position_name}_{view}'].shape}")
                del data[f'img_{position_name}_{view}']
                
        data['input'] = np.vstack(data['input'])
   
        return data

def normalize_intensity(volume):
        min_value = volume.min()
        max_value = volume.max()
        if max_value > min_value:
            out = (volume - min_value) / (max_value - min_value)
        else:
            out = volume
        # out_random = np.random.normal(0, 1, size=volume.shape)
        # out[volume == 0] = out_random[volume == 0]
        return out

class LoadDatad(MapTransform):
    """Load data. The output image values range from -1500 to 1500.

        #. Load data from `data['fpath_key']`;
        #. truncate data image to [-1500, 1500];
        #. Get origin, spacing;
        #. Calculate relative slice number;
        #. Build a data dict.

    Examples:
        :func:`ssc_scoring.mymodules.composed_trans.xformd_pos2score` and
        :func:`ssc_scoring.mymodules.composed_trans.xformd_pos`

    """
    def __init__(self, position_codes, data_image_dir):
        
 
            
        self.position_codes = position_codes
        self.data_image_dir = data_image_dir
        self.position_names = []
        code_name_dt = {'WR': 'Wrist',
                        'MC': 'MCP',
                        'MT': 'Foot'}
        for position_code in position_codes:
            self.position_names.append(code_name_dt[position_code])

    def __call__(self, data: TransInOut) -> TransInOut:
        for position_name, position_code in zip(self.position_names, self.position_codes):
            for view in ['COR', 'TRA']:
                fpath = glob.glob(self.data_image_dir + f"/clean_AIMIRA-LUMC-Treat{data['TENR']:04d}_TRT-*{position_name}_Post{view}*.mha")[0]
                # central_selector(fpath)
                # Target_cent = central_selector(Target_file)  # ':10to15'
                # Target_file = Target_file+Target_cent
                # IMG_path[f'IMG_{site}_{dirc}'].append(Target_file)

                # img = load_itk(fpath)
                data_mha = sitk.ReadImage(fpath)
                data_array = sitk.GetArrayFromImage(data_mha)
                img = normalize_intensity(data_array)  # [5, 512, 512]
                # img = normalize_intensity(img)
                data.update({f"fpath_{position_code}_{view}": fpath,
                            f"img_{position_code}_{view}": img})
  
        
        # world_pos = np.array(data['world_key']).astype(np.float32)
        # data_x = 
        # print('load a image')
        # x = data_x[0]  # shape order: z, y, x
        # print("cliping ... ")
        # x[x < -1500] = -1500
        # x[x > 1500] = 1500
        # x = self.normalize0to1(x)
        # scale data to 0~1, it's convinent for future transform (add noise) during dataloader

        # data_x_np = x.astype(np.float32)
        # data_y_np = y.astype(np.float32)


        return data


    
# class AddChanneld(MapTransform):
#     """Add a channel to the first dimension."""
#     def __init__(self, key='image_key'):
#         self.key = key

#     def __call__(self, data: TransInOut) -> TransInOut:
#         data[self.key] = data[self.key][None]
#         return data


# class RemoveTextd(MapTransform):
#     """
#     Remove the text to avoid the Error: TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <U80
#     """
#     def __init__(self, keys):
#         pass
#         # super().__init__(keys, allow_missing_keys=True)

#     def __call__(self, data: TransInOut) -> TransInOut:
#         d = data.copy()
#         for key in d:
#             if type(d[key]) is str:  # 1to8 is okay 
#                 del data[key] 
#         return data
    
    
    

class ToTensord(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    backend = ToTensor.backend

    def __init__(
        self,
        keys: KeysCollection,
        dtype = None,
        device = None,
        wrap_sequence = True,
        track_meta = None,
        allow_missing_keys = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: target data content type to convert, for example: torch.float, etc.
            device: specify the target device to put the Tensor data.
            wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
                E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.
            track_meta: if `True` convert to ``MetaTensor``, otherwise to Pytorch ``Tensor``,
                if ``None`` behave according to return value of py:func:`monai.data.meta_obj.get_track_meta`.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToTensor(dtype=dtype, device=device, wrap_sequence=wrap_sequence, track_meta=track_meta)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key,v in d.items():
            d[key] = self.converter(d[key])
            self.push_transform(d, key)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            # Remove the applied transform
            self.pop_transform(d, key)
            # Create inverse transform
            inverse_transform = ToNumpy()
            # Apply inverse
            d[key] = inverse_transform(d[key])
        return d



class CastToTyped(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CastToType`.
    """

    backend = CastToType.backend

    def __init__(
        self,
        dtype = np.float32,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of dtypes or torch.dtype,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        self.converter = CastToType()
        self.dtype = dtype

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, v in d.items(): 
            if type(v) != str:
                d[key] = self.converter(d[key], dtype=self.dtype)

        return d

class ToTensord(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    backend = ToTensor.backend

    def __init__(
        self,
        dtype = None,
        device = None,
        wrap_sequence: bool = True,
        track_meta = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: target data content type to convert, for example: torch.float, etc.
            device: specify the target device to put the Tensor data.
            wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
                E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.
            track_meta: if `True` convert to ``MetaTensor``, otherwise to Pytorch ``Tensor``,
                if ``None`` behave according to return value of py:func:`monai.data.meta_obj.get_track_meta`.
            allow_missing_keys: don't raise exception if key is missing.

        """
        self.converter = ToTensor(dtype=dtype, device=device, wrap_sequence=wrap_sequence, track_meta=track_meta)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in d:
            d[key] = self.converter(d[key])
        return d


if __name__ == '__main__':
    dataset = LoadDatad(position_codes=['WR'], data_image_dir='/exports/lkeb-hpc/jjia/project/project/aimira/aimira/data/images')
    a = dataset()
    print('yes')