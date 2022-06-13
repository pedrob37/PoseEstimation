import os
from collections import OrderedDict
import nibabel as nib
from typing import Dict, Hashable, Mapping, Sequence, Union
import numpy as np
from monai.transforms.transform import MapTransform, Transform
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
)

from monai.utils.enums import TransformBackends
from typing import Optional
from monai.config import DtypeLike, KeysCollection
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.config.type_definitions import NdarrayOrTensor
import torch
from monai.utils import convert_data_type
import re


GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_dir(path):              # if folder does not exist, create it
    if not os.path.exists(path):
        os.mkdir(path)


def new_state_dict(file_name):
    state_dict = torch.load(file_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'module':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def create_path(some_dir):
    try:
        if not os.path.exists(some_dir):
            os.makedirs(some_dir)
    except FileExistsError:
        print(f"{some_dir} already exists!")
        pass


def save_img(image, affine, filename):
    nifti_img = nib.Nifti1Image(image, affine)
    if os.path.exists(filename):
        raise OSError("File already exists! Killing job")
    else:
        nib.save(nifti_img, filename)


def create_folds(some_list, train_split=0.8, val_split=0.1):
    # Deterministic! Shuffle outside of this scope
    some_list_len = len(some_list)
    output_train_list = some_list[:int(some_list_len * train_split)]
    output_val_list = some_list[int(some_list_len * train_split):int(some_list_len * (train_split + val_split))]
    output_inf_list = some_list[int(some_list_len * (train_split + val_split)):]
    return output_train_list, output_val_list, output_inf_list


class CoordConv(Transform):
    """
    Appends additional channels encoding coordinates of the input.
    Liu, R. et al. An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution, NeurIPS 2018.
    """

    def __init__(
        self,
        spatial_channels: Sequence[int],
    ) -> None:
        """
        Args:
            spatial_channels: the spatial dimensions that are to have their coordinates encoded in a channel and
                appended to the input. E.g., `(1,2,3)` will append three channels to the input, encoding the
                coordinates of the input's three spatial dimensions (0 is reserved for the channel dimension).
        """
        self.spatial_channels = spatial_channels

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: data to be transformed, assuming `img` is channel first.
        """
        if max(self.spatial_channels) > img.ndim - 1:
            raise ValueError(
                f"input has {img.ndim-1} spatial dimensions, cannot add CoordConv channel for dim {max(self.spatial_channels)}."
            )
        if 0 in self.spatial_channels:
            raise ValueError("cannot add CoordConv channel for dimension 0, as 0 is channel dim.")

        # Correction
        batch_size_shape, dim_x, dim_y, dim_z = img.shape
        xx_range = torch.arange(dim_x, dtype=torch.int32) * (1 / (dim_x - 1))
        xx_range = xx_range * 2 - 1
        xx_range = xx_range[None, :, None, None]
        xx_channel = xx_range.repeat(1, 1, dim_y, dim_z)

        yy_range = torch.arange(dim_y, dtype=torch.int32) * (1 / (dim_x - 1))
        yy_range = yy_range * 2 - 1
        yy_range = yy_range[None, None, :, None]
        yy_channel = yy_range.repeat(1, dim_x, 1, dim_z)

        zz_range = torch.arange(dim_z, dtype=torch.int32) * (1 / (dim_x - 1))
        zz_range = zz_range * 2 - 1
        zz_range = zz_range[None, None, None, :]
        zz_channel = zz_range.repeat(1, dim_x, dim_y, 1)

        coord_channels = torch.cat([xx_channel, yy_channel, zz_channel], dim=0).numpy()

        return coord_channels


class CoordConvd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CoordConv`.
    """

    def __init__(self, keys: KeysCollection, spatial_channels: Sequence[int], allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
            spatial_channels: the spatial dimensions that are to have their coordinates encoded in a channel and
                appended to the input. E.g., `(1,2,3)` will append three channels to the input, encoding the
                coordinates of the input's three spatial dimensions. It is assumed dimension 0 is the channel.
        """
        super().__init__(keys, allow_missing_keys)
        self.coord_conv = CoordConv(spatial_channels)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.key_iterator(d):
            d["coords"] = self.coord_conv(d[key])
        return d

    @ staticmethod
    def normalise_images(array):
        import numpy as np
        return (array - np.min(array)) / (np.max(array) - np.min(array))


# Transforms
class ClipRange(Transform):
    """
    Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    When `b_min` or `b_max` are `None`, `scacled_array * (b_max - b_min) + b_min` will be skipped.
    If `clip=True`, when `b_min`/`b_max` is None, the clipping is not performed on the corresponding edge.

    Args:
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self,
            b_min: float,
            b_max: float,
            dtype: DtypeLike = np.float32,
    ) -> None:
        self.b_min = b_min
        self.b_max = b_max
        self.dtype = dtype

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        dtype = self.dtype or img.dtype
        img = clip(img, self.b_min, self.b_max)
        ret, *_ = convert_data_type(img, dtype=dtype)

        return ret


class ClipRanged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ClipRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ClipRange(b_min, b_max, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


def list_to_dict_reorganiser(input_dict, id_regex=re.compile(r'\d+')):
    varieties = ["heatmap", "PAF"]
    for variety in varieties:
        relevant_items = input_dict[variety]
        for variety_item in relevant_items:
            variety_id = id_regex.findall(variety_item)[-1]
            input_dict[f"{variety}_{variety_id}"] = variety_item
        del input_dict[variety]
    return input_dict
