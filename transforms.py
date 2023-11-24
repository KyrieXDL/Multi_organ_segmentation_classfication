from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    AddChanneld,
    Resized,
    RandZoomd,
    RandRotate90d,
    RandShiftIntensityd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    SpatialPadd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld
)
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import Method, NumpyPadMode, PytorchPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.config import IndexSelection, KeysCollection
from monai.transforms.transform import MapTransform, Randomizable
PadModeSequence = Union[Sequence[Union[NumpyPadMode, PytorchPadMode, str]], NumpyPadMode, PytorchPadMode, str]
import numpy as np
from copy import deepcopy
from scipy.ndimage import binary_fill_holes

z = 48
#z = 64

def image_crop(data, seg_data, val=0):
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    this_mask = data == val
    nonzero_mask = nonzero_mask | this_mask
    # nonzero_mask = binary_fill_holes(nonzero_mask)

    #outside_value = 0
    mask_voxel_coords = np.where(nonzero_mask == 1)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    cropped_seg = seg_data[resizer]
    cropped_data = data[resizer]
    return cropped_data, cropped_seg


class CropOrgand(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        mode: PadModeSequence = NumpyPadMode.CONSTANT,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.organ_to_id = {'image_liver': 1, 'image_spleen': 2, 'image_left_kidney': 3, 'image_right_kidney': 4}

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key, m in self.key_iterator(d, self.mode):
            if key in self.organ_to_id.keys():
                tmp, d[key] = image_crop(d[self.source_key], d[key], self.organ_to_id[key])
        d[self.source_key] = tmp
        return d


class Onehotd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.mask_names = ['mask_liver', 'mask_spleen', 'mask_left_kidney', 'mask_right_kidney']
        self.mask_to_id = {'mask_liver': 1, 'mask_spleen': 2, 'mask_left_kidney': 3, 'mask_right_kidney': 4}

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for new_key in self.mask_names:
            mask_tmp = np.zeros_like(d[self.source_key])
            mask_tmp[d[self.source_key] == self.mask_to_id[new_key]] = 1
            #mask_tmp = binary_fill_holes(mask_tmp)
            d[new_key] = mask_tmp

        return d


class CopyImaged(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.image_names = ['image_liver', 'image_spleen', 'image_left_kidney', 'image_right_kidney']

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for new_key in self.image_names:
            d[new_key] = deepcopy(d[self.source_key])

        return d


transforms1 = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 4),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        ToTensord(keys=["image", "mask", "label"]),
    ]
)


transforms2 = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="mask", select_fn=lambda x: x>0),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        #Resized(keys=["mask"], spatial_size=(96, 96, 48), mode=('nearest', )),
        ToTensord(keys=["image", "mask", "label"]),
    ]
)


transforms_image = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        ToTensord(keys=["image", "label"], allow_missing_keys=True),
    ]
)


transforms_image_swin = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        ToTensord(keys=["image", "label"], allow_missing_keys=True),
    ]
)


transforms_extract = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        Resized(keys=["image"], spatial_size=(192, 192, 96)),
        Resized(keys=["mask"], spatial_size=(192, 192, 96), mode=('nearest', )),
        ToTensord(keys=["image", "mask"]),
    ]
)


transforms_aug = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        #RandZoomd_select(keys=["image", "mask"], prob=0.3, min_zoom=1.3, max_zoom=1.5,
        #                     mode=['area', 'nearest']),
        RandRotate90d(
                keys=["image", "mask"],
                prob=0.10,
                max_k=3,
            ),
        RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        Resized(keys=["mask"], spatial_size=(96, 96, 48), mode=('nearest', )),
        ToTensord(keys=["image", "mask", "label"]),
    ]
)


transforms_segment = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 2),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="image"),
        SpatialPadd(keys=["image", "mask"], spatial_size=(96, 96, 96),
                        mode='constant'),
        RandZoomd(keys=["image", "mask"], prob=0.3, min_zoom=1.3, max_zoom=1.5,
                             mode=['area', 'nearest']),
        #RandCropByPosNegLabeld(
        #        keys=["image", "mask"],
        #        label_key="mask",
        #        spatial_size=(96, 96, 96),  # 192, 192, 64
        #        pos=2,
        #        neg=1,
        #        num_samples=2,
        #        image_key="image",
        #        image_threshold=0,
        #    ),
        RandCropByLabelClassesd(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=(96, 96, 96),  # 192, 192, 64
                ratios=[1, 4, 4, 4, 4],
                num_classes=5,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),  # 9
        RandRotate90d(
                keys=["image", "mask"],
                prob=0.10,
                max_k=3,
            ),
        RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
        #Resized(keys=["mask"], spatial_size=(192, 192, 96), mode=('nearest', )),
        ToTensord(keys=["image", "mask", "label"]),
    ]
)


transforms_test_classify = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        ToTensord(keys=["image"]),
    ]
)


transforms_test_segment = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        #Resized(keys=["image"], spatial_size=(192, 192, 96)),
        ToTensord(keys=["image"]),
    ]
)


transforms_liver = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="mask", select_fn=lambda x: x == 1, margin=2),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        Resized(keys=["mask"], spatial_size=(96, 96, 48), mode=('nearest', )),
        ToTensord(keys=["image", "mask", "label"], allow_missing_keys=True),
    ]
)


transforms_organ = Compose(
    [
        LoadImaged(keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask"]),
        AddChanneld(keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask"]),
        Orientationd(keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask"], axcodes="RAS"),
        #Spacingd(
        #    keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask"],
        #    pixdim=(1.5, 1.5, 1.5),
        #    mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
        #),
        ScaleIntensityRanged(
            keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image_liver"], source_key="mask", select_fn=lambda x: x == 1, margin=5),
        CropForegroundd(keys=["image_spleen"], source_key="mask", select_fn=lambda x: x == 2, margin=5),
        CropForegroundd(keys=["image_left_kidney"], source_key="mask", select_fn=lambda x: x == 3, margin=5),
        CropForegroundd(keys=["image_right_kidney"], source_key="mask", select_fn=lambda x: x == 4, margin=5),
        #CropForegroundd(keys=["mask"], source_key="mask", select_fn=lambda x: x == 1, margin=2),
        Resized(keys=["image_liver"], spatial_size=(96, 96, 48)),
        Resized(keys=["image_spleen"], spatial_size=(96, 96, 48)),
        Resized(keys=["image_left_kidney"], spatial_size=(96, 96, 48)),
        Resized(keys=["image_right_kidney"], spatial_size=(96, 96, 48)),
        #Resized(keys=["mask"], spatial_size=(96, 96, 48), mode=('nearest', )),
        ToTensord(keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask", "label"], allow_missing_keys=True),
    ]
)


transforms_organ_mask = Compose(
    [
        #LoadImaged(keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask"]),
        #AddChanneld(keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask"]),
        #Orientationd(keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask"], axcodes="RAS"),
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Onehotd(keys=["mask"], source_key="mask"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CopyImaged(keys=["image"], source_key="image"),
        CropForegroundd(keys=["image_liver", "mask_liver"], source_key="mask_liver", select_fn=lambda x: x == 1, margin=2),
        CropForegroundd(keys=["image_spleen", "mask_spleen"], source_key="mask_spleen", select_fn=lambda x: x == 1, margin=2),
        CropForegroundd(keys=["image_left_kidney", "mask_left_kidney"], source_key="mask_left_kidney", select_fn=lambda x: x == 1, margin=2),
        CropForegroundd(keys=["image_right_kidney", "mask_right_kidney"], source_key="mask_right_kidney", select_fn=lambda x: x == 1, margin=2),
        Resized(keys=["image_liver"], spatial_size=(96, 96, z)),
        Resized(keys=["image_spleen"], spatial_size=(96, 96, z)),
        Resized(keys=["image_left_kidney"], spatial_size=(96, 96, z)),
        Resized(keys=["image_right_kidney"], spatial_size=(96, 96, z)),
        Resized(keys=["mask_liver"], spatial_size=(96, 96, z), mode=('nearest', )),
        Resized(keys=["mask_spleen"], spatial_size=(96, 96, z), mode=('nearest', )),
        Resized(keys=["mask_left_kidney"], spatial_size=(96, 96, z), mode=('nearest', )),
        Resized(keys=["mask_right_kidney"], spatial_size=(96, 96, z), mode=('nearest', )),
        CropForegroundd(keys=["image", "mask"], source_key="mask"),
        Resized(keys=["image"], spatial_size=(96, 96, z)),
        Resized(keys=["mask"], spatial_size=(96, 96, z), mode=('nearest', )),
        ToTensord(keys=["image", "image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask_liver", "mask_spleen", "mask_left_kidney", "mask_right_kidney", "mask", "label"], allow_missing_keys=True),
    ]
)
#z = 32

transforms_organ_mask_aug = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Onehotd(keys=["mask"], source_key="mask"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CopyImaged(keys=["image"], source_key="image"),
        CropForegroundd(keys=["image_liver", "mask_liver"], source_key="mask_liver", select_fn=lambda x: x == 1, margin=2),
        CropForegroundd(keys=["image_spleen", "mask_spleen"], source_key="mask_spleen", select_fn=lambda x: x == 1, margin=2),
        CropForegroundd(keys=["image_left_kidney", "mask_left_kidney"], source_key="mask_left_kidney", select_fn=lambda x: x == 1, margin=2),
        CropForegroundd(keys=["image_right_kidney", "mask_right_kidney"], source_key="mask_right_kidney", select_fn=lambda x: x == 1, margin=2),
        Resized(keys=["image_liver"], spatial_size=(96, 96, z)),
        Resized(keys=["image_spleen"], spatial_size=(96, 96, z)),
        Resized(keys=["image_left_kidney"], spatial_size=(96, 96, z)),
        Resized(keys=["image_right_kidney"], spatial_size=(96, 96, z)),
        Resized(keys=["mask_liver"], spatial_size=(96, 96, z), mode=('nearest', )),
        Resized(keys=["mask_spleen"], spatial_size=(96, 96, z), mode=('nearest', )),
        Resized(keys=["mask_left_kidney"], spatial_size=(96, 96, z), mode=('nearest', )),
        Resized(keys=["mask_right_kidney"], spatial_size=(96, 96, z), mode=('nearest', )),
        RandRotate90d(keys=["image_liver", "mask_liver"], prob=0.10, max_k=3),
        RandRotate90d(keys=["image_spleen", "mask_spleen"], prob=0.10, max_k=3),
        RandRotate90d(keys=["image_left_kidney", "mask_left_kidney"], prob=0.10, max_k=3),
        RandRotate90d(keys=["image_right_kidney", "mask_right_kidney"], prob=0.10, max_k=3),
        RandFlipd(keys=["image_liver", "mask_liver"], prob=0.10,),
        RandFlipd(keys=["image_spleen", "mask_spleen"], prob=0.10,),
        RandFlipd(keys=["image_left_kidney", "mask_left_kidney"], prob=0.10,),
        RandFlipd(keys=["image_right_kidney", "mask_right_kidney"], prob=0.10,),
        RandShiftIntensityd(keys=["image_liver"], offsets=0.10, prob=0.20),
        RandShiftIntensityd(keys=["image_spleen"], offsets=0.10, prob=0.20),
        RandShiftIntensityd(keys=["image_left_kidney"], offsets=0.10, prob=0.20),
        RandShiftIntensityd(keys=["image_right_kidney"], offsets=0.10, prob=0.20),
        #CropForegroundd(keys=["image"], source_key="mask"),
        #Resized(keys=["image"], spatial_size=(96, 96, 48)),
        #RandRotate90d(keys=["image"], prob=0.10, max_k=3),
        #RandFlipd(keys=["image"], prob=0.10,),
        #RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        ToTensord(keys=["image_liver", "image_spleen", "image_left_kidney", "image_right_kidney", "mask_liver", "mask_spleen", "mask_left_kidney", "mask_right_kidney", "label"], allow_missing_keys=True),
    ]
)


transforms_organ_kidney = Compose(
    [
        LoadImaged(keys=["image_left_kidney", "image_right_kidney", "mask"]),
        AddChanneld(keys=["image_left_kidney", "image_right_kidney", "mask"]),
        Orientationd(keys=["image_left_kidney", "image_right_kidney", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image_left_kidney", "image_right_kidney", "mask"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image_left_kidney", "image_right_kidney"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        #CropForegroundd(keys=["image_liver"], source_key="mask", select_fn=lambda x: x == 1, margin=2),
        #CropForegroundd(keys=["image_spleen"], source_key="mask", select_fn=lambda x: x == 2, margin=2),
        CropForegroundd(keys=["image_left_kidney"], source_key="mask", select_fn=lambda x: x == 3, margin=5),
        CropForegroundd(keys=["image_right_kidney"], source_key="mask", select_fn=lambda x: x == 4, margin=5),
        #CropForegroundd(keys=["mask"], source_key="mask", select_fn=lambda x: x == 3, margin=5),
        #CropOrgand(keys=["image_left_kidney", "image_right_kidney"], source_key="mask"),
        Resized(keys=["image_liver"], spatial_size=(96, 96, 48)),
        Resized(keys=["image_spleen"], spatial_size=(96, 96, 48)),
        Resized(keys=["image_left_kidney"], spatial_size=(64, 64, 32)),
        Resized(keys=["image_right_kidney"], spatial_size=(64, 64, 32)),
        Resized(keys=["mask"], spatial_size=(64, 64, 32), mode=('nearest', )),
        ToTensord(keys=["image_left_kidney", "image_right_kidney", "mask", "label"], allow_missing_keys=True),
    ]
)


transforms_spleen = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="mask", select_fn=lambda x: x == 2, margin=2),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        Resized(keys=["mask"], spatial_size=(96, 96, 48), mode=('nearest', )),
        ToTensord(keys=["image", "mask", "label"], allow_missing_keys=True),
    ]
)


transforms_left_kidney = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="mask", select_fn=lambda x: x == 3, margin=5),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        Resized(keys=["mask"], spatial_size=(96, 96, 48), mode=('nearest', )),
        #Resized(keys=["image"], spatial_size=(64, 64, 32)),
        #Resized(keys=["mask"], spatial_size=(64, 64, 32), mode=('nearest', )),
        ToTensord(keys=["image", "mask", "label"], allow_missing_keys=True),
    ]
)


transforms_right_kidney = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0,
            b_max=1,
            clip=True
        ),
        CropForegroundd(keys=["image", "mask"], source_key="mask", select_fn=lambda x: x == 4, margin=5),
        Resized(keys=["image"], spatial_size=(96, 96, 48)),
        Resized(keys=["mask"], spatial_size=(96, 96, 48), mode=('nearest', )),
        #Resized(keys=["image"], spatial_size=(64, 64, 32)),
        #Resized(keys=["mask"], spatial_size=(64, 64, 32), mode=('nearest', )),
        ToTensord(keys=["image", "mask", "label"], allow_missing_keys=True),
    ]
)


transforms_dict = {
    'trans1': transforms1,
    'trans2': transforms2,
    'trans3': transforms_segment,
    'trans4': transforms_test_classify,
    'trans5': transforms_test_segment,
    'trans6': transforms_aug,
    'trans_liver': transforms_liver,
    'trans_spleen': transforms_spleen,
    'trans_left_kidney': transforms_left_kidney,
    'trans_right_kidney': transforms_right_kidney,
    'trans_extract': transforms_extract,
    'trans_image': transforms_image,
    'trans_image_swin': transforms_image_swin,
    'trans_organ': transforms_organ,
    'trans_organ_mask': transforms_organ_mask,
    'trans_organ_mask_aug': transforms_organ_mask_aug,
    'trans_organ_kidney': transforms_organ_kidney,
    'trans_segment': transforms_segment
}

