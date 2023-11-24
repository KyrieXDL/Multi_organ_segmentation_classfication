import torch
import numpy as np
import random
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
import glob
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
    Invertd,
    AsDiscreted,
    SaveImaged,
    Activationsd
)
from monai.data import CacheDataset, DataLoader, Dataset
import matplotlib.pylab as plt
import os
import SimpleITK as sitk
import pandas as pd
from transforms import transforms_dict
import copy


@torch.no_grad()
def momentum_update(model_pair, momentum):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data = param_m.data * momentum + param.data * (1. - momentum)


@torch.no_grad()
def copy_params(model_pair):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data.copy_(param.data)  # initialize
        param_m.requires_grad = False  # not update by gradient


def swa(model, model_dir, swa_start=1, swa_end=100):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    epochs = [15, 16, 17, 18, 19]
    model_path_list = os.listdir(model_dir)
    # model_path_list = [os.path.join(model_dir, f'model_ema_{e}.pt') for e in epochs]
    model_path_list = [os.path.join(model_dir, f'model_{e}.pt') for e in epochs]
    model_path_list += [os.path.join(model_dir, f'model_ema_{e}.pt') for e in [26, 27, 28, 29]]
    # model_path_list += [os.path.join(model_dir, f'model_ema_{e}.pt') for e in [12, 13, 14, 15, 16, 25, 26, 27, 28, 29]]
    model_path_list = sorted(model_path_list)
    print(model_path_list)

    # assert 0 <= swa_start < len(model_path_list) - 1, \
    #    f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list:
            print('swa: ', _ckpt)
            # logger.info(f'Load model from {_ckpt}')
            checkpoint = torch.load(_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint)
            # model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    return swa_model


# set random seed
def set_seed(seed, base=0, is_set=True):
    seed += base
    assert seed >= 0, '{} >= {}'.format(seed, 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_logger(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def collate_fn(batch):
    # image = torch.stack([d['image'] for d in batch], dim=0)

    # raw_mask = torch.stack([d['mask'] for d in batch], dim=0)
    # label = torch.stack([d['label'] for d in batch], dim=0)
    # item = {
    #    'image': image
    # }

    image, mask = [], []
    for d in batch:
        if type(d) == list:
            for di in d:
                image.append(di['image'])
                raw_mask = di['mask']
                tmp_mask = torch.zeros_like(raw_mask).repeat(4, 1, 1, 1)
                tmp_mask[0][raw_mask[0] == 1] = 1  # liver
                tmp_mask[1][raw_mask[0] == 2] = 1  # spleen
                tmp_mask[2][raw_mask[0] == 3] = 1  # left kidney
                tmp_mask[3][raw_mask[0] == 4] = 1  # right kidney
                mask.append(tmp_mask)
        else:
            image.append(d['image'])
            raw_mask = d['mask']
            tmp_mask = torch.zeros_like(raw_mask).repeat(4, 1, 1, 1)
            tmp_mask[0][raw_mask[0] == 1] = 1  # liver
            tmp_mask[1][raw_mask[0] == 2] = 1  # spleen
            tmp_mask[2][raw_mask[0] == 3] = 1  # left kidney
            tmp_mask[3][raw_mask[0] == 4] = 1  # right kidney
            mask.append(tmp_mask)
    image = torch.stack(image, dim=0)
    mask = torch.stack(mask, dim=0)

    item = {
        'image': image,
        'mask': mask
    }

    # mask, label = [], []
    # is_test = True if ('mask' not in batch[0]) or ('label' not in batch[0]) else False

    # if not is_test and type(batch[0]['mask']) != str:
    #    for d in batch:
    #        raw_mask = d['mask']
    #        #print(torch.unique(raw_mask))
    #        tmp_mask = torch.zeros_like(raw_mask).repeat(4, 1, 1, 1)
    #        tmp_mask[0][raw_mask[0] == 1] = 1  # liver
    #        tmp_mask[1][raw_mask[0] == 2] = 1  # spleen
    #        tmp_mask[2][raw_mask[0] == 3] = 1  # left kidney
    #        tmp_mask[3][raw_mask[0] == 4] = 1  # right kidney
    #
    #        mask.append(tmp_mask)
    #    mask = torch.stack(mask, dim=0)
    #    item['mask'] = mask
    # if 'label' in batch[0]:
    #    for d in batch:
    #        raw_label = d['label']
    #        tmp_label = torch.LongTensor(raw_label)
    #        label.append(tmp_label)
    #    #mask = torch.stack(mask, dim=0)
    #    label = torch.stack(label, dim=0)
    #    #item['mask'] = mask
    #    item['label'] = label

    # if 'mask' in batch[0] and type(batch[0]['mask']) != str:
    #    item['raw_mask'] = torch.stack([d['mask'] for d in batch], dim=0)
    # item = {
    #    'image': image,
    #    'mask': mask,
    #    'label': label,
    #    'raw_mask': raw_mask
    # }

    # if 'case_name' in batch[0]:
    #    case_name = [d['case_name'] for d in batch]
    #    item['case_name'] = case_name

    # if 'feat' in batch[0]:
    #    feature = [torch.tensor(d['feat']) for d in batch]
    #    feature = torch.cat(feature, dim=0)
    #    item['feat'] = feature

    return item


def collate_organ_fn(batch):
    # image_liver = torch.stack([d['image_liver'] for d in batch], dim=0)
    # image_spleen = torch.stack([d['image_spleen'] for d in batch], dim=0)
    # image_left_kidney = torch.stack([d['image_left_kidney'] for d in batch], dim=0)
    # image_right_kidney = torch.stack([d['image_right_kidney'] for d in batch], dim=0)

    # item = {
    #    #'image_liver': image_liver,
    #    #'image_spleen': image_spleen,
    #    'image_left_kidney': image_left_kidney,
    #    'image_right_kidney': image_right_kidney
    # }

    item = {}

    for organ in ['image', 'image_liver', 'image_spleen', 'image_left_kidney', 'image_right_kidney']:
        if organ in batch[0].keys() and type(batch[0][organ]) != str and type(batch[0][organ]) != np.ndarray:
            item[organ] = torch.stack([d[organ] for d in batch], dim=0)

    for mask in ['mask', 'mask_liver', 'mask_spleen', 'mask_left_kidney', 'mask_right_kidney']:
        if mask in batch[0].keys() and type(batch[0][mask]) != str and type(batch[0][mask]) != np.ndarray:
            item[mask] = torch.stack([d[mask] for d in batch], dim=0)

    mask, label = [], []
    is_test = True if ('mask' not in batch[0]) or ('label' not in batch[0]) else False

    # if not is_test and type(batch[0]['mask']) != str:
    #    for d in batch:
    #        raw_mask = d['mask']
    #        #print(torch.unique(raw_mask))
    #        tmp_mask = torch.zeros_like(raw_mask).repeat(4, 1, 1, 1)
    #        tmp_mask[0][raw_mask[0] == 1] = 1  # liver
    #        tmp_mask[1][raw_mask[0] == 2] = 1  # spleen
    #        tmp_mask[2][raw_mask[0] == 3] = 1  # left kidney
    #        tmp_mask[3][raw_mask[0] == 4] = 1  # right kidney

    #        mask.append(tmp_mask)
    #    mask = torch.stack(mask, dim=0)
    #    item['mask'] = mask

    if 'label' in batch[0]:
        for d in batch:
            raw_label = d['label']
            tmp_label = torch.LongTensor(raw_label)
            label.append(tmp_label)
        label = torch.stack(label, dim=0)
        item['label'] = label

    # if 'mask' in batch[0] and type(batch[0]['mask']) != str:
    #    item['raw_mask'] = torch.stack([d['mask'] for d in batch], dim=0)

    if 'case_name' in batch[0]:
        case_name = [d['case_name'] for d in batch]
        item['case_name'] = case_name

    return item


def create_dataset(args, train_index=None, val_index=None):
    if args.phase in ['train', 'val']:
        ## read data: images, masks, csv
        image_dir = os.path.join(args.data_dir, args.phase, 'data')
        mask_dir = os.path.join(args.data_dir, args.phase, 'mask')

        data_path = os.path.join(args.data_dir, args.phase, 'train.csv')
        data = pd.read_csv(data_path)
        label_names = ['liver', 'spleen', 'left kidney', 'right kidney']

        data_dicts = []
        for image_name, label in zip(data['ID'].values, data[label_names].values):
            item = {"image": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "mask": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "label": label,
                    "case_name": image_name}

            if args.use_feat:
                item['feat'] = np.load(f'./feature/unet/{image_name}.npy')  # [:, :, :, 0, 0, 0]
            data_dicts.append(item)

        random.shuffle(data_dicts)
        if train_index is not None and val_index is not None:
            train_files = np.array(data_dicts)[train_index]
            val_files = np.array(data_dicts)[val_index]
        else:
            train_size = int(len(data_dicts) * args.train_size)
            train_files, val_files = data_dicts[:train_size], data_dicts[train_size:]
        print(f'train_size={len(train_files)}, val_size={len(val_files)}')

        post_transforms = Compose(
            [
                Invertd(
                    keys="pred",  # invert the `pred` data field, also support multiple fields
                    transform=transforms_dict[args.val_transform],
                    orig_keys="mask",  # get the previously applied pre_transforms information on the `img` data field,
                    # then invert `pred` based on this information. we can use same info
                    # for multiple fields, also support different orig_keys for different fields
                    # nearest_interp=True,  # don't change the interpolation mode to "nearest" when inverting transforms
                    ## to ensure a smooth output, then execute `AsDiscreted` transform
                    to_tensor=True,  # convert to PyTorch Tensor after inverting
                ),
                AsDiscreted(keys="pred", threshold=0.5),
                # SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False)
            ]
        )

        train_dataset = Dataset(data=train_files, transform=transforms_dict[args.train_transform])
        val_dataset = Dataset(data=val_files, transform=transforms_dict[args.val_transform])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=6, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=6)

        return train_loader, val_loader, post_transforms
    else:

        data_dicts = []
        for image_file in os.listdir(args.data_dir):
            item = {"image": os.path.join(args.data_dir, image_file),
                    "case_name": image_file.split('.')[0]}
            data_dicts.append(item)

        test_transforms = transforms_dict[args.test_transform]
        post_transforms = Compose(
            [
                Activationsd(keys='pred', sigmoid=True),
                Invertd(
                    keys="pred",  # invert the `pred` data field, also support multiple fields
                    transform=test_transforms,
                    orig_keys="image",
                    nearest_interp=False,
                    to_tensor=True,  # convert to PyTorch Tensor after inverting
                ),
                AsDiscreted(keys="pred", threshold=0.5),
                SaveImaged(keys="pred", output_dir="./model/tmp_data", output_postfix="mask", separate_folder=False,
                           resample=False)
            ]
        )

        test_dataset = Dataset(data=data_dicts, transform=test_transforms)
        # test_dataset = CacheDataset(data=data_dicts, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        return test_loader, post_transforms


def create_organ_dataset(args, train_index=None, val_index=None):
    if args.phase in ['train', 'val']:
        ## read data: images, masks, csv
        image_dir = os.path.join(args.data_dir, args.phase, 'data')
        mask_dir = os.path.join(args.data_dir, args.phase, 'mask')

        data_path = os.path.join(args.data_dir, args.phase, 'train.csv')
        data = pd.read_csv(data_path)
        label_names = ['liver', 'spleen', 'left kidney', 'right kidney']
        # label_names = ['left kidney', 'right kidney']

        data_dicts = []
        for image_name, label in zip(data['ID'].values, data[label_names].values):
            item = {"image": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "image_liver": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "image_spleen": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "image_left_kidney": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "image_right_kidney": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "mask": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "mask_liver": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "mask_spleen": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "mask_left_kidney": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "mask_right_kidney": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "label": label,
                    "case_name": image_name}
            data_dicts.append(item)
        random.shuffle(data_dicts)
        if train_index is not None and val_index is not None:
            train_files = np.array(data_dicts)[train_index]
            val_files = np.array(data_dicts)[val_index]
        else:
            train_size = int(len(data_dicts) * args.train_size)
            train_files, val_files = data_dicts[:train_size], data_dicts[train_size:]
        # train_size = int(len(data_dicts) * args.train_size)
        # train_files, val_files = data_dicts[:train_size], data_dicts[train_size:]
        print(f'train_size={len(train_files)}, val_size={len(val_files)}')

        train_dataset = Dataset(data=train_files, transform=transforms_dict[args.train_transform])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_organ_fn,
                                  num_workers=6, pin_memory=True)

        val_dataset = Dataset(data=val_files, transform=transforms_dict[args.val_transform])
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_organ_fn, num_workers=4,
                                pin_memory=True)

        return train_loader, val_loader, None
    else:

        data_dicts = []
        pred_mask_dir = './result/segment/'
        # pred_mask_dir = './segment_processed'
        for image_file in os.listdir(args.data_dir):
            case_name = image_file.split('.')[0]
            item = {"image": os.path.join(args.data_dir, image_file),
                    "image_liver": os.path.join(args.data_dir, image_file),
                    "image_spleen": os.path.join(args.data_dir, image_file),
                    "image_left_kidney": os.path.join(args.data_dir, image_file),
                    "image_right_kidney": os.path.join(args.data_dir, image_file),
                    "case_name": image_file.split('.')[0],
                    "mask": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz'),
                    "mask_liver": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz'),
                    "mask_spleen": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz'),
                    "mask_left_kidney": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz'),
                    "mask_right_kidney": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz')}
            data_dicts.append(item)

        post_transforms = None
        test_dataset = Dataset(data=data_dicts, transform=transforms_dict[args.test_transform])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_organ_fn, num_workers=4)

        return test_loader, post_transforms


def create_multi_organ_dataset(args):
    if args.phase == 'train':
        ## read data: images, masks, csv
        image_dir = os.path.join(args.data_dir, args.phase, 'data')
        mask_dir = os.path.join(args.data_dir, args.phase, 'mask')

        data_path = os.path.join(args.data_dir, args.phase, 'train.csv')
        data = pd.read_csv(data_path)
        label_names = ['liver', 'spleen', 'left kidney', 'right kidney']
        # label_names = ['left kidney', 'right kidney']

        data_dicts = []
        for image_name, label in zip(data['ID'].values, data[label_names].values):
            item = {"image": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "image_liver": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "image_spleen": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "image_left_kidney": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "image_right_kidney": os.path.join(image_dir, f'{image_name}.nii.gz'),
                    "mask": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "mask_liver": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "mask_spleen": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "mask_left_kidney": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "mask_right_kidney": os.path.join(mask_dir, f'{image_name}_mask.nii.gz'),
                    "label": label,
                    "case_name": image_name}
            data_dicts.append(item)
        random.shuffle(data_dicts)
        train_size = int(len(data_dicts) * 0.8)
        train_files, val_files = data_dicts[:train_size], data_dicts[train_size:]
        # print(val_files[:10])
        print(f'train_size={len(train_files)}, val_size={len(val_files)}')

        train_liver_dataset = Dataset(data=train_files, transform=transforms_dict['trans_liver'])
        train_spleen_dataset = Dataset(data=train_files, transform=transforms_dict['trans_spleen'])
        train_left_kidney_dataset = Dataset(data=train_files, transform=transforms_dict['trans_left_kidney'])
        train_right_kidney_dataset = Dataset(data=train_files, transform=transforms_dict['trans_right_kidney'])

        train_liver_loader = DataLoader(train_liver_dataset, batch_size=args.batch_size, shuffle=True,
                                        collate_fn=collate_organ_fn, num_workers=4)
        train_spleen_loader = DataLoader(train_spleen_dataset, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=collate_fn, num_workers=4)
        train_left_kidney_loader = DataLoader(train_left_kidney_dataset, batch_size=args.batch_size, shuffle=True,
                                              collate_fn=collate_fn, num_workers=4)
        train_right_kidney_loader = DataLoader(train_right_kidney_dataset, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=collate_fn, num_workers=4)

        val_liver_dataset = Dataset(data=val_files, transform=transforms_dict['trans_liver'])
        val_spleen_dataset = Dataset(data=val_files, transform=transforms_dict['trans_spleen'])
        val_left_kidney_dataset = Dataset(data=val_files, transform=transforms_dict['trans_left_kidney'])
        val_right_kidney_dataset = Dataset(data=val_files, transform=transforms_dict['trans_right_kidney'])

        val_liver_loader = DataLoader(val_liver_dataset, batch_size=1, shuffle=False, collate_fn=collate_organ_fn,
                                      num_workers=4)
        val_spleen_loader = DataLoader(val_spleen_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                       num_workers=4)
        val_left_kidney_loader = DataLoader(val_left_kidney_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                            num_workers=4)
        val_right_kidney_loader = DataLoader(val_right_kidney_dataset, batch_size=1, shuffle=False,
                                             collate_fn=collate_fn, num_workers=4)

        train_loader = [train_liver_loader, train_spleen_loader, train_left_kidney_loader, train_right_kidney_loader]
        val_loader = [val_liver_loader, val_spleen_loader, val_left_kidney_loader, val_right_kidney_loader]
        return train_loader, val_loader, None

    else:

        data_dicts = []
        pred_mask_dir = './output/submit/segment'
        pred_mask_dir = './segment_processed'
        for image_file in os.listdir(args.data_dir):
            case_name = image_file.split('.')[0]
            item = {  # "image": os.path.join(args.data_dir, image_file),
                "image_liver": os.path.join(args.data_dir, image_file),
                "image_spleen": os.path.join(args.data_dir, image_file),
                "image_left_kidney": os.path.join(args.data_dir, image_file),
                "image_right_kidney": os.path.join(args.data_dir, image_file),
                "case_name": image_file.split('.')[0],
                "mask": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz'),
                "mask_liver": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz'),
                "mask_spleen": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz'),
                "mask_left_kidney": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz'),
                "mask_right_kidney": os.path.join(pred_mask_dir, f'{case_name}_mask.nii.gz')}
            data_dicts.append(item)
        # random.shuffle(data_dicts)
        # test_transforms = transforms_dict[args.test_transform]
        post_transforms = None

        test_liver_dataset = Dataset(data=data_dicts, transform=transforms_dict['trans_liver'])
        test_spleen_dataset = Dataset(data=data_dicts, transform=transforms_dict['trans_spleen'])
        test_left_kidney_dataset = Dataset(data=data_dicts, transform=transforms_dict['trans_left_kidney'])
        test_right_kidney_dataset = Dataset(data=data_dicts, transform=transforms_dict['trans_right_kidney'])

        test_liver_loader = DataLoader(test_liver_dataset, batch_size=1, shuffle=False, collate_fn=collate_organ_fn,
                                       num_workers=4)
        test_spleen_loader = DataLoader(test_spleen_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                        num_workers=4)
        test_left_kidney_loader = DataLoader(test_left_kidney_dataset, batch_size=1, shuffle=False,
                                             collate_fn=collate_fn, num_workers=4)
        test_right_kidney_loader = DataLoader(test_right_kidney_dataset, batch_size=1, shuffle=False,
                                              collate_fn=collate_fn, num_workers=4)

        test_loader = [test_liver_loader, test_spleen_loader, test_left_kidney_loader, test_right_kidney_loader]

        return test_loader, post_transforms
