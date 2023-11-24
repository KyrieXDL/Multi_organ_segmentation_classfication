#import torch.utils.data
from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import SimpleITK
import SimpleITK as sitk
import torch
import numpy as np


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


class SegDataset(Dataset):
    def __init__(self, data_dir, phase='train'):
        super().__init__()
        self.phase = phase
        self.image_dir = os.path.join(data_dir, phase, 'data')
        self.mask_dir = os.path.join(data_dir, phase, 'mask')

        # self.image_names = [file.split('.')[0] for file in os.listdir(self.image_dir)]
        data_path = os.path.join(data_dir, phase, f'{phase}.csv')
        self.data = pd.read_csv(data_path)
        self.label_names = ['liver', 'spleen', 'left kidney', 'right kidney']
        self.size = [128, 128, 32]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data.iloc[index]['ID']
        image_path = os.path.join(self.image_dir, f'{image_name}.nii.gz')
        image = SimpleITK.ReadImage(image_path)
        image = resize_image_itk(image, self.size)
        image = SimpleITK.GetArrayFromImage(image)
        item = {'image': image/255} # (C, H, W)

        if self.phase != 'test':
            mask_path = os.path.join(self.mask_dir, f'{image_name}_mask.nii.gz')
            mask = SimpleITK.ReadImage(mask_path)
            mask = resize_image_itk(mask, self.size, sitk.sitkNearestNeighbor)
            mask = SimpleITK.GetArrayFromImage(mask)
            masks = []
            for i in range(32):
                tmp = np.where(mask == (i+1), np.ones_like(mask), np.zeros_like(mask))
                masks.append(tmp)
            mask = np.stack(masks, axis=0)
            item['mask'] = mask # (num_class, C, H, W)

            labels = self.data.iloc[index][self.label_names].values.tolist()
            item['label'] = torch.LongTensor(labels)

        return item


def collate_fn(batch):
    images = [d['image'] for d in batch]
    max_depth = max([image.shape[0] for image in images])

    batch_images, batch_masks = [], []
    for i in range(len(batch)):
        cur_depth = images[i].shape[0]
        width, height = images[i].shape[1:]
        padding_depth = max_depth - cur_depth

        padding_image = torch.zeros((padding_depth, width, height))
        image = torch.cat([torch.tensor(images[i]), padding_image], dim=0)
        batch_images.append(image)

        if 'mask' in batch[0].keys():
            padding_mask = torch.zeros((4, padding_depth, width, height))
            mask = torch.cat([torch.tensor(batch[i]['mask']), padding_mask], dim=0)
            batch_masks.append(mask)

    item = {
        'image': torch.stack(batch_images, dim=0)
    }

    if 'mask' in batch[0].keys():
        item['mask'] = torch.stack(batch_masks, dim=0)

    if 'label' in batch[0].keys():
        batch_label = torch.LongTensor([d['label'] for d in batch])
        item['label'] = batch_label
    print(item['image'].shape, item['mask'].shape, item['label'].shape)
    return item


if __name__ == '__main__':
    import numpy as np
    x = np.random.rand(4, 6)
    x = torch.tensor(x)
    print(x)
