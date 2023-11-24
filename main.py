import argparse

from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from mydataset.mydataset import SegDataset
from model.Universal_model import Universal_model
from utils import set_seed, create_logger
import os
from datetime import datetime
import torch
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from loss.loss import Multi_BCELoss, DiceLoss
from typing import Any, Callable, List, Sequence, Tuple, Union
import torch.nn.functional as F
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple
import time


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    num_spatial_dims = len(inputs.shape) - 2
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        seg_prob = predictor(window_data, *args, **kwargs).to(device)  # batched patch segmentation

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            count_map[original_idx] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing]


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def train(model, dataloader, optimizer, scheduler, loss_criterion, device, logger):
    model.train()
    total_loss = 0
    for step, batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['label'].to(device)

        # output_segment, output_classify = model(images)
        # segment_loss = loss_criterion(output_segment, masks)
        # classify_loss = loss_criterion(output_classify, labels)
        # loss = (segment_loss + classify_loss) / args.accumulate_steps

        output_segment = model(images)
        segment_loss = loss_criterion(output_segment, masks)
        loss = (segment_loss + 0) / args.accumulate_steps

        loss.backward()
        total_loss += loss.item()

        if (step + 1) % args.accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % (args.accumulate_steps * 10) == 0:
            logger.info(f'train loss={total_loss / (step+1)}')


def val(model, dataloader, logger):
    model.eval()
    for batch in dataloader:
        images = batch['image'].unsqueeze(1)
        masks = batch['mask']
        labels = batch['label']

        with torch.no_grad():
            output_segment, output_classify = model(images)


def main(args):
    # init
    set_seed(args.seed)
    year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
    log_path = os.path.join(args.output_dir, args.save_name, f'{year}-{month}-{day}.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = create_logger(log_path)
    device = torch.device(args.device)

    # create dataset and dataloader
    dataset = SegDataset(args.data_dir, args.phase)
    train_size = 0.8
    train_set, val_set = random_split(dataset, lengths=[int(len(dataset) * train_size),
                                                        len(dataset) - int(len(dataset) * train_size)],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=6)

    # create model
    # model = SegModel()
    model = Universal_model(img_size=(96, 96, 96),
                            in_channels=1,
                            out_channels=4,
                            backbone='unet',
                            encoding='word_embedding'
                            ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_steps = len(train_dataloader) * args.epochs
    if args.schedule_type == 'warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
                                                    num_training_steps=total_steps)
    elif args.schedule_type == 'poly':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
            lr_end=0,
            power=1,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)



    # train test
    if args.phase == 'train':
        loss_criterion = DiceLoss()
        for epoch in range(args.epochs):
            train(model, train_dataloader, optimizer, scheduler, loss_criterion, device, logger)
            val(model, val_dataloader, logger)

            save_path = os.path.join(args.output_dir, args.save_name, f'model_{epoch}.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
    elif args.phase == 'test':
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--seed', default=16, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--accumulate_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--data_dir', default='./output', type=str)
    parser.add_argument('--save_name', default='base', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
    torch.binary_cross_entropy_with_logits()

