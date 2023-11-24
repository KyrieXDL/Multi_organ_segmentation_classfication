import argparse
from mydataset.mydataset import SegDataset, collate_fn
from model.seg_model import SegModel
from model.Universal_model import Universal_model
from utils import set_seed, create_logger, create_dataset
import os
from datetime import datetime
import torch
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from loss.loss import Multi_BCELoss, DiceLoss, OrganBCELoss
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from metrics import dice_score, jaccard, case_sen_f1_acc, organ_acc_f1
import SimpleITK as sitk
from monai.data import decollate_batch
import time
from postprocess import multiprocess_image


def train(model, dataloader, optimizer, scheduler, loss_dic, device, logger):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    for step, batch in tqdm(enumerate(dataloader)):
        images = batch['image'].to(device).to(torch.float)
        masks = batch['mask'].to(device)
        # labels = batch['label'].to(device)
        # np.save(f'./organ_images/image_{step}.npy', images.detach().clone().cpu().numpy())
        # np.save(f'./organ_images/mask_{step}.npy', masks.detach().clone().cpu().numpy())
        if args.use_fp16:
            with torch.cuda.amp.autocast():
                output_segment = model(images)
                # print(output_segment.shape, masks.shape)
                loss = loss_dic['segment_loss_1'](output_segment, masks) + loss_dic['segment_loss_2'](output_segment,
                                                                                                      masks)
                loss = loss / args.accumulate_steps
            scaler.scale(loss).backward()
        else:
            output_segment = model(images)
            print(output_segment.shape, masks.shape)
            loss = loss_dic['segment_loss_1'](output_segment, masks) + loss_dic['segment_loss_2'](output_segment, masks)
            loss = loss / args.accumulate_steps
            loss.backward()
        total_loss += loss.item()

        if (step + 1) % args.accumulate_steps == 0:
            if args.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % (args.accumulate_steps * 10) == 0:
            avg_loss = total_loss / (step + 1)
            logger.info(f'train loss={avg_loss}')
            # print('train loss=', avg_loss)


def val(model, dataloader, device, logger, post_transforms):
    model.eval()
    dice_score_list, jaccard_list = [], []
    all_classify_pred, all_classify_target = [], []
    print('evaluation')
    for step, batch in tqdm(enumerate(dataloader)):
        images = batch['image'].to(device)  # .half()#.to(torch.float)
        masks = batch['mask'].to(device)
        # labels = batch['label'].to(device)

        with torch.no_grad():
            # output_segment = sliding_window_inference(images, (96, 96, 64), 1, model, overlap=0.5, mode='gaussian')
            output_segment = model(images)
            output_segment = torch.sigmoid(output_segment)
            segment_mask = torch.zeros_like(output_segment)
            segment_mask[output_segment >= 0.5] = 1

        tmp_dice = dice_score(segment_mask.detach().cpu(), masks.detach().cpu())
        tmp_jac = jaccard(segment_mask.detach().cpu(), masks.detach().cpu())
        dice_score_list += tmp_dice
        jaccard_list += tmp_jac

        print(tmp_dice, tmp_jac)
        # if step < 1:
        #    for i in range(len(segment_mask)):
        #        tmp = segment_mask[i].detach().cpu().numpy()
        #        image = images[i].detach().cpu().numpy()
        #        mask = masks[i].detach().cpu().numpy()
        #        raw_mask = batch['mask'][i].numpy()
        #        np.save(f'./output_masks/{step}_pred.npy', tmp)
        #        np.save(f'./output_masks/{step}_image.npy', image)
        #        np.save(f'./output_masks/{step}_mask.npy', mask)
        #        np.save(f'./output_masks/{step}_raw_mask.npy', raw_mask)
    # all_classify_pred = torch.cat(all_classify_pred, dim=0)
    # all_classify_target = torch.cat(all_classify_target, dim=0)
    # case_sen, case_f1, case_acc = case_sen_f1_acc(all_classify_pred, all_classify_target)
    # organ_acc, organ_f1 = organ_acc_f1(all_classify_pred, all_classify_target)
    seg_dice_score = np.mean(dice_score_list)
    seg_jaccard = np.mean(jaccard_list)
    # classify_score = 0.3 * case_sen + 0.2 * case_f1 + 0.2 * organ_acc + 0.2 * organ_f1 + 0.1 * case_acc
    segment_score = 0.5 * seg_dice_score + 0.5 * seg_jaccard

    logger.info(
        f'segment: segment_score={round(segment_score.item(), 4)}, dice_score={round(seg_dice_score.item(), 4)}, jaccard={round(seg_jaccard.item(), 4)}')


def inference(model, dataloader, device, logger, post_transforms):
    model.eval()

    print('inference')
    all_data = []
    for step, batch in tqdm(enumerate(dataloader)):
        images = batch['image'].to(device)  # .to(torch.float)
        # case_ids = batch['case_name']
        if torch.cuda.is_available():
            images = images.half()
        print(images.shape)
        start = time.time()
        with torch.no_grad():
            output_segment = sliding_window_inference(images, (96, 96, 48), 1, model, overlap=0.5, mode='gaussian')
            # output_segment = output_segment[:, [0, 1, 2, 5]]
            # output_segment = torch.sigmoid(output_segment)
            # segment_mask = torch.zeros_like(output_segment)
            # segment_mask[output_segment >= 0.5] = 1
        end = time.time()
        print('cost', end - start)
        print('pred: ', output_segment.shape)
        batch['pred'] = output_segment
        # for i in decollate_batch(batch):
        #    print(i['pred'].shape)
        batch = [post_transforms(i) for i in decollate_batch(batch)]
        # seg_images = [d['pred'].cpu().detach().clone().numpy() for d in batch]
        # all_data += seg_images
        # if step > 20:
        #    break
        # multiprocess_image(seg_images)
        # print('post: ', batch[0]['pred'].shape, type(batch[0]['pred']))
        # for i in range(len(segment_mask)):
        #    tmp = segment_mask[i].detach().cpu().numpy()
        #    sitk.WriteImage(tmp, os.path.join('./output/submit/segment/', f'{case_ids[i]}_mask.nii.gz'))
        torch.cuda.empty_cache()
    multiprocess_image(all_data)


def main(args):
    # init
    set_seed(args.seed)
    year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
    log_path = os.path.join(args.output_dir, args.save_name, f'{year}-{month}-{day}.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = create_logger(log_path)
    device = torch.device(args.device)
    logger.info(args)
    print('device: ', device)
    # create model
    # model = SegModel()
    model = Universal_model(img_size=(96, 96, 96),
                            in_channels=1,
                            out_channels=4,
                            class_num=4,
                            backbone=args.backbone,
                            encoding='word_embedding',
                            task=args.task
                            ).to(device)
    # Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.pretrain, map_location='cpu')
    load_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        if 'organ_embedding' in key:
            value = value[[5, 0, 2, 1]]
        if 'swinViT' in name or 'coder' in name:
            name = 'backbone.' + name
        store_dict[name] = value
    msg = model.load_state_dict(store_dict, strict=False)
    print('Use pretrained weights ', msg)
    if args.phase == 'test' and args.device == 'cuda':
        model = model.half()
    if os.path.exists(args.ckpt):
        # store_dict = model.state_dict()
        checkpoint = torch.load(args.ckpt)
        load_dict = checkpoint
        msg = model.load_state_dict(load_dict, strict=False)
        print('Load ckpt ', msg)

    group_params = [{'params': [p for n, p in model.named_parameters() if 'classify_head' in n], 'lr': 1e-4},
                    {'params': [p for n, p in model.named_parameters() if 'classify_head' not in n], 'lr': 1e-5}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(group_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

    # train test
    if args.phase == 'train':
        # create dataset and dataloader
        train_dataloader, val_dataloader, post_transforms = create_dataset(args)

        loss_dic = {
            "segment_loss_1": Multi_BCELoss(num_classes=4).to(device),
            "segment_loss_2": DiceLoss(num_classes=4).to(device),
            "classify_loss": OrganBCELoss().to(device)
        }
        for epoch in range(args.epochs):
            train(model, train_dataloader, optimizer, scheduler, loss_dic, device, logger)

            save_path = os.path.join(args.output_dir, args.save_name, f'model_{epoch}.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            if len(val_dataloader) > 0:
                val(model, val_dataloader, device, logger, post_transforms)
    elif args.phase == 'val':
        val(model, val_dataloader, device, logger)
    elif args.phase == 'test':
        # create dataset and dataloader
        test_dataloader, post_transforms = create_dataset(args)
        inference(model, test_dataloader, device, logger, post_transforms)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--seed', default=16, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--accumulate_steps', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--data_dir', default='./output', type=str)
    parser.add_argument('--save_name', default='base', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--task', default='segment', type=str)
    parser.add_argument('--ckpt', default='', type=str)
    parser.add_argument('--backbone', default='unet', type=str)
    parser.add_argument('--pretrain', default='./pretrained_model/unet.pth', type=str)
    parser.add_argument('--train_transform', default='trans2', type=str)
    parser.add_argument('--val_transform', default='trans2', type=str)
    parser.add_argument('--test_transform', default='trans5', type=str)
    parser.add_argument('--use_fp16', action='store_true', default=False)
    parser.add_argument('--use_feat', action='store_true', default=False)
    parser.add_argument('--train_size', default=0.8, type=float)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
