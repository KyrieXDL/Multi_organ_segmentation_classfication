import argparse
import os
from datetime import datetime
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from loss.loss import Multi_BCELoss, DiceLoss, OrganBCELoss
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from metrics import dice_score, jaccard, case_sen_f1_acc, organ_acc_f1, organ_acc_f1_dic
from mydataset.mydataset import SegDataset, collate_fn
from model.seg_model import SegModel
from model.Universal_model import Universal_model
from model.Universal_classify_modelV2 import Universal_classify_modelV2
from utils import set_seed, create_logger, create_dataset
from loss.asl_loss import AsymmetricLossOptimized
import pandas as pd
from collections import OrderedDict
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import warnings

warnings.filterwarnings('ignore')

loss_fun = {
    'asl_loss': AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0),
    'organ_bce_loss': OrganBCELoss(),
    'multi_bce_loss': Multi_BCELoss(num_classes=4),
    'dice_loss': DiceLoss(num_classes=4),
    'bce_loss': nn.BCEWithLogitsLoss()
}


def train(model, dataloader, optimizer, scheduler, loss_dic, device, epoch, logger, args):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    for step, batch in tqdm(enumerate(dataloader)):
        # images = batch['image'].to(device)#.to(torch.float)
        # masks = batch['mask'].to(device)
        # labels = batch['label'].to(device)
        torch.cuda.empty_cache()
        batch = {k: v.to(device) for k, v in batch.items() if k in ['image', 'mask', 'label', 'feat']}
        labels = batch['label'].to(torch.float)
        # print(batch['image'].shape)
        if args.cat_mask:
            images = torch.cat([images, batch['raw_mask'].to(device) / 4], dim=-1)

        if args.use_fp16:
            with torch.cuda.amp.autocast():
                output_classify = model(**batch)
                loss_classify = loss_dic['classify_loss'](output_classify, labels)
                loss = loss_classify / args.accumulate_steps
            scaler.scale(loss).backward()
        else:
            output_classify = model(**batch)
            loss_classify = loss_dic['classify_loss'](output_classify, labels)

            loss = loss_classify / args.accumulate_steps
            # print(f'total loss={loss.item()}, classify_loss={loss_classify.item()}')
            loss.backward()
        total_loss += loss.item()

        if (step + 1) % args.accumulate_steps == 0:
            if args.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if args.schedule_type != 'reduce':
                scheduler.step()

        if (step + 1) % 10 == 0:
            avg_loss = total_loss / (step + 1)
            lr = optimizer.param_groups[0].get('lr')
            logger.info(f'train epoch {epoch} lr={lr}  loss={avg_loss}')


def val(model, dataloader, loss_dic, device, logger, post_transforms, args):
    model.eval()
    avg_loss, cnt = 0, 0
    all_classify_pred, all_classify_target = [], []
    print('evaluation')
    for step, batch in enumerate(dataloader):
        # images = batch['image'].to(device)#.to(torch.float)
        # masks = batch['mask'].to(device)
        # labels = batch['label'].to(device)
        torch.cuda.empty_cache()
        batch = {k: v.to(device) for k, v in batch.items() if k in ['image', 'mask', 'label', 'feat']}
        labels = batch['label'].to(torch.float)

        if args.cat_mask:
            images = torch.cat([images, batch['raw_mask'].to(device) / 4], dim=-1)

        with torch.no_grad():
            output_classify = model(**batch)
            loss_classify = loss_dic['classify_loss'](output_classify, labels)

            output_classify = torch.sigmoid(output_classify)
            classify_pred = torch.zeros_like(output_classify)
            classify_pred[output_classify >= 0.5] = 1

        cnt += 1
        avg_loss += loss_classify.item()
        all_classify_pred.append(classify_pred.detach())
        all_classify_target.append(labels.detach())

    all_classify_pred = torch.cat(all_classify_pred, dim=0)
    all_classify_target = torch.cat(all_classify_target, dim=0)

    case_sen, case_f1, case_acc = case_sen_f1_acc(all_classify_pred, all_classify_target)
    organ_acc, organ_f1 = organ_acc_f1(all_classify_pred, all_classify_target)
    organ_metric = organ_acc_f1_dic(all_classify_pred, all_classify_target)
    classify_score = 0.3 * case_sen + 0.2 * case_f1 + 0.2 * organ_acc + 0.2 * organ_f1 + 0.1 * case_acc

    avg_loss /= cnt
    logger.info(
        f'classify: loss={avg_loss} classify_score={round(classify_score.item(), 4)}, case_sen={round(case_sen.item(), 4)}, case_f1={round(case_f1.item(), 4)}, case_acc={round(case_acc.item(), 4)}, organ_f1={round(organ_f1.item(), 4)}, organ_acc={round(organ_acc.item(), 4)}')
    print('metrics = ', organ_metric)
    return avg_loss


def inference(model, dataloader, device, logger, post_transforms):
    model.eval()
    print('inference')

    all_pred = []
    all_ids = []
    for step, batch in enumerate(dataloader):
        # images = batch['image'].to(device)#.to(torch.float)
        case_ids = batch['case_name']
        batch = {k: v.to(device) for k, v in batch.items() if k in ['image', 'mask', 'label', 'feat']}

        with torch.no_grad():
            output_classify = model(**batch)
            output_classify = torch.sigmoid(output_classify)
            classify_pred = torch.zeros_like(output_classify)
            classify_pred[output_classify >= 0.5] = 1
            # classify_pred = classify_pred[:, [0, 1, 2, 5]]

        all_pred.append(classify_pred.detach().clone().cpu())
        all_ids += case_ids

    all_pred = torch.cat(all_pred, dim=0)
    submit_df = pd.DataFrame()
    submit_df['ID'] = all_ids
    submit_df[['liver', 'spleen', 'left kidney', 'right kidney']] = all_pred.numpy()
    submit_df.to_csv('./output/submit/result.csv', index=False)


def main(args):
    # init
    set_seed(args.seed)
    year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
    log_path = os.path.join(args.output_dir, args.save_name, f'{year}-{month}-{day}.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = create_logger(log_path)
    device = torch.device(args.device)
    logger.info(args)
    ## create dataloader
    # train_dataloader, val_dataloader, post_transforms = create_dataset(args)

    # create model
    model = Universal_model(img_size=(96, 96, 96),
                            # model = Universal_classify_modelV2(img_size=(96, 96, 96),
                            in_channels=1,
                            out_channels=4,
                            class_num=4,
                            backbone=args.backbone,
                            encoding='word_embedding',
                            task=args.task,
                            frozen_backbone=args.frozen_backbone,
                            split_patch=args.split_patch,
                            # use_feat=args.use_feat
                            ).to(device)
    # Load pre-trained weights
    store_dict = OrderedDict()
    checkpoint = torch.load(args.pretrain)
    load_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        if 'organ_embedding' in key:
            value = value[[5, 0, 2, 1]]
        # if 'GAP' in key:
        #    continue
        if 'swinViT' in name or 'coder' in name:
            name = 'backbone.' + name
        store_dict[name] = value
    msg = model.load_state_dict(store_dict, strict=False)
    print('Use pretrained weights ', msg)

    # random_params = ['classify_head', 'GAP']
    random_params = ['classify_head']
    group_params = [
        {'params': [p for n, p in model.named_parameters() if any([rn in n for rn in random_params])], 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() if all([rn not in n for rn in random_params])], 'lr': 1e-5}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(group_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
    total_steps = args.epochs * 320
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)

    start_epoch = 0
    if os.path.exists(args.ckpt) and args.resume:
        # store_dict = model.state_dict()
        checkpoint = torch.load(args.ckpt)
        load_dict = checkpoint
        msg = model.load_state_dict(load_dict, strict=False)
        print('load ckpt ', msg)

        optim_path = os.path.join(os.path.dirname(args.ckpt), 'optimizer.pt')
        checkpoint = torch.load(optim_path)
        optimizer.load_state_dict(checkpoint['optim'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(start_epoch, checkpoint['epoch'])

    # train test
    if args.phase == 'train':
        # create dataloader
        train_dataloader, val_dataloader, post_transforms = create_dataset(args)

        loss_dic = {
            "classify_loss": loss_fun[args.loss_func].to(device)
        }
        for epoch in range(start_epoch, args.epochs):
            train(model, train_dataloader, optimizer, scheduler, loss_dic, device, epoch, logger, args)

            save_path = os.path.join(args.output_dir, args.save_name, f'model_{epoch}.pt')
            optim_save_path = os.path.join(args.output_dir, args.save_name, f'optimizer.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            torch.save({'optim': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()},
                       optim_save_path)
            print('epoch: ', epoch)
            if len(val_dataloader) > 0:
                torch.cuda.empty_cache()
                avg_loss = val(model, val_dataloader, loss_dic, device, logger, post_transforms, args)

                if args.schedule_type == 'reduce':
                    scheduler.step(avg_loss)
    elif args.phase == 'val':
        val(model, val_dataloader, device, logger)
    elif args.phase == 'test':
        # create dataloader
        test_dataloader, post_transforms = create_dataset(args)
        inference(model, test_dataloader, device, logger, post_transforms)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--seed', default=42, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--warmup_ratio', default=0.2, type=float)
    parser.add_argument('--accumulate_steps', default=1, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--data_dir', default='./output', type=str)
    parser.add_argument('--save_name', default='base_classify', type=str)
    parser.add_argument('--task', default='classify', type=str)
    parser.add_argument('--train_transform', default='trans1', type=str)
    parser.add_argument('--val_transform', default='trans1', type=str)
    parser.add_argument('--test_transform', default='trans4', type=str)
    parser.add_argument('--loss_func', default='organ_bce_loss', type=str)
    parser.add_argument('--split_patch', action='store_true', default=False)
    parser.add_argument('--frozen_backbone', action='store_true', default=False)
    parser.add_argument('--use_fp16', action='store_true', default=False)
    parser.add_argument('--cat_mask', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--use_feat', action='store_true', default=False)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--schedule_type', default='reduce', type=str)
    parser.add_argument('--ckpt', default='', type=str)
    parser.add_argument('--pretrain', default='./pretrained_model/unet.pth', type=str)
    parser.add_argument('--backbone', default='unet', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--train_size', default=0.8, type=float)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
