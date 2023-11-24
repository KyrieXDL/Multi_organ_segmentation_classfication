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
from metrics import dice_score, jaccard, case_sen_f1_acc, organ_acc_f1
from mydataset.mydataset import SegDataset, collate_fn
from model.seg_model import SegModel
from model.Universal_classify_model import Universal_classify_model
from utils import set_seed, create_logger, create_dataset, create_organ_dataset
from loss.asl_loss import AsymmetricLossOptimized
from loss.multi_asl_loss import MultiAsymmetricLossOptimized
import pandas as pd
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from collections import OrderedDict
import utils
import copy
import warnings

warnings.filterwarnings('ignore')

loss_fun = {
    'asl_loss': AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0),
    'multi_asl_loss': MultiAsymmetricLossOptimized(gamma_neg_list=[2, 2, 4, 4], gamma_pos_list=[0, 0, 0, 0],
                                                   clip_list=[0.05, 0.05, 0.05, 0.05]),
    'organ_bce_loss': OrganBCELoss(),
    'multi_bce_loss': Multi_BCELoss(num_classes=4),
    'dice_loss': DiceLoss(num_classes=4),
    'bce_loss': nn.BCEWithLogitsLoss()
}

organ_names = ['image_liver', 'image_spleen', 'image_left_kidney', 'image_right_kidney']
# organ_names = ['image_left_kidney', 'image_right_kidney']
mask_names = ['mask_liver', 'mask_spleen', 'mask_left_kidney', 'mask_right_kidney']
organ_ids = [5, 0, 2, 1]
# organ_ids = [2, 1]
save_names = ['liver', 'spleen', 'left kidney', 'right kidney']


# save_names = ['left kidney', 'right kidney']


def train(model, dataloader, optimizer, scheduler, loss_dic, device, epoch, logger, args, momentum_model):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    for step, batch in tqdm(enumerate(dataloader)):
        images = [batch[organ].to(device) for organ in organ_names]
        labels = batch['label'].to(device)  # .to(torch.float)
        # labels = torch.stack(labels, dim=1).to(torch.float)
        masks = [batch[organ].to(device) for organ in mask_names]
        # print(images[0].shape, labels.shape)
        if args.mask_filter:
            for i in range(len(images)):
                images[i] = images[i] * masks[i] + images[i] * ((1 - masks[i]) * 0.65)

        item = {'organ_x': images}
        if args.use_global:
            item['global_x'] = batch['image']
        # np.save(f'./organ_images/train_liver_{step}.npy', images[0].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_spleen_{step}.npy', images[1].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_left_kidney_{step}.npy', images[2].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_right_kidney_{step}.npy', images[3].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_mask_liver_{step}.npy', masks[0].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_mask_spleen_{step}.npy', masks[1].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_mask_left_kidney_{step}.npy', masks[2].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_mask_right_kidney_{step}.npy', masks[3].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/mask_{step}.npy', batch['raw_mask'].detach().clone().cpu().numpy())

        # raw_masks = [organ_batch['raw_mask'].to(device) for organ_batch in batch]
        # for i in range(4):
        #    tmp_mask = torch.zeros_like(raw_masks[i])
        #    tmp_mask[(raw_masks[i] == 0) | (raw_masks[i] == (i+1))] = 1
        #    images[i] = images[i] * tmp_mask.to(device)

        if args.cat_mask:
            for i in range(len(images)):
                images[i] = torch.cat([images[i], masks[i]], dim=-1)
            # images = torch.cat([images, batch['raw_mask'].to(device)/4], dim=-1)

        if args.use_fp16:
            with torch.cuda.amp.autocast():
                output_classify = model(**item)
                loss_classify = loss_dic['classify_loss'](output_classify, labels)
                loss = loss_classify / args.accumulate_steps
            scaler.scale(loss).backward()
        else:
            # print('forward')
            output_classify = model(images)
            # print(output_classify.shape)
            # print('cal loss')
            loss_classify = loss_dic['classify_loss'](output_classify, labels)

            loss = loss_classify / args.accumulate_steps
            # print(f'total loss={loss.item()}, classify_loss={loss_classify.item()}')
            # print(loss.device)
            loss.backward()
            # print('backward')
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
            if args.use_ema:
                with torch.no_grad():
                    utils.momentum_update([model, momentum_model], args.momentum)

        if (step + 1) % 10 == 0:
            avg_loss = total_loss / (step + 1)
            lr = optimizer.param_groups[0].get('lr')
            logger.info(f"train epoch={epoch} learning rate={lr} loss={avg_loss}")


def val(model, dataloader, loss_dic, device, logger, post_transforms, args):
    model.eval()
    avg_loss, cnt = 0, 0
    all_classify_pred, all_classify_target = [], []
    all_ids = []
    print('evaluation')
    for step, batch in tqdm(enumerate(dataloader)):
        images = [batch[organ].to(device) for organ in organ_names]
        labels = batch['label'].to(device).to(torch.float)
        masks = [batch[organ].to(device) for organ in mask_names]
        case_ids = batch['case_name']
        # for img in images:
        #    print(img.shape)
        #    continue
        if args.mask_filter:
            for i in range(len(images)):
                images[i] = images[i] * masks[i] + images[i] * ((1 - masks[i]) * 0.65)

        if args.cat_mask:
            for i in range(len(images)):
                images[i] = torch.cat([images[i], masks[i]], dim=-1)
            # images = torch.cat([images, batch['raw_mask'].to(device)/4], dim=-1)

        item = {'organ_x': images}
        if args.use_global:
            item['global_x'] = batch['image']

        with torch.no_grad():
            output_classify = model(**item)
            loss_classify = loss_dic['classify_loss'](output_classify, labels)

            output_classify = torch.sigmoid(output_classify)
            classify_pred = torch.zeros_like(output_classify)
            classify_pred[output_classify >= 0.5] = 1
        # if cnt >5:
        #    break
        cnt += 1
        avg_loss += loss_classify.item()
        all_classify_pred.append(classify_pred.detach())
        all_classify_target.append(labels.detach())
        all_ids += case_ids

        # for i in range(len(classify_pred)):
        #    #continue
        #    pred = classify_pred[i].cpu().numpy()
        #    pred = ''.join([str(int(p)) for p in pred])
        #    target = labels[i].cpu().numpy()
        #    target = ''.join([str(int(p)) for p in target])
        #    np.save(f'./organ_images/image_{step}_{pred}_{target}.npy', batch['image'][i].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/mask_{step}_{pred}_{target}.npy', batch['mask'][i].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/val_liver_{step}.npy', images[0].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/val_spleen_{step}.npy', images[1].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/val_left_kidney_{step}.npy', images[2].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/val_right_kidney_{step}.npy', images[3].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/val_mask_liver_{step}.npy', masks[0].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/val_mask_spleen_{step}.npy', masks[1].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/train_mask_left_kidney_{step}.npy', masks[2].detach().clone().cpu().numpy())
        #    np.save(f'./organ_images/train_mask_right_kidney_{step}.npy', masks[3].detach().clone().cpu().numpy())

    all_classify_pred = torch.cat(all_classify_pred, dim=0)
    all_classify_target = torch.cat(all_classify_target, dim=0)

    # submit_df = pd.DataFrame()
    # submit_df['id'] = all_ids
    # submit_df[['liver', 'spleen', 'left kidney', 'right kidney']] = all_classify_pred.cpu().numpy()
    # submit_df[['liver_gt', 'spleen_gt', 'left kidney_gt', 'right kidney_gt']] = all_classify_target.cpu().numpy()
    # submit_df.to_csv('./val_result.csv', index=False)

    case_sen, case_f1, case_acc = case_sen_f1_acc(all_classify_pred, all_classify_target)
    organ_acc, organ_f1 = organ_acc_f1(all_classify_pred, all_classify_target)
    classify_score = 0.3 * case_sen + 0.2 * case_f1 + 0.2 * organ_acc + 0.2 * organ_f1 + 0.1 * case_acc
    avg_loss /= cnt
    logger.info(
        f'classify: loss={avg_loss} classify_score={round(classify_score.item(), 4)}, case_sen={round(case_sen.item(), 4)}, case_f1={round(case_f1.item(), 4)}, case_acc={round(case_acc.item(), 4)}, organ_f1={round(organ_f1.item(), 4)}, organ_acc={round(organ_acc.item(), 4)}')
    return avg_loss


def inference(model, dataloader, device, logger, post_transforms, args):
    model.eval()
    print('inference')

    all_pred = []
    all_ids = []
    # for step, batch in enumerate(dataloader):
    for step, batch in tqdm(enumerate(dataloader)):
        images = [batch[organ].to(device) for organ in organ_names]
        # images = [batch[organ].to(device) for organ in ['image_liver', 'image_spleen', 'image_left_kidney', 'image_right_kidney']]
        # images = [organ_batch['image'].to(device) for organ_batch in batch]
        # images = batch['image'].to(device).to(torch.float)
        case_ids = batch['case_name']
        masks = [batch[organ].to(device) for organ in mask_names]

        if args.mask_filter:
            for i in range(len(images)):
                # tmp_mask = torch.zeros_like(raw_masks[i])
                # tmp_mask[(raw_masks[i] == 0) | (raw_masks[i] == (i+1))] = 1
                images[i] = images[i] * masks[i]

        # np.save(f'./organ_images/train_liver_{step}.npy', images[0].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_spleen_{step}.npy', images[1].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_left_kidney_{step}.npy', images[2].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_right_kidney_{step}.npy', images[3].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_mask_liver_{step}.npy', masks[0].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_mask_spleen_{step}.npy', masks[1].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_mask_left_kidney_{step}.npy', masks[2].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/train_mask_right_kidney_{step}.npy', masks[3].detach().clone().cpu().numpy())

        # np.save(f'./organ_images/test_liver_{case_ids[0]}.npy', images[0].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/test_spleen_{case_ids[0]}.npy', images[1].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/test_left_kidney_{case_ids[0]}.npy', images[2].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/test_right_kidney_{case_ids[0]}.npy', images[3].detach().clone().cpu().numpy())
        # np.save(f'./organ_images/test_mask_{case_ids[0]}.npy', batch['raw_mask'].detach().clone().cpu().numpy())
        # case_ids = batch['case_name']
        # raw_masks = [organ_batch['raw_mask'].to(device) for organ_batch in batch]
        # for i in range(4):
        #    tmp_mask = torch.zeros_like(raw_masks[i])
        #    tmp_mask[(raw_masks[i] == 0) | (raw_masks[i] == (i+1))] = 1
        #    images[i] = images[i] * tmp_mask.to(device)
        # images = [batch[organ].to(device) for organ in organ_names]
        with torch.no_grad():
            output_classify = model(images)
            output_classify = torch.sigmoid(output_classify)
            classify_pred = torch.zeros_like(output_classify)
            classify_pred[output_classify >= 0.5] = 1
            # classify_pred = classify_pred[:, [0, 1, 2, 5]]

        all_pred.append(classify_pred.detach().clone().cpu())
        all_ids += case_ids
        print(output_classify)
        print(classify_pred)
    all_pred = torch.cat(all_pred, dim=0)

    submit_df = pd.DataFrame()
    submit_df['ID'] = all_ids
    # submit_df[['liver', 'spleen', 'left kidney', 'right kidney']] = all_pred.numpy()
    submit_df[save_names] = all_pred.numpy()
    submit_df.to_csv('./result/result.csv', index=False)


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
    model = Universal_classify_model(img_size=(96, 96, 96),
                                     in_channels=1,
                                     out_channels=args.class_num,
                                     class_num=args.class_num,
                                     backbone=args.backbone,
                                     encoding='word_embedding',
                                     task=args.task,
                                     frozen_backbone=args.frozen_backbone,
                                     split_patch=args.split_patch,
                                     task_type=args.task_type,
                                     use_text_prompt=args.use_text_prompt,
                                     share_weight=args.share_weight,
                                     use_2d_encoder=args.use_2d_encoder
                                     ).to(device)
    # Load pre-trained weights
    store_dict = OrderedDict()
    # checkpoint = torch.load('./pretrained_model/unet.pth')
    checkpoint = torch.load(args.pretrain, map_location='cpu')
    if 'net' in checkpoint:
        load_dict = checkpoint['net']
    elif 'state_dict' in checkpoint:
        load_dict = checkpoint['state_dict']
    else:
        load_dict = checkpoint

    for key, value in load_dict.items():
        if args.backbone == 'unet':
            name = '.'.join(key.split('.')[1:])
        else:
            name = key
        if '.up_tr' in key or 'decode' in key:
            continue
        if 'organ_embedding' in key:
            # value = value[[5, 0, 2, 1]]
            value = value[organ_ids]
        if 'swinViT' in name or ('coder' in name and 'swin' in args.pretrain):
            name = 'backbone.' + name

        if 'backbone' in name and not args.share_weight:
            store_dict[name.replace('backbone', 'backbone1')] = value.detach().clone()
            store_dict[name.replace('backbone', 'backbone2')] = value.detach().clone()
            store_dict[name.replace('backbone', 'backbone3')] = value.detach().clone()
        if 'GAP' in name and not args.share_weight:
            store_dict[name.replace('GAP', 'GAP1')] = value.detach().clone()
            store_dict[name.replace('GAP', 'GAP2')] = value.detach().clone()
            store_dict[name.replace('GAP', 'GAP3')] = value.detach().clone()

        store_dict[name] = value

    if args.use_2d_encoder:
        state_dict_2d = torch.load(args.pretrain_2d)
        for key, value in state_dict_2d.items():
            store_dict['encoder_2d.' + key] = value

    msg = model.load_state_dict(store_dict, strict=False)
    print('Use pretrained weights ', msg)

    random_names = ['classify']
    group_params = [
        {'params': [p for n, p in model.named_parameters() if any(rn in n for rn in random_names)], 'lr': 1e-3},
        {'params': [p for n, p in model.named_parameters() if not any(rn in n for rn in random_names)], 'lr': 1e-4}]
    # print('pretrained: ', [n for n, p in model.named_parameters() if not any(rn in n for rn in random_names)])
    # print('random: ', [n for n, p in model.named_parameters() if any(rn in n for rn in random_names)])
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(group_params)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(group_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

    start_epoch = 0
    if os.path.exists(args.ckpt) and args.resume:
        # store_dict = model.state_dict()
        checkpoint = torch.load(args.ckpt)
        load_dict = checkpoint
        msg = model.load_state_dict(load_dict, strict=False)
        print('load ckpt ', msg)

        optim_path = os.path.join(os.path.dirname(args.ckpt), 'optimizer.pt')
        checkpoint = torch.load(optim_path)
        # optimizer.load_state_dict(checkpoint['optim'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # start_epoch = checkpoint['epoch'] + 1

    # train test
    if args.phase == 'train':
        if args.use_ema:
            momentum_model = Universal_classify_model(img_size=(96, 96, 96),
                                                      in_channels=1,
                                                      out_channels=args.class_num,
                                                      class_num=args.class_num,
                                                      backbone=args.backbone,
                                                      encoding='word_embedding',
                                                      task=args.task,
                                                      frozen_backbone=args.frozen_backbone,
                                                      split_patch=args.split_patch,
                                                      task_type=args.task_type,
                                                      use_text_prompt=args.use_text_prompt,
                                                      share_weight=args.share_weight,
                                                      use_2d_encoder=args.use_2d_encoder
                                                      ).to(device)
            utils.copy_params([model, momentum_model])
            momentum_model.eval()
        else:
            momentum_model = None

        # create dataloader
        train_dataloader, val_dataloader, post_transforms = create_organ_dataset(args)

        loss_dic = {
            "classify_loss": loss_fun[args.loss_func].to(device)
        }
        for epoch in range(start_epoch, args.epochs):
            train(model, train_dataloader, optimizer, scheduler, loss_dic, device, epoch, logger, args, momentum_model)

            save_path = os.path.join(args.output_dir, args.save_name, f'model_{epoch}.pt')
            optim_save_path = os.path.join(args.output_dir, args.save_name, f'optimizer.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            torch.save({'optim': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()},
                       optim_save_path)
            if args.use_ema:
                torch.save(momentum_model.state_dict(), save_path.replace('model_', 'model_ema_'))

            if len(val_dataloader) > 0:
                avg_loss = val(model, val_dataloader, loss_dic, device, logger, post_transforms, args)
                if args.use_ema:
                    val(momentum_model, val_dataloader, loss_dic, device, logger, post_transforms, args)
                scheduler.step(avg_loss)
    elif args.phase == 'val':
        # create dataloader
        train_dataloader, val_dataloader, post_transforms = create_organ_dataset(args)
        val(model, val_dataloader, loss_dic, device, logger, post_transforms, args)
    elif args.phase == 'test':
        # if args.use_swa:
        #    swa_raw_model = copy.deepcopy(model)
        #    model_save_path = os.path.join(args.output_dir, args.save_name)
        #    model = utils.swa(swa_raw_model, model_save_path, swa_start=17, swa_end=20)
        # torch.save(model.state_dict(), './best_model/base_classify_organ_aug/model.pt')
        model_path = os.path.join(args.output_dir, args.save_name, f'model.pt')
        ckpt = torch.load(model_path)
        msg = model.load_state_dict(ckpt, strict=False)
        print(msg)
        # create dataloader
        test_dataloader, post_transforms = create_organ_dataset(args)
        inference(model, test_dataloader, device, logger, post_transforms, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--seed', default=42, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--accumulate_steps', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
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
    parser.add_argument('--use_text_prompt', action='store_true', default=False)
    parser.add_argument('--share_weight', action='store_false', default=True)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--class_num', default=4, type=int)
    parser.add_argument('--warmup_ratio', default=0.2, type=float)
    parser.add_argument('--momentum', default=0.999, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--schedule_type', default='reduce', type=str)
    parser.add_argument('--task_type', default='single', type=str)
    parser.add_argument('--ckpt', default='', type=str)
    parser.add_argument('--backbone', default='unet', type=str)
    parser.add_argument('--pretrain', default='./pretrained_model/unet.pth', type=str)
    parser.add_argument('--mask_filter', action='store_true', default=False)
    parser.add_argument('--use_global', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--use_swa', action='store_true', default=False)
    parser.add_argument('--use_2d_encoder', action='store_true', default=False)
    parser.add_argument('--pretrain_2d', default='./pretrained_model/resnet50.pth', type=str)
    parser.add_argument('--train_size', default=0.8, type=float)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
