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
from sklearn.model_selection import StratifiedKFold, KFold
import utils
import warnings

warnings.filterwarnings('ignore')
import random

loss_fun = {
    'asl_loss': AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0),
    # 'multi_asl_loss': MultiAsymmetricLossOptimized(gamma_neg_list=[2, 2, 4, 4], gamma_pos_list=[0, 0, 0, 0]),
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


def train(model, momentum_model, dataloader, optimizer, scheduler, loss_dic, device, epoch, logger, args):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    for step, batch in tqdm(enumerate(dataloader)):
        images = [batch[organ].to(device) for organ in organ_names]
        labels = batch['label'].to(device)  # .to(torch.float)
        # labels = torch.stack(labels, dim=1).to(torch.float)
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
        # np.save(f'./organ_images/mask_{step}.npy', batch['raw_mask'].detach().clone().cpu().numpy())

        # raw_masks = [organ_batch['raw_mask'].to(device) for organ_batch in batch]
        # for i in range(4):
        #    tmp_mask = torch.zeros_like(raw_masks[i])
        #    tmp_mask[(raw_masks[i] == 0) | (raw_masks[i] == (i+1))] = 1
        #    images[i] = images[i] * tmp_mask.to(device)

        if args.cat_mask:
            images = torch.cat([images, batch['raw_mask'].to(device) / 4], dim=-1)

        if args.use_fp16:
            with torch.cuda.amp.autocast():
                output_classify = model(images)
                loss_classify = loss_dic['classify_loss'](output_classify, labels)
                loss = loss_classify / args.accumulate_steps
            scaler.scale(loss).backward()
        else:
            output_classify = model(images)
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
            if args.use_ema:
                with torch.no_grad():
                    utils.momentum_update([model, momentum_model], 0.999)

        if (step + 1) % 10 == 0:
            avg_loss = total_loss / (step + 1)
            lr = optimizer.param_groups[0].get('lr')
            logger.info(f'train epoch {epoch} lr={lr}  loss={avg_loss}')


def val(model, dataloader, loss_dic, device, logger, post_transforms, args):
    model.eval()
    avg_loss, cnt = 0, 0
    all_classify_pred, all_classify_target = [], []
    print('evaluation')
    for step, batch in tqdm(enumerate(dataloader)):
        images = [batch[organ].to(device) for organ in organ_names]
        labels = batch['label'].to(device).to(torch.float)
        # images = [organ_batch['image'].to(device) for organ_batch in batch]
        # labels = [organ_batch['label'][:, i].to(device) for i, organ_batch in enumerate(batch)]
        # labels = torch.stack(labels, dim=1).to(torch.float)
        # raw_masks = [organ_batch['raw_mask'].to(device) for organ_batch in batch]
        # for i in range(4):
        #    tmp_mask = torch.zeros_like(raw_masks[i])
        #    tmp_mask[(raw_masks[i] == 0) | (raw_masks[i] == (i+1))] = 1
        #    images[i] = images[i] * tmp_mask.to(device)
        # print(torch.unique(raw_masks[0]), torch.unique(raw_masks[1]), torch.unique(raw_masks[2]), torch.unique(raw_masks[3]))
        masks = [batch[organ].to(device) for organ in mask_names]

        if args.mask_filter:
            for i in range(len(images)):
                # tmp_mask = torch.zeros_like(raw_masks[i])
                # tmp_mask[(raw_masks[i] == 0) | (raw_masks[i] == (i+1))] = 1
                images[i] = images[i] * masks[i]

        if args.cat_mask:
            images = torch.cat([images, batch['raw_mask'].to(device) / 4], dim=-1)

        with torch.no_grad():
            output_classify = model(images)
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

    all_classify_pred = torch.cat(all_classify_pred, dim=0)
    all_classify_target = torch.cat(all_classify_target, dim=0)

    # submit_df = pd.DataFrame()
    # submit_df[['liver', 'spleen', 'left kidney', 'right kidney']] = all_classify_pred.cpu().numpy()
    # submit_df.to_csv('./val_result.csv', index=False)

    case_sen, case_f1, case_acc = case_sen_f1_acc(all_classify_pred, all_classify_target)
    organ_acc, organ_f1 = organ_acc_f1(all_classify_pred, all_classify_target)
    classify_score = 0.3 * case_sen + 0.2 * case_f1 + 0.2 * organ_acc + 0.2 * organ_f1 + 0.1 * case_acc
    avg_loss /= cnt
    logger.info(
        f'classify: loss={avg_loss} classify_score={round(classify_score.item(), 4)}, case_sen={round(case_sen.item(), 4)}, case_f1={round(case_f1.item(), 4)}, case_acc={round(case_acc.item(), 4)}, organ_f1={round(organ_f1.item(), 4)}, organ_acc={round(organ_acc.item(), 4)}')
    return classify_score


def train_one_fold(model, momentum_model, optimizer, scheduler, train_dataloader, val_dataloader, loss_dic, device,
                   logger,
                   args, fold_dir=''):
    best_score = 0
    best_model_path = ''
    for epoch in range(args.epochs):
        train(model, momentum_model, train_dataloader, optimizer, scheduler, loss_dic, device, epoch, logger, args)

        save_path = os.path.join(args.output_dir, args.save_name, fold_dir, f'model_{epoch}.pt')
        optim_save_path = os.path.join(args.output_dir, args.save_name, f'optimizer.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        torch.save({'optim': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()},
                   optim_save_path)

        score = val(model, val_dataloader, loss_dic, device, logger, None, args)
        if args.use_ema:
            val(momentum_model, val_dataloader, loss_dic, device, logger, None, args)
            torch.save(momentum_model.state_dict(), save_path.replace('model_', 'model_ema_'))

        if score > best_score:
            best_score = score
            best_model_path = save_path

    return best_model_path, best_score


def inference(model_list, dataloader, device, logger, post_transforms, args):
    for model in model_list:
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
            mean_prob = torch.zeros((len(case_ids), 4), device=device)
            for model in model_list:
                output_classify = model(images)
                output_classify = torch.sigmoid(output_classify)
                mean_prob += output_classify
            mean_prob /= len(model_list)
            # output_classify = torch.sigmoid(mean_prob)
            classify_pred = torch.zeros_like(mean_prob)
            classify_pred[mean_prob >= 0.5] = 1

        all_pred.append(classify_pred.detach().clone().cpu())
        all_ids += case_ids
        print(output_classify)
        print(classify_pred)
    all_pred = torch.cat(all_pred, dim=0)

    submit_df = pd.DataFrame()
    submit_df['ID'] = all_ids
    # submit_df[['liver', 'spleen', 'left kidney', 'right kidney']] = all_pred.numpy()
    submit_df[save_names] = all_pred.numpy()
    submit_df.to_csv('./result_organ.csv', index=False)


def create_model(device, args):
    # create model
    model = Universal_classify_model(img_size=(96, 96, 96),
                                     in_channels=1,
                                     out_channels=args.class_num,
                                     class_num=args.class_num,
                                     backbone='unet',
                                     encoding='word_embedding',
                                     task=args.task,
                                     frozen_backbone=args.frozen_backbone,
                                     split_patch=args.split_patch,
                                     task_type=args.task_type,
                                     use_text_prompt=args.use_text_prompt,
                                     share_weight=args.share_weight
                                     ).to(device)
    # Load pre-trained weights
    store_dict = OrderedDict()
    checkpoint = torch.load('./pretrained_model/unet.pth')
    load_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        if '.up_tr' in key or 'decode' in key:
            continue
        if 'organ_embedding' in key:
            # value = value[[5, 0, 2, 1]]
            value = value[organ_ids]
        if 'backbone' in key and not args.share_weight:
            store_dict[name.replace('backbone', 'backbone1')] = value.detach().clone()
            store_dict[name.replace('backbone', 'backbone2')] = value.detach().clone()
            store_dict[name.replace('backbone', 'backbone3')] = value.detach().clone()
        if 'GAP' in key and not args.share_weight:
            store_dict[name.replace('GAP', 'GAP1')] = value.detach().clone()
            store_dict[name.replace('GAP', 'GAP2')] = value.detach().clone()
            store_dict[name.replace('GAP', 'GAP3')] = value.detach().clone()

        store_dict[name] = value
    if args.phase != 'test':
        msg = model.load_state_dict(store_dict, strict=False)
        print('Use pretrained weights ', msg)

    # create optimizer and scheduler
    group_params = [{'params': [p for n, p in model.named_parameters() if 'classify' in n], 'lr': 1e-3},
                    {'params': [p for n, p in model.named_parameters() if 'classify' not in n], 'lr': 1e-4}]
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

        # optim_path = os.path.join(os.path.dirname(args.ckpt), 'optimizer.pt')
        # checkpoint = torch.load(optim_path)
        # optimizer.load_state_dict(checkpoint['optim'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # start_epoch = checkpoint['epoch'] + 1
    return model, optimizer, scheduler


def main(args):
    # init
    set_seed(args.seed)
    year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
    log_path = os.path.join(args.output_dir, args.save_name, f'{year}-{month}-{day}.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = create_logger(log_path)
    device = torch.device(args.device)
    logger.info(args)

    # train test
    if args.phase == 'train':
        kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        model_paths = []
        avg_score = 0
        loss_dic = {
            "classify_loss": loss_fun[args.loss_func].to(device)
        }
        for fold_i, (train_index, val_index) in enumerate(kfold.split(list(range(400)))):
            # create dataloader
            train_dataloader, val_dataloader, post_transforms = create_organ_dataset(args, train_index, val_index)
            # create model and optimizer
            model, optimizer, scheduler = create_model(device, args)
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
                                                          use_2d_encoder=False
                                                          ).to(device)
                utils.copy_params([model, momentum_model])
                momentum_model.eval()
            else:
                momentum_model = None
            # cross validation
            best_model_path, best_score = train_one_fold(model, momentum_model, optimizer, scheduler, train_dataloader,
                                                         val_dataloader, loss_dic, device, logger, args,
                                                         fold_dir=f'fold_{fold_i}')

            model_paths.append(best_model_path)
            avg_score += best_score / args.n_splits

        logger.info(f'avg score: {avg_score}')
        with open(os.path.join(args.output_dir, args.save_name, 'cv.txt'), 'w') as fw:
            fw.write(str(model_paths))
    elif args.phase == 'val':
        val(model, val_dataloader, device, logger)
    elif args.phase == 'test':
        # create dataloader
        test_dataloader, post_transforms = create_organ_dataset(args)
        with open(os.path.join(args.output_dir, args.save_name, 'cv.txt'), 'r') as fr:
            model_paths = eval(fr.readline())
        model_list = []
        # random.shuffle(model_paths)
        # model_paths = model_paths[:18]
        for mp in model_paths:
            args.ckpt = mp
            print(args.ckpt)
            model, optimizer, scheduler = create_model(device, args)
            model_list.append(model)

        inference(model_list, test_dataloader, device, logger, post_transforms, args)


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
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--share_weight', action='store_false', default=True)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--class_num', default=4, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--warmup_ratio', default=0.2, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--schedule_type', default='reduce', type=str)
    parser.add_argument('--task_type', default='single', type=str)
    parser.add_argument('--backbone', default='unet', type=str)
    parser.add_argument('--ckpt', default='', type=str)
    parser.add_argument('--mask_filter', action='store_true', default=False)
    parser.add_argument('--train_size', default=0.8, type=float)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
