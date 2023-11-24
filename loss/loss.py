import torch
import torch.nn.functional as F
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2 * num / den
        dice_loss = 1 - dice_score

        dice_loss_avg = dice_loss[target[:, 0] != -1].sum() / dice_loss[target[:, 0] != -1].shape[0]

        return dice_loss_avg


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target):

        predict = F.sigmoid(predict)

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            # for organ in [0, 1, 2, 5]:
            for organ in [0, 1, 2, 3]:
                dice_loss = self.dice(predict[b, organ], target[b, organ])
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)

        return total_loss.sum() / total_loss.shape[0]


class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            # for organ in [0, 1, 2, 5]:
            for organ in [0, 1, 2, 3]:
                ce_loss = self.criterion(predict[b, organ], target[b, organ])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)

        return total_loss.sum() / total_loss.shape[0]


class OrganBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mask = torch.zeros((1, 32))
        # self.mask[:, [0, 1, 2, 5]] = 1
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, predict, target):
        b = predict.shape[0]
        prob = torch.sigmoid(predict)
        loss = self.criterion(prob, target)
        # loss = loss * self.mask.to(predict.device)
        # loss = torch.sum(loss) / torch.sum(self.mask.repeat(b, 1))

        return loss
