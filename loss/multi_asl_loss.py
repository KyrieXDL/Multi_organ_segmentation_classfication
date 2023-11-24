import torch
import torch.nn as nn


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        # self.mask = torch.zeros((1, 32))
        # self.mask[:, [0, 1, 2, 5]] = 1
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum(1).mean()
        # return -self.loss.sum()
        # loss = self.loss * self.mask.to(x.device)
        # loss = torch.sum(loss) / torch.sum(self.mask.repeat(len(x), 1))
        # return -loss


class MultiAsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg_list=[4, 4, 4, 4], gamma_pos_list=[1, 1, 1, 1], clip_list=[0.05, 0.05, 0.05, 0.05]):
        super().__init__()
        self.loss_fn_list = []
        for gamma_neg, gamma_pos, clip in zip(gamma_neg_list, gamma_pos_list, clip_list):
            loss_fn = AsymmetricLossOptimized(gamma_neg, gamma_pos, clip)
            self.loss_fn_list.append(loss_fn)

    def forward(self, x, y):
        total_loss = 0
        for i in range(4):
            loss = self.loss_fn_list[i](x[:, i:i + 1], y[:, i:i + 1])
            total_loss += loss
        return total_loss
