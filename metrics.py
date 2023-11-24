import numpy as np
import torch


def dice_score(seg_pred, seg_target):
    seg_pred = seg_pred.detach().cpu().numpy()
    seg_target = seg_target.detach().cpu().numpy()

    score_list = []
    for idx in range(len(seg_pred)):
        score = 0
        for organ in [0, 1, 2, 5]:
            pred = seg_pred[idx][organ]
            target = seg_target[idx][organ]
            tmp_score = (pred * target).sum() / (pred.sum() + target.sum() + 1e-12)
            score += tmp_score
        score /= 4
        score_list.append(score)

    return score_list


def jaccard(seg_pred, seg_target):
    seg_pred = seg_pred.detach().cpu().numpy()
    seg_target = seg_target.detach().cpu().numpy()

    score_list = []
    for idx in range(len(seg_pred)):
        score = 0
        for organ in [0, 1, 2, 5]:
            pred = seg_pred[idx][organ]
            target = seg_target[idx][organ]

            tp = (pred * target).sum()
            fp = (pred * (1-target)).sum()
            fn = ((1-pred) * target).sum()
            tmp_score = tp / (tp + fp + fn + 1e-12)

            score += tmp_score
        score /= 4
        score_list.append(score)

    return score_list


def case_sen_f1_acc(classify_pred, classify_target):
    classify_pred = classify_pred.detach().cpu().numpy()
    classify_target = classify_target.detach().cpu().numpy()

    classify_pred = classify_pred[:, [0, 1, 2, 5]].sum(1)
    classify_target = classify_target[:, [0, 1, 2, 5]].sum(1)

    classify_pred[classify_pred > 0] = 1
    classify_target[classify_target > 0] = 1

    tp = (classify_pred * classify_target).sum()
    fp = (classify_pred * (1-classify_target)).sum()
    fn = ((1-classify_pred) * classify_target).sum()
    tn = ((1-classify_pred) * (1-classify_target)).sum()

    sen = tp / (tp + fn + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    f1 = (2 * prec * sen) / (prec + sen + 1e-12)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-12)

    return sen, f1, acc


def organ_acc_f1(classify_pred, classify_target):
    classify_pred = classify_pred.detach().cpu().numpy()
    classify_target = classify_target.detach().cpu().numpy()

    classify_pred = classify_pred[:, [0, 1, 2, 5]]
    classify_target = classify_target[:, [0, 1, 2, 5]]

    tp = (classify_pred * classify_target).sum()
    fp = (classify_pred * (1 - classify_target)).sum()
    fn = ((1 - classify_pred) * classify_target).sum()
    tn = ((1 - classify_pred) * (1 - classify_target)).sum()

    sen = tp / (tp + fn + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    f1 = (2 * prec * sen) / (prec + sen + 1e-12)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-12)

    return acc, f1


# x = torch.randint(5, (9,))
# y = torch.randint(5, (9,))
# print(x)
# print(y)
# print(torch.sum(x == y))


if __name__ == '__main__':
    x = torch.tensor([[0, 1, 0, 1], [1, 1, 1, 1]])
    y = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]])

    # tmp = x - y
    # k = torch.sum(torch.sum(tmp == 0, dim=1) == 4)
    # print(k)
    # k = (x * y).sum(1)
    # k = torch.sum(k == y.sum(1))
    # N = len(y)
    # acc = k / N
    # print(acc)

    a = np.array([1, 0, 0, 0])
    b = np.array([1, 1, 0, 0])
    print((a == b).all())