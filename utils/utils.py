from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
from monai.losses import DiceLoss, HausdorffDTLoss, FocalLoss, LogHausdorffDTLoss
import utils.surface_distance as surfdist

class DiceLogHDLoss(_Loss):
    def __init__(self, include_background=False, to_onehot_y=True, softmax=True, lambda_dice = 1.0, lambda_hausdorff = 0.2,) -> None:
        super().__init__()
        self.dice = DiceLoss(include_background=include_background,
                            to_onehot_y=to_onehot_y,
                            softmax=softmax)
        self.hausdorff = HausdorffDTLoss(include_background=include_background,
                                        to_onehot_y=to_onehot_y,
                                        softmax=softmax)
        self.lambda_dice = lambda_dice
        self.lambda_hausdorff = lambda_hausdorff
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dice_loss = self.dice(input, target)
        hausdorff_loss = self.hausdorff(input, target)
        total_loss = self.lambda_dice * dice_loss + self.lambda_hausdorff * hausdorff_loss
        return total_loss


class FocalLogHDLoss(_Loss):
    def __init__(self, include_background=False, to_onehot_y=True, softmax=True, lambda_dice = 1.0, lambda_hausdorff = 0.1,) -> None:
        super().__init__()
        self.focal = FocalLoss(include_background=include_background,
                                to_onehot_y=to_onehot_y)
        self.hausdorff = LogHausdorffDTLoss(include_background=include_background,
                                            to_onehot_y=to_onehot_y,
                                            softmax=softmax)
        self.lambda_dice = lambda_dice
        self.lambda_hausdorff = lambda_hausdorff
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        focal_loss = self.focal(input, target)
        hausdorff_loss = self.hausdorff(input, target)
        total_loss = self.lambda_dice * focal_loss + self.lambda_hausdorff * hausdorff_loss
        return total_loss


def dice(im1, im2, tid):
    im1=im1==tid
    im2=im2==tid
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc = 2. * intersection.sum() / (im1.sum() + im2.sum()+0.00001)
    return dsc

def jaccard(im1, im2, tid):
    im1=im1==tid
    im2=im2==tid
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    ji = intersection.sum() / (union.sum() + 0.00001)
    return ji

def rAVD(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)

    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')

    return abs((vol1 - vol2) / float(vol2))

def ASSD(mask_pred, mask_gt, spacing_mm):
    surf_dist  = surfdist.compute_surface_distances(mask_gt=mask_gt, mask_pred=mask_pred, spacing_mm=spacing_mm)
    avg_surf_dist  = surfdist.compute_average_surface_distance(surf_dist)
    avg_sym_surf_dist = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
    return avg_sym_surf_dist

def hausdorff_distance(mask_pred, mask_gt, spacing_mm):
    surf_dist  = surfdist.compute_surface_distances(mask_gt=mask_gt, mask_pred=mask_pred, spacing_mm=spacing_mm)
    hd_dist_95 = surfdist.compute_robust_hausdorff(surf_dist, 95)
    return hd_dist_95

def get_tp_fp_fn(pred, targ):
    intersection = np.logical_and(pred, targ)
    tp = intersection.sum()
    fp = pred.sum() - tp
    fn = targ.sum() - tp
    return tp, fp, fn

def precision(pred, targ):
    tp, fp, fn = get_tp_fp_fn(pred, targ)
    pre = tp / (tp + fp)
    return pre

def recall(pred, targ):
    tp, fp, fn = get_tp_fp_fn(pred, targ)
    rec = tp / (tp + fn)
    return rec


def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


# def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
#     """
#     net_output must be (b, c, x, y(, z)))
#     gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
#     if mask is provided it must have shape (b, 1, x, y(, z)))
#     :param net_output:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(net_output.size())))

#     shp_x = net_output.shape
#     shp_y = gt.shape

#     with torch.no_grad():
#         if len(shp_x) != len(shp_y):
#             gt = gt.view((shp_y[0], 1, *shp_y[1:]))

#         if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
#             # if this is the case then gt is probably already a one hot encoding
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(shp_x)
#             if net_output.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(net_output.device.index)
#             y_onehot.scatter_(1, gt, 1)

#     tp = net_output * y_onehot
#     fp = net_output * (1 - y_onehot)
#     fn = (1 - net_output) * y_onehot

#     if mask is not None:
#         tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
#         fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
#         fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

#     if square:
#         tp = tp ** 2
#         fp = fp ** 2
#         fn = fn ** 2

#     tp = sum_tensor(tp, axes, keepdim=False)
#     fp = sum_tensor(fp, axes, keepdim=False)
#     fn = sum_tensor(fn, axes, keepdim=False)

#     return tp, fp, fn


# class AsymLoss(nn.Module):
#     def __init__(self, include_background=False, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False, get_beta=False):
#         """
#         paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779
#         """
#         super(AsymLoss, self).__init__()
#         self.include_background = include_background
#         self.square = square
#         self.do_bg = do_bg
#         self.batch_dice = batch_dice
#         self.apply_nonlin = apply_nonlin
#         self.smooth = smooth
#         self.get_beta = get_beta
#         self.beta = 1.5

#     def forward(self, x, y, loss_mask=None):
#         if not self.include_background:
#             x = x[:, 1:]
#             y = y[:, 1:]
#         shp_x = x.shape

#         if self.batch_dice:
#             axes = [0] + list(range(2, len(shp_x)))
#         else:
#             axes = list(range(2, len(shp_x)))

#         if self.apply_nonlin is not None:
#             x = self.apply_nonlin(x)

#         tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)# shape: (batch size, class num)

#         # beta = 2* fn / (fn + fp)
#         beta = self.beta


#         weight = (beta**2)/(1+beta**2)
#         asym = (tp + self.smooth) / (tp + weight*fn + (1-weight)*fp + self.smooth)

#         if not self.do_bg:
#             if self.batch_dice:
#                 asym = asym[1:]
#             else:
#                 asym = asym[:, 1:]
#         asym = asym.mean()
#         if self.get_beta:
#             beta = beta.mean(dim=0)
#             return 1 - asym, beta
#         else:
#             return 1 - asym
