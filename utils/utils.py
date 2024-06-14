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
