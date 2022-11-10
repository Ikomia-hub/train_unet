import torch
from torch import Tensor

"""
dice function input1: IA model prediction, input2:: the real mask image
Input1: the model prediction is a tensor of predicted probabilities for each class, shape:(Batch_size, N_classes, h, w)
input2: the real mask image of shape: (Batch_size, ch, h, w)

Output: the score = 2TP/(2TP+FP+FN)
"""


def dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))  # Will be zero if Truth=0 or Prediction=0
        sets_sum = torch.sum(input) + torch.sum(target) #  Will be zzero if both are 0
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)  # epsilon: We smooth our devision to avoid 0/0
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target)
