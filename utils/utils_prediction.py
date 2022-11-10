import matplotlib.pyplot as plt
import PIL.Image
import torch
import os

# Input: predicted mask of the corresponding class to each pixel
# output : Numpy array
# mapping = {(
def mask_to_image(mask_pred, mapping):
    # remove added dim if exists
    mask_pred = mask_pred.squeeze(0)
    # the mapping values we have used during training
    rev_mapping = {mapping[k]: k for k in mapping}
    # create an empty image with 3 channels of shape : (3, h, w)
    pred_image = torch.zeros(3, mask_pred.size(0), mask_pred.size(1), dtype=torch.uint8)

    # replace predicted mask values with mapped values
    for k in rev_mapping:
        pred_image[:, mask_pred == k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)

    final_mask_pred = pred_image.permute(1, 2, 0).numpy()
    return final_mask_pred


