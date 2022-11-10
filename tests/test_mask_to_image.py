import unittest
import torch
import numpy as np
from utils.mask_to_image import mask_to_image


class TestPredict(unittest.TestCase):

    def test_mask_to_image(self):

        # Input : the output of the predict_mask function : tensor of shape (h, w)
        # create a random tensor containing int values from 0 to 4
        mask = torch.randint(0, 4, (10, 10))
        predicted_image = mask_to_image(mask)

        # then
        self.assertEqual("<class 'PIL.Image.Image'>", str(type(predicted_image)))
        self.assertTrue(predicted_image.size == mask.shape)
        self.assertTrue(np.array(predicted_image).ndim == 3)
        self.assertTrue(np.array(predicted_image).shape[2] == 3)


if __name__ == '__main__':
    unittest.main()










