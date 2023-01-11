import unittest
import PIL.Image
import numpy as np
from utils.my_dataset import My_dataset
class TestDataLoading(unittest.TestCase):

    def test_load(self):
        # assert
        img_path = 'image.png'
        # when
        image = My_dataset.load(img_path)
        # then
        expected_type = "<class 'PIL.Image.Image'>"
        expected_ndim = 3
        expected_channels = 3
        expected_values = [0, 255]

        self.assertEqual(expected_type, str(type(image)))
        self.assertEqual(expected_ndim, np.array(image).ndim)
        self.assertEqual(expected_channels, np.array(image).shape[2])
        self.assertTrue(np.amin(np.array(image)) >= expected_values[0])
        self.assertTrue(np.amax(np.array(image)) <= expected_values[1])

    def test_preprocess(self):
        # assert
        input_img = PIL.Image.open('image.png').convert('RGB')
        s = 0.5
        # when
        output_img = My_dataset.preprocess(input_img, scale=s, is_mask=False)
        # then
        expected_shape = (3, input_img.size[0]*s, input_img.size[1]*s)
        self.assertEqual(expected_shape, output_img.shape)
        self.assertTrue(output_img.shape[1] > 0)
        self.assertTrue(output_img.shape[2] > 0)

    def test_mask_to_class(self):
        """test on a image mask that contains all classes to check that all classes are mapped"""
        # assert:
        mask_image = PIL.Image.open("mask.png").convert('RGB')
        mask_image = np.array(mask_image)
        mask_image = mask_image.transpose((2, 0, 1))
        # when:
        # example: mapping a mask containing 3 classes
        mapped_mask = My_dataset.mask_to_class(mask_image, mapping= {(0, 0, 0): 0, (255, 0, 255): 1, (0, 255, 255): 2})
        # then:
        expected_shape = mask_image.shape[1:]
        expected_values = [0, 1, 2]

        self.assertEqual(expected_shape, mapped_mask.shape)

        # count occurrence of the mapping values, the sum of all occurrence must be == the number of values of the array
        occur_0 = (mapped_mask == 0).sum()
        occur_1 = (mapped_mask == 1).sum()
        occur_2 = (mapped_mask == 2).sum()

        expected_values = occur_0 + occur_1 + occur_2
        self.assertEqual(expected_values, mapped_mask.shape[0] * mapped_mask.shape[1])


if __name__ == "__main__":
    unittest.main()
