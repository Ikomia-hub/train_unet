import unittest
import torch
import torch.nn.functional as F
from utils.dice_score import dice_coeff, multiclass_dice_coeff


class TestDiceScore(unittest.TestCase):
    """
    dice function input1: IA model prediction, input2:: the real mask image
    Input1: the model prediction is a tensor of predicted probabilities for each class, shape:(Batch_size, N_classes, h, w)
    input2: the real mask image of shape: (Batch_size, ch, h, w)

    Output: the score = 2TP/(2TP+FP+FN)
    """

    def test_multiclass_dice_on_random_tensor(self):
        # create a random input a tensor of probabilities, dim : (4, 10, 10)
        # 4 is the number of classes
        # create a random output: 2D tensor containing class mapped values [0, 1, 2, 3] of dim : (10, 10)
        input_tensor = torch.rand(4, 10, 10)
        # add a dim to the axe 0: the batch size
        input_tensor = torch.unsqueeze(input_tensor, dim=0)
        print('input size', input_tensor.shape)
        output_tensor = torch.randint(4, (10, 10))
        # add a dim to axe 0
        output_tensor = torch.unsqueeze(output_tensor, dim=0)
        print(output_tensor.shape)
        # extract indexes of classes with high probability prediction following the class dimension
        input_softmax = F.softmax(input_tensor.float(), dim=1).float()
        print('softmax', input_softmax.shape)
        # convert class indexes to one hot encoding format
        output_one_hot = F.one_hot(output_tensor, 4).permute(0, 3, 1, 2).float()
        print('one hot', output_one_hot.shape)

        # when
        dice_score = multiclass_dice_coeff(input_softmax, output_one_hot)
        print('dice', dice_score)

        # then
        self.assertEqual(input_softmax.shape, output_one_hot.shape)
        self.assertTrue(0 < dice_score < 1)


if __name__ == '__main__':
    unittest.main()
