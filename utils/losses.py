"""
Description
++++++++++++++++++++++
Addition losses module defines classses which are commonly used particularly in segmentation and are not part of standard pytorch library.

Usage
++++++++++++++++++++++
Import the package and Instantiate any loss class you want to you::

    from nn_common_modules import losses as additional_losses
    loss = additional_losses.DiceLoss()

    Note: If you use DiceLoss, insert Softmax layer in the architecture. In case of combined loss, do not put softmax as it is in-built

Members
++++++++++++++++++++++
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss


class DiceLoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target):
        output = F.softmax(output, dim=1)

        eps = 0.0001
        encoded_target = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)

        intersection = output * encoded_target
        intersection = intersection.sum(2).sum(2)

        num_union_pixels = output + encoded_target
        num_union_pixels = num_union_pixels.sum(2).sum(2)

        loss_per_class = 1 - ((2 * intersection) / (num_union_pixels + eps))

        return (loss_per_class.sum(1) / (num_union_pixels != 0).sum(1).float()).mean()


class CombinedLoss(_Loss):
    """
    A combination of dice and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.dice_loss = DiceLoss()

    def forward(self, input, target, class_weights=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param class_weights: torch.tensor (NxHxW)
        :return: scalar
        """
        y_2 = self.dice_loss(input, target)
        if class_weights is None:
            y_1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            y_1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), class_weights))
        return y_1 + y_2
