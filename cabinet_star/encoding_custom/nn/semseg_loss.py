from torch import nn as nn
from torch.nn import functional as F

from encoding_custom.utils.swiftnet_utils import upsample


class SemsegCrossEntropy(nn.Module):
    def __init__(self, num_classes=19, ignore_id=-1):
        super(SemsegCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id

    def loss(self, y, t):
        if y.shape[2:4] != t.shape[1:3]:
            y = upsample(y, t.shape[1:3])
        return F.cross_entropy(y, target=t, ignore_index=self.ignore_id)

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss
