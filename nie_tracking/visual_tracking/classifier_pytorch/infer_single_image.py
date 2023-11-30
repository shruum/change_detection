# test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from conf import settings


def infer(classifier, img, img_size=44, topk=1):

    classes = np.array(classifier.classes)
    settings.NUM_CLASSES = len(classes)
    model = classifier.model
    model.eval()

    transform_test = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )
    image = transform_test(img)
    image = image.unsqueeze(0)

    output = model(image.cuda())
    output = nn.Softmax(dim=1)(output)
    score_tensor, class_tensor = output.topk(topk, 1, largest=True, sorted=True)

    class_list = classes[class_tensor.cpu().numpy()].tolist()[0]
    score_list = score_tensor.detach().cpu().numpy().tolist()[0]

    if topk == 1:
        return class_list[0], score_list[0]
    else:
        return class_list, score_list
