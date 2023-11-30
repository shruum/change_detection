# test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
import torch
from classes import get_class_dict
from torch.autograd import Variable

from conf import settings
from dataloader import get_test_dataloader, get_training_dataloader
from classifier_utils import get_network

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument(
        "-weights", type=str, required=True, help="the weights file you want to test"
    )
    parser.add_argument("-gpu", type=bool, default=True, help="use gpu or not")
    parser.add_argument(
        "-w", type=int, default=2, help="number of workers for dataloader"
    )
    parser.add_argument("-b", type=int, default=16, help="batch size for dataloader")
    parser.add_argument(
        "-s", type=bool, default=True, help="whether shuffle the dataset"
    )
    parser.add_argument(
        "-dataset_type",
        type=str,
        help="specify the dataset type to do the right class mapping",
    )
    parser.add_argument("-img_size", type=int, default=32, help="Size to resize images")

    args = parser.parse_args()

    test_loader = get_test_dataloader(
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s,
        dataset_type=args.dataset_type,
        img_size=args.img_size,
    )
    training_loader = get_training_dataloader(
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s,
        dataset_type=args.dataset_type,
        img_size=args.img_size,
    )
    classes, class_to_idx = get_class_dict(args.dataset_type)
    settings.NUM_CLASSES = len(classes)
    # num_class = len(test_loader.dataset.classes)
    num_class = len(classes)
    net = get_network(args.net)

    net.load_state_dict(torch.load(args.weights), args.gpu)
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    class_acc = torch.zeros(num_class)
    class_num = torch.zeros(num_class)
    correctmax = 0

    for n_iter, (image, label) in enumerate(test_loader):
        # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)
        _, predmax = output.max(1)
        correctmax += predmax.eq(label).float()

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        # compute top 5
        correct_5 += correct[:, :5].sum()

        # compute top1
        correct_1 += correct[:, :1].sum()
        class_acc[label] += correct[:, :1].sum()
        class_num[label] += 1

    res = [i / j for i, j in zip(class_acc, class_num)]
    for i in range(len(res)):
        print("per class acc ", i, res[i].item())
    print()
    print(len(test_loader.dataset))
    print("Accuracy: ", correct_1 / len(test_loader.dataset))
    print("Accuracy max: ", correctmax.float() / len(test_loader.dataset))

    # print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    # print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    # print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
