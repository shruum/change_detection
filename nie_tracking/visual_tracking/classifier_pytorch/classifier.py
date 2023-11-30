import torch

from infer_single_image import infer
from classifier_utils import find_classes, get_network


class Classifier:
    def __init__(self, net, model_path, dataset_type=None, data_dir="."):
        self.net = net
        self.model_path = model_path
        self.dataset_type = dataset_type
        self.classes, self.class_to_idx = find_classes(dataset_type, data_dir)
        self.num_classes = len(self.classes)
        self.model = self.init_network()

    def init_network(self):
        self.model = get_network(self.net, self.num_classes)
        checkpoint = torch.load(self.model_path)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        return self.model

    def infer_image(self, image, image_size, topk=1):
        return infer(self, image, image_size, topk)
