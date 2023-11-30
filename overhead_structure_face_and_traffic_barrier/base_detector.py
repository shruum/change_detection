import os
from PIL import Image
from tqdm import tqdm
import numpy as np


class BaseDetector:
    def __init__(
        self, segmenter, task, target_class, vpoints=None, imgsize=None, save=False
    ):
        self.save = save
        self.segmenter = segmenter
        self.target_class = target_class

        self.cache_dir = os.path.join(self.segmenter.cache_dir, task)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.vpoints = vpoints
        self.target_color = Image.new("RGB", imgsize[::-1], (255, 0, 0))

    @staticmethod
    def _get_mask(img_size=(1080, 1920), vpoint=(540, 920)):
        NotImplementedError

    def _extract_classes(self, seg_map):
        target_map = np.zeros(seg_map.shape, dtype=np.uint8)
        for ele in self.target_class:
            target_map[seg_map == ele] = ele
        return target_map

    def _calculate_mass(self, path, run):
        NotImplementedError

    def calculate_mass(self, paths, run):
        if os.path.isfile(os.path.join(self.cache_dir, run + ".txt")):
            with open(os.path.join(self.cache_dir, run + ".txt"), "r") as f:
                mass = [ele.replace("\n", "").split(",") for ele in f.readlines()]
        else:
            mass = [
                self._calculate_mass(path, run)
                for path in tqdm(paths, total=len(paths), desc=run)
            ]
            with open(os.path.join(self.cache_dir, run + ".txt"), "w") as filehandle:
                for m, cm in mass:
                    filehandle.write(f"{m},{cm}\n")
        return np.array(mass, dtype=np.float)

    def save_detections(self, path, run, mask):
        fname = os.path.join(
            self.cache_dir, run, os.path.basename(path).replace(".jpg", ".png")
        )
        if not os.path.exists(os.path.join(self.cache_dir, run)):
            os.makedirs(os.path.join(self.cache_dir, run))
        img = Image.open(path).convert(mode="RGB")
        mask = 0.25 * 255.0 * mask
        tmp = Image.fromarray(np.uint8(mask), mode="L")

        out = Image.composite(self.target_color, img, tmp)
        out.convert(mode="RGB").save(fname)
