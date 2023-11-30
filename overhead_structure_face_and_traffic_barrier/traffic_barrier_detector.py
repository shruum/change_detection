import numpy as np
from PIL import Image, ImageDraw

from overhead_structure_face_and_traffic_barrier.base_detector import BaseDetector

TARGET_CLASS = {
    9: "guard_rail",
}


class TrafficBarrierDetector(BaseDetector):
    def __init__(
        self,
        segmenter,
        config,
        target_class=TARGET_CLASS.keys(),
        vpoints=None,
        imgsize=None,
        save=False,
    ):
        super(TrafficBarrierDetector, self).__init__(
            segmenter,
            "traffic_barrier",
            target_class,
            vpoints=vpoints,
            imgsize=imgsize,
            save=save,
        )
        self.config = config
        self.mask, self.mask_sum = zip(
            *[self._get_mask(imgsize, vpoints[i]) for i in [0, 1]]
        )
        self.weights = [self._get_weights(imgsize, vpoints[i]) for i in [0, 1]]

    def _get_mask(self, img_size=(1080, 1920), vpoint=(540, 920)):
        polygons = [
            (self.config["left_margin"] * img_size[1], vpoint[0]),
            (vpoint[1], 2 * img_size[0]),
            (self.config["right_margin"] * img_size[1], vpoint[0]),
            (
                (vpoint[0] * vpoint[1] + self.config["d"] * img_size[1])
                / (vpoint[0] + self.config["d"])
                - self.config["d"] / 2,
                vpoint[0],
            ),
            (vpoint[1], vpoint[0] + self.config["d"]),
            (
                (vpoint[0] * vpoint[1]) / (vpoint[0] + self.config["d"])
                + self.config["d"] / 2,
                vpoint[0],
            ),
        ]
        mask = Image.new("1", img_size[::-1], 0)
        ImageDraw.Draw(mask).polygon(polygons, outline=1, fill=1)
        mask = np.array(mask)
        return mask, np.sum(mask)

    def _get_weights(self, imgsize, vpoint):
        mu = vpoint[1]
        w_left = self._gaussian([0, mu], mu, mu / 3)
        w_right = self._gaussian([mu, imgsize[1]], mu, (imgsize[1] - mu) / 3)
        return imgsize[1] * np.concatenate([w_left, w_right])

    @staticmethod
    def _gaussian(x_range, mu, sigma):
        return np.exp(-0.5 * np.square((np.arange(*x_range) - mu) / sigma)) / (
            sigma * np.sqrt(2 * np.pi)
        )

    def _calculate_mass(self, path, run):
        run2idx = {"run1": 0, "run2": 1}
        map = self.segmenter.segment(path)
        target_map = self._extract_classes(map)
        target_map[~self.mask[run2idx[run]]] = 0
        target_map_bool = target_map > 0  # currently class agnostic
        mass_row = np.sum(target_map_bool, axis=0) / self.mask_sum[run2idx[run]]
        mass = 100 * np.dot(self.weights[run2idx[run]], mass_row)
        if self.save:
            self.save_detections(path, run, target_map)
        return mass, -1
