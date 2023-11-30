import numpy as np
from PIL import Image, ImageDraw

from overhead_structure_face_and_traffic_barrier.base_detector import BaseDetector

TARGET_CLASS = {
    3: "poles",
    4: "traffic_sign",
    10: "wall",
    14: "overhead_structure",
}


class OverheadStructureFaceDetector(BaseDetector):
    def __init__(
        self,
        segmenter,
        config,
        vpoints=None,
        imgsize=None,
        save=False,
        target_class=TARGET_CLASS.keys(),
    ):
        super(OverheadStructureFaceDetector, self).__init__(
            segmenter,
            "overhead_structure_face",
            target_class,
            vpoints=vpoints,
            imgsize=imgsize,
            save=save,
        )
        self.config = config
        self.mask, self.mask_sum_col = zip(
            *[self._get_mask(imgsize, vpoints[i]) for i in [0, 1]]
        )

        self.y_axis = [np.arange(self.mask_sum_col[i].shape[0]) for i in [0, 1]]

    def _get_mask(self, img_size=(1080, 1920), vpoint=(540, 920)):

        mask_height = vpoint[0] - 50
        circle_r = self._find_circle(
            vpoint[0] + self.config["dy"],
            vpoint[1],
            mask_height,
            vpoint[1] + self.config["dx_r"],
            0,
            self.config["right_margin"] * img_size[1],
        )
        circle_l = self._find_circle(
            vpoint[0] + self.config["dy"],
            vpoint[1],
            mask_height,
            vpoint[1] - self.config["dx_l"],
            0,
            self.config["left_margin"] * img_size[1],
        )
        mask = Image.new("1", img_size[::-1], 1)
        ImageDraw.Draw(mask).chord(circle_r, 0, 300, outline=0, fill=0)
        ImageDraw.Draw(mask).chord(circle_l, -180, 179, outline=0, fill=0)
        ImageDraw.Draw(mask).rectangle(
            [(0, mask_height), img_size[::-1]], outline=0, fill=0
        )
        mask = np.array(mask)
        mask_sum_col = 0.9 * np.sum(mask, axis=1) + 1e-6
        return mask, mask_sum_col

    @staticmethod
    def _find_circle(x1, y1, x2, y2, x3, y3):
        """
        Returns Bounding box of circle found by with 3 points
        """
        # Difference of values
        dx12 = x1 - x2
        dx13 = x1 - x3
        dy12 = y1 - y2
        dy13 = y1 - y3
        dy31 = y3 - y1
        dy21 = y2 - y1
        dx31 = x3 - x1
        dx21 = x2 - x1

        # Difference of squared values
        sx13 = x1 ** 2 - x3 ** 2
        sy13 = y1 ** 2 - y3 ** 2
        sx21 = x2 ** 2 - x1 ** 2
        sy21 = y2 ** 2 - y1 ** 2

        f = (sx13 * dx12 + sy13 * dx12 + sx21 * dx13 + sy21 * dx13) // (
            2 * (dy31 * dx12 - dy21 * dx13)
        )
        g = (sx13 * dy12 + sy13 * dy12 + sx21 * dy13 + sy21 * dy13) // (
            2 * (dx31 * dy12 - dx21 * dy13)
        )
        c = -(x1 ** 2) - y1 ** 2 - 2 * g * x1 - 2 * f * y1

        # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
        # where centre is (h = -g, k = -f) and
        # radius r as r^2 = h^2 + k^2 - c
        cx = -g
        cy = -f
        sqr_of_radius = cx * cx + cy * cy - c
        radius = float(np.sqrt(sqr_of_radius))
        topleft = (round(cy - radius), round(cx - radius))
        bottomright = (round(cy + radius), round(cx + radius))
        return topleft, bottomright

    def _calculate_mass(self, path, run):
        run2idx = {"run1": 0, "run2": 1}
        allclassesmap = self.segmenter.segment(path)
        target_map = self._extract_classes(allclassesmap)
        target_map[~self.mask[run2idx[run]]] = 0
        target_map_bool = target_map > 0  # currently class agnostic
        mass_col = np.sum(target_map_bool, axis=1) / self.mask_sum_col[run2idx[run]]
        mass = np.sum(mass_col)
        centroid = int(np.dot(self.y_axis[run2idx[run]], mass_col) / (mass + 1e-3))
        if self.save:
            self.save_detections(path, run, target_map_bool)
        return [mass, centroid]
