import logging
import numpy as np
from PIL import Image

from overhead_structure_face_and_traffic_barrier.base_change_detector import (
    BaseChangeDetector,
)
from overhead_structure_face_and_traffic_barrier.overhead_structure_face_detector import (
    OverheadStructureFaceDetector,
)


class OverheadStructureFaceChangeDetector(BaseChangeDetector):
    def __init__(self, segmenter, config, vpoints=None, imgsize=None, save=False):
        super(OverheadStructureFaceChangeDetector, self).__init__(
            OverheadStructureFaceDetector,
            segmenter,
            config,
            vpoints=vpoints,
            imgsize=imgsize,
            save=save,
        )

        self.config = config

    def get_changes(self, run1_paths, run2_paths, run1_gps, run2_gps):
        logging.info("Segmenting Overhead Structure Face...")
        mass_cm = [
            self.detector.calculate_mass(paths, run)
            for paths, run in zip([run1_paths, run2_paths], ["run1", "run2"])
        ]

        run1_gps = np.array(run1_gps)
        run2_gps = np.array(run2_gps)
        gps = [run1_gps[:, :-1], run2_gps[:, :-1]]

        diffdist, cumdist = zip(*[self.get_distances(gps_coord) for gps_coord in gps])

        mass = [self.process_single_path(mass_cm[i][:, 0], diffdist[i]) for i in [0, 1]]

        idx = self.compare_paths(mass, gps, cumdist)
        return self.process_change(
            idx[0],
            idx[1],
            gps[0],
            gps[1],
            run1_paths,
            run2_paths,
            "Overhead Structure Face",
        )

    def compare_paths(self, mass, gps, cumdist, mode="frame"):
        r, s = self.get_shifts(gps[0], gps[1], cumdist[0], cumdist[1])
        if mode == "frame":
            mA = 1 * mass[0]
            mB = 1 * mass[1]
        elif mode == "dist":
            # x-axis in distance [homogenous separation]
            x = np.linspace(0, s, int(s))

            # interpolate form heterogenous separation to shifted homogenous
            mA = 1.0 * (np.interp(x, cumdist[0] + r, mass[0]) > 0)
            # interpolate form heterogenous separation to shifted homogenous
            mB = 1.0 * (np.interp(x, cumdist[1], mass[1]) > 0)

        # changed regions
        change = (mA - mB) * (
            self.windowcorrelation(mA, mB, 2 * self.config["gps_uncertainty"] + 1)
            < 0.01
        )

        # last x-position, in meters,  of change
        chng = (change[1:] - change[:-1]) * change[:-1] < 0

        # added change with gps uncertainity added in meters
        changes_add_dist = (
            np.where(chng * change[:-1] < 0)[0] - self.config["gps_uncertainty"]
        )

        # added change with gps uncertainity added in meters
        changes_rem_dist = (
            np.where(chng * change[:-1] > 0)[0] - self.config["gps_uncertainty"]
        )

        if mode == "dist":
            # find closest frame idx to the changing point
            idx1 = [
                (np.abs(cumdist[0] - rem_dist)).argmin()
                for rem_dist in changes_rem_dist
            ]

            # find closest frame idx to the changing point
            idx2 = [
                (np.abs(cumdist[1] - add_dist)).argmin()
                for add_dist in changes_add_dist
            ]
            return idx1, idx2
        elif mode == "frame":
            return changes_add_dist, changes_rem_dist

    def process_single_path(self, mass, diffdist):
        mass = self.convolve_with_rightexp(mass, ks=10, rep=1)
        mass /= np.max(mass)
        mass = self.schmitt_trigger(mass > 0.2, diffdist, 25)
        return self.medfilt(mass, 11) > 0


def detect_overhead_structure_changes(
    run1_paths, run2_paths, run1_gps, run2_gps, segmenter, vpoints=None,
):
    imgsize = Image.open(run2_paths[10]).convert("RGB").size[::-1]
    config = {
        "dx_l": 40,
        "dx_r": 20,
        "dy": 300,
        "left_margin": 0.35,
        "right_margin": 1 - 0.40,
        "gps_uncertainty": 75,
    }
    osf_change_detector = OverheadStructureFaceChangeDetector(
        segmenter,
        config,
        vpoints=[vpoints[0][::-1], vpoints[1][::-1]],
        imgsize=imgsize,
    )
    return osf_change_detector.get_changes(run1_paths, run2_paths, run1_gps, run2_gps)
