import logging
import numpy as np
from PIL import Image

from overhead_structure_face_and_traffic_barrier.base_change_detector import (
    BaseChangeDetector,
)
from overhead_structure_face_and_traffic_barrier.traffic_barrier_detector import (
    TrafficBarrierDetector,
)


class TrafficBarrierChangeDetector(BaseChangeDetector):
    def __init__(
        self, segmenter, config, vpoints=None, imgsize=None, save=False,
    ):
        super(TrafficBarrierChangeDetector, self).__init__(
            TrafficBarrierDetector,
            segmenter,
            config,
            vpoints=vpoints,
            imgsize=imgsize,
            save=save,
        )
        self.config = config

    def get_changes(self, run1_paths, run2_paths, run1_gps, run2_gps):
        logging.info("Segmenting Traffic Barrier...")
        mass = [
            self.detector.calculate_mass(paths, run)[:, 0]
            for paths, run in zip([run1_paths, run2_paths], ["run1", "run2"])
        ]
        run1_gps = np.array(run1_gps)
        run2_gps = np.array(run2_gps)
        gps = [run1_gps[:, :-1], run2_gps[:, :-1]]

        diffdist, cumdist = zip(*[self.get_distances(gps_coord) for gps_coord in gps])

        mass = [self.process_single_path(mass[i], diffdist[i]) for i in [0, 1]]

        idx = self.compare_paths(mass, gps, cumdist)  # return frame indices of changes

        return self.process_change(
            idx[0], idx[1], gps[0], gps[1], run1_paths, run2_paths, "Traffic Barrier"
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
            self.windowcorrelation(mA, mB, 4 * self.config["gps_uncertainty"] + 1)
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
        # Averaging Smoothing
        mass = self.convolve_with_non_linearity(mass, ks=5, rep=10)
        # Remove sudden changes(salt and pepper noise)
        mass_bool = self.medfilt(mass, 31) > 0.1

        # Adding hysteresis i.e., ignore a break of self.config["continuity_threshold"] distance
        mass_bool = self.schmitt_trigger(
            mass_bool, diffdist, self.config["continuity_threshold"]
        )
        return mass_bool


def detect_traffic_barrier_changes(
    run1_paths, run2_paths, run1_gps, run2_gps, segmenter, vpoints=None,
):
    imgsize = Image.open(run2_paths[10]).convert("RGB").size[::-1]
    config = {
        "d": 60,
        "left_margin": 0.05,
        "right_margin": 1 - 0.1,
        "gps_uncertainty": 75,
        "continuity_threshold": 100,
    }
    tb_change_detector = TrafficBarrierChangeDetector(
        segmenter, config, vpoints=[vpoints[0][::-1], vpoints[1][::-1]], imgsize=imgsize
    )
    return tb_change_detector.get_changes(run1_paths, run2_paths, run1_gps, run2_gps)
