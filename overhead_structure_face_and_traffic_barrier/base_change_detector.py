import numpy as np
from scipy.signal import convolve


class BaseChangeDetector:
    def __init__(self, Detector, segmenter, config, vpoints, imgsize, save):
        self.detector = Detector(
            segmenter, config, vpoints=vpoints, imgsize=imgsize, save=save
        )

    def get_distances(self, gps):
        diffdist = self.gps2meters(gps[:-1, 0], gps[:-1, 1], gps[1:, 0], gps[1:, 1])
        diffdist = np.insert(diffdist, 0, 0)
        cumdist = diffdist.cumsum()
        return diffdist, cumdist

    @staticmethod
    def convolve_with_non_linearity(mass, ks=7, rep=20):
        filt = np.ones(ks)
        filt /= np.sum(filt)
        mass /= np.max(mass)
        for j in range(rep):
            mass = convolve(mass, filt, mode="same")
            if j % 2 == 0:
                mass = np.tanh(2 * mass - 0.05)
                mass[mass < 0] = 0
        return mass

    @staticmethod
    def convolve_with_rightexp(mass, ks=7, rep=20):
        filt = np.concatenate([np.exp(np.arange(ks)), np.zeros(ks - 1)])[::-1]
        filt /= np.sum(filt)
        mass /= np.max(mass)
        for j in range(rep):
            mass = convolve(mass, filt, mode="same")
            if j % 2 == 0:
                mass = np.tanh(2 * mass - 0.05)
                mass[mass < 0] = 0
        return mass

    @staticmethod
    def derivative(mass,):
        filt = np.array([-1, 1])
        mass = convolve(mass, filt, mode="same")
        return mass

    @staticmethod
    def medfilt(x, k):
        """
        Apply a length-k median filter to a 1D array x.
        Boundaries are extended by repeating endpoints.
        """
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros((len(x), k), dtype=x.dtype)
        y[:, k2] = x
        for i in range(k2):
            j = k2 - i
            y[:j, i] = x[0]
            y[j:, i] = x[:-j]
            y[:-j, -(i + 1)] = x[j:]
            y[-j:, -(i + 1)] = x[-1]
        return np.median(y, axis=1)

    @staticmethod
    def schmitt_trigger(mass_bool, diffdist, left_hysteresis_threshold):
        u, v = 10 * left_hysteresis_threshold, 10 * left_hysteresis_threshold
        for j in range(1, len(mass_bool)):
            if mass_bool[j] and u < left_hysteresis_threshold:
                mass_bool[(j - v) : j] = True
                u = 0
                v = 0
            elif mass_bool[j - 1] and not mass_bool[j]:
                u = diffdist[j]
                v = 1
            elif not mass_bool[j - 1] and not mass_bool[j]:
                u += diffdist[j]
                v += 1
        return mass_bool

    @staticmethod
    def windowcorrelation(x1, x2, k):
        """Apply a length-k windowed correlation to a
        1D arrays x1 and x2. Boundaries are extended
        by repeating endpoints.
        """
        assert k % 2 == 1, "Window length must be odd."
        assert x1.ndim == 1, "Input x1 must be one-dimensional."
        assert x2.ndim == 1, "Input x2 must be one-dimensional."
        k2 = (k - 1) // 2
        y1 = np.zeros((len(x1), k), dtype=x1.dtype)
        y2 = np.zeros((len(x2), k), dtype=x2.dtype)
        y1[:, k2] = x1
        y2[:, k2] = x2
        for i in range(k2):
            j = k2 - i

            y1[j:, i] = x1[:-j]
            y1[:j, i] = x1[0]
            y1[:-j, -(i + 1)] = x1[j:]
            y1[-j:, -(i + 1)] = x1[-1]

            y2[j:, i] = x2[:-j]
            y2[:j, i] = x2[0]
            y2[:-j, -(i + 1)] = x2[j:]
            y2[-j:, -(i + 1)] = x2[-1]
        return np.mean(y1 * y2, axis=1)

    @staticmethod
    def gps2meters(lat1, lon1, lat2, lon2):
        """
        Calculate surface distance between two points from Latitude
        and Longitude. If inputs are array, they should be same size
        and distance will be calculated between corresponding indices
        i.e., output[i] = gps2meter(lat1[i], lon1[i], lat2[i], lon2[i])
        Haversine Formula:  https://en.wikipedia.org/wiki/Haversine_formula
        """
        R = 6378.137  # Radius of earth in KM
        dLat = lat2 * (np.pi / 180.0) - lat1 * (np.pi / 180.0)
        dLon = lon2 * (np.pi / 180.0) - lon1 * (np.pi / 180.0)
        a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(lat1 * np.pi / 180) * np.cos(
            lat2 * np.pi / 180
        ) * np.sin(dLon / 2) * np.sin(dLon / 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = R * c
        return d * 1000

    def get_shifts(self, gpsA, gpsB, cumdistA, cumdistB):
        """
        Calculates two things:
        i ) how early pathA starts from pathB in meters.
            will have negative value if pathB starts first
        ii) Distance in meters of the path shared by
            both pathA and pathB
        """
        a = self.gps2meters(*gpsA[0, :], *gpsB[0, :])
        b = self.gps2meters(*gpsA[0, :], *gpsB[1, :])
        c = self.gps2meters(*gpsA[-1, :], *gpsB[-1, :])
        d = self.gps2meters(*gpsA[-1, :], *gpsB[-2, :])

        if a < 50:
            r = 0
            s = min(cumdistA[-1], cumdistB[-1])
        elif a < b:
            rep_B0 = gpsB[0, :] * np.ones(gpsA.shape)
            r_array = self.gps2meters(
                rep_B0[:, 0], rep_B0[:, 1], gpsA[:, 0], gpsA[:, 1]
            )
            r_idx = np.argmin(r_array)
            r = cumdistA[r_idx] + self.gps2meters(*gpsB[0, :], *gpsA[r_idx, :])
            s = cumdistB[-1] - c if c > d else cumdistB[-1]
        elif a > b:
            rep_A0 = gpsA[0, :] * np.ones(gpsB.shape)
            r_array = self.gps2meters(
                rep_A0[:, 0], rep_A0[:, 1], gpsB[:, 0], gpsB[:, 1]
            )
            r_idx = np.argmin(r_array)
            r = -(cumdistB[r_idx] + self.gps2meters(*gpsA[0, :], *gpsB[r_idx, :]))
            s = cumdistA[-1] - c if c < d else cumdistA[-1]
        return r, s

    def process_change(
        self, idx1, idx2, gps1, gps2, run1_paths, run2_paths, object_category
    ):
        """
        Convert the output of calculations of Overhead Structure Face(./osf.py)
        and Traffic barrier(./tb.py) to the required output format in ../main.py
        """
        changes = []
        for i_2 in idx2:
            u = gps2[i_2, :] * np.ones(gps1.shape)
            distances = self.gps2meters(u[:, 0], u[:, 1], gps1[:, 0], gps1[:, 1])
            i_1 = np.argmin(distances)
            changes.append(
                [
                    "Added",
                    object_category,
                    "",
                    "",
                    "change_detected",
                    "",
                    1,
                    "",
                    "",
                    "",
                    "",
                    tuple(gps1[i_1, :]),
                    tuple(gps2[i_2, :]),
                    "CRDv2",
                    "",
                    "",
                    "",
                ]
            )

        for i_1 in idx1:
            u = gps1[i_1, :] * np.ones(gps2.shape)
            distances = self.gps2meters(u[:, 0], u[:, 1], gps2[:, 0], gps2[:, 1])
            i_2 = np.argmin(distances)
            changes.append(
                [
                    "Removed",
                    object_category,
                    "",
                    "",
                    "change_detected",
                    "",
                    1,
                    "",
                    "",
                    "",
                    "",
                    tuple(gps1[i_1, :]),
                    tuple(gps2[i_2, :]),
                    "CRDv2",
                    "",
                    "",
                    "",
                ]
            )

        return changes
