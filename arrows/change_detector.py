from arrows.utils import (
    smooth_count,
    get_shifts,
    change_type,
    set_index,
    process_change,
)
from arrows.detector import ArrowDetector


def detect_arrow_changes(
    run1_paths,
    run2_paths,
    run1_coords,
    run2_coords,
    segmenter,
    target_class=15,
    confounding_class=18,
    confounding_border_thresh=100,
    thresh=8000,
    dilation_size=11,
    min_cnt_dist=75,
    write_output=True,
    shift_tolerance=50,
    search_boundary=5,
    min_track_length=5,
):
    arrow_cd = ArrowChangeDetector(
        segmenter,
        target_class=target_class,
        confounding_class=confounding_class,
        confounding_border_thresh=confounding_border_thresh,
        thresh=thresh,
        dilation_size=dilation_size,
        min_cnt_dist=min_cnt_dist,
        write_output=write_output,
        shift_tolerance=shift_tolerance,
        search_boundary=search_boundary,
        min_track_length=min_track_length,
    )

    changes = arrow_cd.get_changes(run1_paths, run2_paths, run1_coords, run2_coords)
    return changes


class ArrowChangeDetector:
    def __init__(
        self,
        segmenter,
        target_class,
        confounding_class,
        confounding_border_thresh,
        thresh,
        dilation_size,
        min_cnt_dist,
        shift_tolerance,
        search_boundary,
        min_track_length,
        write_output,
    ):
        """

        :param shift_tolerance: Max. shift b/w two runs at which changes are not analyzed->When unaligned or noisy
        :param search_boundary: Boundary around the shift b/w two runs to look for changes
        """
        self.arrow_detector = ArrowDetector(
            segmenter,
            target_class=target_class,
            confounding_class=confounding_class,
            confounding_border_thresh=confounding_border_thresh,
            thresh=thresh,
            dilation_size=dilation_size,
            min_cnt_dist=min_cnt_dist,
            write_output=write_output,
        )
        self.shift_tolerance = shift_tolerance
        self.mode = "left"
        self.search_boundary = search_boundary
        self.min_track_length = min_track_length

    def get_changes(self, run1_paths, run2_paths, run1_coords, run2_coords):
        """
        :param run1_paths: paths to aligned images from run 1
        :param run2_paths: paths to aligned images from run 2
        :return: changes from run 1 -> run 2
        """
        counts_run1, counts_run2 = self.arrow_detector.detect_and_count(
            run1_paths, run2_paths
        )
        counts_run1 = smooth_count(counts_run1)
        counts_run2 = smooth_count(counts_run2)
        shift_run1, shift_run2 = get_shifts(counts_run1, counts_run2)
        if shift_run1 > shift_run2:
            counts_run1, counts_run2 = counts_run2, counts_run1
            shift_run1, shift_run2 = shift_run2, shift_run1
            self.mode = "right"
        if shift_run1 < self.shift_tolerance:
            changes = self.extract_changes_new(counts_run1, counts_run2, shift_run1)
            # Save the GPS coordinates in the final array
            for i, change in enumerate(changes):
                change[11] = run1_coords[change[-2]]
                change[12] = run2_coords[change[-1]]
        else:
            changes = []
        return changes

    def extract_changes(self, counts_run1, counts_run2, shift_run1):
        """

        :param counts_run1: per-frame detection counts for left i.e leading run
        :param counts_run2: per-frame detection counts for right i.e lagging run
        :param shift_run1: shift b/w two runs
        :return: changes b/w two runs, adjusted for lag/lead wrt run 1 and run 2
        """
        pointer_run1 = 0
        pointer_run2 = shift_run1
        length_run1 = len(counts_run1)
        length_run2 = len(counts_run2)
        traversed = [False] * length_run2
        changes = []

        while pointer_run1 < length_run1 and pointer_run2 < length_run2:
            if (
                counts_run1[pointer_run1] != 0
                and pointer_run1 != length_run1 - 1
                and counts_run1[pointer_run1 + 1] == 0
            ):
                j = -1
                search_index_run1 = pointer_run1 + j
                max_run1 = counts_run1[pointer_run1]
                while search_index_run1 > 0 and counts_run1[search_index_run1] != 0:
                    start = set_index(pointer_run1 + shift_run1 - self.search_boundary)
                    end = set_index(
                        pointer_run1 + shift_run1 + self.search_boundary + 1,
                        length_run1,
                    )
                    length = end - start
                    traversed[start:end] = [True] * length
                    if counts_run1[search_index_run1] > max_run1:
                        max_run1 = counts_run1[search_index_run1]
                    search_index_run1 = pointer_run1 + j
                    j -= 1

                max_run2 = counts_run2[pointer_run2]
                start = set_index(pointer_run2 - self.search_boundary)
                end = set_index(pointer_run2 + 1, length_run2)
                length = end - start
                traversed[start:end] = [True] * length
                search_index_run2 = pointer_run2 + j
                while j < 0 and 0 <= search_index_run2 < length_run2:
                    if counts_run2[search_index_run2] > max_run2:
                        max_run2 = counts_run2[search_index_run2]
                    traversed[search_index_run2] = True
                    j += 1
                for j in range(self.search_boundary):
                    if search_index_run2 >= length_run2:
                        break
                    traversed[search_index_run2] = True
                    if counts_run2[search_index_run2] > max_run2:
                        max_run2 = counts_run2[search_index_run2]

                if max_run1 != max_run2:
                    change = process_change(
                        change_type(
                            counts_run1[pointer_run1] - max_run2, mode=self.mode
                        ),
                        search_index_run1,
                        search_index_run2,
                    )
                    changes.append(change)
            pointer_run1 += 1
            pointer_run2 += 1

        pointer_run2 = 0
        while pointer_run2 < length_run2:
            if (
                pointer_run2 < shift_run1
                and sum(counts_run2[pointer_run2 : pointer_run2 + self.search_boundary])
                != 0
            ):
                status = "Arrow added" if self.mode == "left" else "Arrow removed"
                change = process_change(
                    status, pointer_run2, pointer_run2 + self.search_boundary
                )
                changes.append(change)
                pointer_run2 = shift_run1

            else:

                pointer_run1 = pointer_run2 - shift_run1
                index_1_start = set_index(pointer_run2 - self.search_boundary)
                index_1_end = set_index(
                    pointer_run2 + self.search_boundary + 1, length_run2
                )
                s1 = sum(traversed[index_1_start:index_1_end])

                index_2_start = set_index(pointer_run2 - 1)
                index_2_end = set_index(pointer_run2 + 2, length_run2)
                s2 = sum(counts_run2[index_2_start:index_2_end])

                index_3_start = set_index(pointer_run1 - self.search_boundary)
                index_3_end = set_index(
                    pointer_run1 + self.search_boundary + 1, length_run1
                )
                s3 = sum(counts_run1[index_3_start:index_3_end])

                if s1 == 0 and s2 != 0 and s3 == 0:
                    status = "Arrow added" if self.mode == "left" else "Arrow removed"
                    change = process_change(status, index_3_start, index_1_end)
                    changes.append(change)
                    start = set_index(pointer_run2 - 2)
                    end = set_index(pointer_run2 + 3, length_run2)
                    traversed[pointer_run2 - 2 : pointer_run2 + 3] = [True] * (
                        end - start
                    )
                    pointer_run2 = end

                    while counts_run2[pointer_run2] != 0:
                        traversed[pointer_run2] = True
                        pointer_run2 += 1
                    continue
                else:
                    pointer_run2 += 1
        return changes

    def filter_tracks(self, det_list):
        det_list = [det for det in det_list if det[2] - det[1] > self.min_track_length]
        return det_list

    def extract_changes_new(self, counts_l, counts_r, left_shift):
        """
        Updated function for extracting changes
        :param counts_l: per-frame detection counts for left i.e leading run
        :param counts_r: per-frame detection counts for right i.e lagging run
        :param left_shift: shift b/w two runs
        :return: changes b/w two runs, adjusted for lag/lead wrt run 1 and run 2
        """
        changes = []
        detections_run1 = self.get_stats_detections(counts_l, left_shift=left_shift)
        detections_run2 = self.get_stats_detections(counts_r)

        # Filter the tracks based on track length
        detections_run1 = self.filter_tracks(detections_run1)
        detections_run2 = self.filter_tracks(detections_run2)

        l = len(detections_run1)
        r = len(detections_run2)

        i = 0
        j = 0

        while i < l and j < r:
            frame_start = detections_run2[j][1]
            frame_end = detections_run2[j][2]
            if detections_run1[i][1] in list(
                range(
                    max(frame_start - self.search_boundary, 0),
                    frame_end + self.search_boundary,
                )
            ):
                detections_run1[i].append(True)
                detections_run2[j].append(True)
                j += 1
                i += 1
                if detections_run1[i - 1][0] != detections_run2[j - 1][0]:
                    status = change_type(
                        detections_run1[i - 1][0] - detections_run2[j - 1][0], self.mode
                    )
                    detections_run2[j - 1].append(True)
                    change = process_change(status, frame_start, frame_end)
                    changes.append(change)
                continue
            else:
                if detections_run1[i][1] < frame_start:
                    i += 1
                else:
                    j += 1
        changes += self.get_residual_changes(detections_run1, 1)
        changes += self.get_residual_changes(detections_run2, -1)
        return changes

    def get_stats_detections(self, counts, left_shift=0):
        dets = []
        i = 0
        n = len(counts)
        while i < n:
            if counts[i] != 0:
                start = i
                max_dets = counts[i]
                while sum(counts[i : i + self.search_boundary]) != 0:
                    max_temp = max(counts[i : i + self.search_boundary])
                    if max_temp > max_dets:
                        max_dets = max_temp
                    i += self.search_boundary
                end = i
                if left_shift > 0:
                    dets.append(
                        [max_dets, (start + end) // 2 + left_shift, end + left_shift]
                    )
                else:
                    dets.append([max_dets, start, end])
            i += 1
        return dets

    def get_residual_changes(self, dets, diff=1):
        changes = []
        for det in dets:
            if len(det) != 4:
                status = change_type(diff, mode=self.mode)
                change = process_change(status, det[1], det[2])
                changes.append(change)
        return changes


def test_change_function(left="left_full", right="right_full"):
    cd = ArrowChangeDetector(None, 15, 300, 11, 75, 25, 5)
    base = "/data/nie/teams/arl/unit_tests_data/CRDv2/Arrows/"
    runs_l = []
    runs_r = []
    with open(base + left + ".txt", "r") as f:
        left_lines = f.readlines()
    with open(base + right + ".txt", "r") as f:
        right_lines = f.readlines()
    for line in left_lines:
        runs_l.append(int(line.strip("\n")))
    for line in right_lines:
        runs_r.append(int(line.strip("\n")))
    left_shift, right_shift = get_shifts(runs_l, runs_r)
    mode = "left"
    print(left_shift, right_shift)
    if left_shift > right_shift:
        runs_l, runs_r = runs_r, runs_l
        left_shift, right_shift = right_shift, left_shift
        mode = "right"
    cd.mode = mode
    changes = cd.extract_changes_new(runs_l, runs_r, left_shift)
    return changes


if __name__ == "__main__":
    print(test_change_function())
