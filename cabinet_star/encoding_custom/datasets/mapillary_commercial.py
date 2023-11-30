import json
import numpy as np
import os

from collections import defaultdict, OrderedDict

import torch

# NOTE: uncomment when running as script
# import sys
# sys.path.insert(0, "../../")

from encoding_custom.datasets.mapillary import MapillaryResearch


class MapillaryCommercial(MapillaryResearch):
    NUM_CLASS = 151
    IGNORE_INDEX = -1

    def __init__(
        self,
        root=os.path.expanduser("~/.encoding/data"),
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        **kwargs,
    ):
        super(MapillaryCommercial, self).__init__(
            root, split, mode, transform, target_transform, **kwargs
        )

    def _mask_transform(self, mask):
        mask = np.array(mask).astype("int32")
        mask[mask == 151] = -1
        return torch.from_numpy(mask).long()


class MapillaryMerged(MapillaryCommercial):
    NUM_CLASS = 26
    IGNORE_INDEX = -1

    # TODO: sync with self.new_class_dict
    CLASS_NAMES = {
        0: "background",
        1: "lane_boundary_dashed",
        2: "lane_boundary_solid",
        3: "poles",
        4: "traffic_sign",
        5: "curb",
        6: "other_barrier",
        7: "jersey_barrier",
        8: "fence",
        9: "guard_rail",
        10: "wall",
        11: "temporary_barrier",
        12: "temporary_sign",
        13: "traffic_cone",
        14: "overhead_structure",
        15: "arrows",
        16: "text",
        17: "symbol",
        18: "filled_area",
        19: "occlusions",
        20: "egovehicle",
        21: "traffic_light",
        22: "stop_line",
        23: "zebra",
        24: "zigzag",
        25: "vegetation",
    }

    def __init__(
        self,
        root=os.path.expanduser("~/.encoding/data"),
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        **kwargs,
    ):
        super(MapillaryCommercial, self).__init__(
            root, split, mode, transform, target_transform, **kwargs
        )

        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "config_commercial.json"
        )
        with open(config_path, "r") as json_file:
            json_data = json.load(json_file, object_pairs_hook=OrderedDict)
            self.class_dict = json_data["labels"]

        self.name_to_id = {x["name"]: i for i, x in enumerate(self.class_dict)}
        self._mapping()

    def _mapping(self):
        # all_class_names = set(self.name_to_id.keys())
        ignore = {
            "construction--barrier--ambiguous",
            "marking--discrete--ambiguous",
            "marking--discrete--arrow--ambiguous",
            "marking--discrete--graphics--ambiguous",
            "marking--discrete--text--ambiguous",
            "object--sign--ambiguous",
            "object--traffic-light--ambiguous",
            "object--traffic-sign--ambiguous",
            "void--unlabeled",
        }
        lane_boundary_dashed = {"marking--continuous--dashed"}
        lane_boundary_solid = {"marking--continuous--solid"}
        poles = {
            "object--support--pole",
            "object--support--pole-group",
            "object--support--traffic-sign-frame",
            "object--support--utility-pole",
        }
        traffic_sign = {
            "object--traffic-sign--back",
            "object--traffic-sign--direction-back",
            "object--traffic-sign--direction-front",
            "object--traffic-sign--front",
            "object--traffic-sign--information-parking",
        }
        curb = {"construction--barrier--curb", "construction--flat--curb-cut"}
        other_barrier = {
            "construction--barrier--acoustic",
            "construction--barrier--other-barrier",
            "construction--barrier--road-median",
            "construction--barrier--road-side",
            "construction--flat--traffic-island",
        }
        jersey_barrier = {"construction--barrier--concrete-block"}
        fence = {"construction--barrier--fence"}
        guard_rail = {"construction--barrier--guard-rail"}
        wall = {"construction--barrier--wall"}
        temporary_barrier = {"construction--barrier--temporary"}
        temporary_sign = {
            "object--traffic-sign--temporary-back",
            "object--traffic-sign--temporary-front",
        }
        traffic_cone = {"object--traffic-cone"}
        overhead_structure = {
            "construction--structure--bridge",
            "construction--structure--building",
            "construction--structure--garage",
            "construction--structure--tunnel",
        }
        arrows = {
            "marking--discrete--arrow--left",
            "marking--discrete--arrow--other",
            "marking--discrete--arrow--right",
            "marking--discrete--arrow--split-left-or-right",
            "marking--discrete--arrow--split-left-or-straight",
            "marking--discrete--arrow--split-left-right-or-straight",
            "marking--discrete--arrow--split-right-or-straight",
            "marking--discrete--arrow--straight",
            "marking--discrete--arrow--u-turn",
        }
        text = {
            "marking--discrete--text--30",
            "marking--discrete--text--40",
            "marking--discrete--text--50",
            "marking--discrete--text--bus",
            "marking--discrete--text--other",
            "marking--discrete--text--school",
            "marking--discrete--text--slow",
            "marking--discrete--text--stop",
            "marking--discrete--text--taxi",
        }
        symbol = {
            "marking--discrete--graphics--bicycle",
            "marking--discrete--graphics--other",
            "marking--discrete--graphics--pedestrian",
            "marking--discrete--graphics--wheelchair",
        }
        filled_area = {
            "marking--discrete--hatched--chevron",
            "marking--discrete--hatched--diagonal",
        }
        occlusions = {
            "human--person--individual",
            "human--person--person-group",
            "human--rider--bicyclist",
            "human--rider--motorcyclist",
            "human--rider--other-rider",
            "object--vehicle--bicycle",
            "object--vehicle--boat",
            "object--vehicle--bus",
            "object--vehicle--car",
            "object--vehicle--caravan",
            "object--vehicle--motorcycle",
            "object--vehicle--on-rails",
            "object--vehicle--other-vehicle",
            "object--vehicle--trailer",
            "object--vehicle--truck",
            "object--vehicle--vehicle-group",
            "object--vehicle--wheeled-slow",
        }
        egovehicle = {"void--car-mount", "void--ego-vehicle"}
        traffic_light = {
            "object--traffic-light--cyclists-back",
            "object--traffic-light--cyclists-front",
            "object--traffic-light--cyclists-side",
            "object--traffic-light--general-horizontal-back",
            "object--traffic-light--general-horizontal-front",
            "object--traffic-light--general-horizontal-side",
            "object--traffic-light--general-single-back",
            "object--traffic-light--general-single-front",
            "object--traffic-light--general-single-side",
            "object--traffic-light--general-upright-back",
            "object--traffic-light--general-upright-front",
            "object--traffic-light--general-upright-side",
            "object--traffic-light--other",
            "object--traffic-light--pedestrians-back",
            "object--traffic-light--pedestrians-front",
            "object--traffic-light--pedestrians-side",
            "object--traffic-light--warning",
        }
        stop_line = {"marking--discrete--stop-line"}
        zebra = {"marking--discrete--crosswalk-zebra"}
        zigzag = {"marking--continuous--zigzag"}
        vegetation = {"nature--vegetation"}

        background = {
            "animal--bird",
            "animal--ground-animal",
            "construction--barrier--separator",
            "construction--flat--bike-lane",
            "construction--flat--crosswalk-plain",
            "construction--flat--driveway",
            "construction--flat--parking",
            "construction--flat--parking-aisle",
            "construction--flat--pedestrian-area",
            "construction--flat--rail-track",
            "construction--flat--road",
            "construction--flat--road-shoulder",
            "construction--flat--service-lane",
            "construction--flat--sidewalk",
            "marking--discrete--give-way-row",
            "marking--discrete--give-way-single",
            "marking--discrete--other-marking",
            "nature--beach",
            "nature--mountain",
            "nature--sand",
            "nature--sky",
            "nature--snow",
            "nature--terrain",
            "nature--water",
            "object--banner",
            "object--bench",
            "object--bike-rack",
            "object--catch-basin",
            "object--cctv-camera",
            "object--fire-hydrant",
            "object--junction-box",
            "object--mailbox",
            "object--manhole",
            "object--parking-meter",
            "object--phone-booth",
            "object--pothole",
            "object--ramp",
            "object--sign--advertisement",
            "object--sign--back",
            "object--sign--information",
            "object--sign--other",
            "object--sign--store",
            "object--street-light",
            "object--trash-can",
            "object--water-valve",
            "object--wire-group",
            "void--dynamic",
            "void--ground",
            "void--static",
        }

        # TODO: sync with CLASS_NAMES
        new_class_dict = {
            -1: ignore,
            0: background,
            1: lane_boundary_dashed,
            2: lane_boundary_solid,
            3: poles,
            4: traffic_sign,
            5: curb,
            6: other_barrier,
            7: jersey_barrier,
            8: fence,
            9: guard_rail,
            10: wall,
            11: temporary_barrier,
            12: temporary_sign,
            13: traffic_cone,
            14: overhead_structure,
            15: arrows,
            16: text,
            17: symbol,
            18: filled_area,
            19: occlusions,
            20: egovehicle,
            21: traffic_light,
            22: stop_line,
            23: zebra,
            24: zigzag,
            25: vegetation,
        }

        # NOTE sanity check
        s = set()
        for new_class_id, subset in new_class_dict.items():
            assert s.isdisjoint(subset), f"Repeated classes: {s & subset}"
            s = s | subset

        mapping = defaultdict(int)
        for new_class_id, subset in new_class_dict.items():
            for class_name in subset:
                mapping[self.name_to_id[class_name]] = new_class_id

        self.indexer = np.array([mapping[x] for x in range(len(self.class_dict))])

        # NOTE: get csv style, like so
        # list(zip(self.indexer, self.name_to_id.keys()))

    def _mask_transform(self, mask):
        mask = np.array(mask).astype("int32")
        # NOTE using torch gather might be faster
        return torch.from_numpy(self.indexer[mask]).long()


if __name__ == "__main__":
    mm = MapillaryMerged()
