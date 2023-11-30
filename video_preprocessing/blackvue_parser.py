# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
import os
import re
import logging
import pandas as pd


class BlackVueParser:
    def __init__(self):
        # this is required to parse the GPS output
        # this datastructure becomes the PANDAS dataframe
        self._gpsdata = {
            "GPSDateTime": [],
            "GPSLatitude": [],
            "GPSLongitude": [],
            "GPSTrack": [],
            "GPSSpeed": [],
            "SampleTime": [],
        }

        self._timestamp_ms_list = list()
        self._identifier_list = list()
        self._nmea_line_list = list()

    def parse_exiftool_sample_time(self, sample_time):
        sample_parts = sample_time.split(":")
        if len(sample_parts) == 1:  # e.g. 29.75s
            return float(sample_parts[0][:-1])
        elif len(sample_parts) == 3:  # e.g. 00:31:00
            return (
                float(sample_parts[0]) * 3600
                + float(sample_parts[1]) * 60
                + float(sample_parts[2])
            )
        else:
            raise ValueError(
                "Got some bad SampleTime with Value {0}. "
                "Please double check the input.".format(sample_time)
            )

    def parse_nmea_line(self, nmea_line):
        line = nmea_line.strip("\n")
        timestamp_ms = line[line.find("[") + 1 : line.find("]")]

        # strip timestamps in brackets
        csv_line = line[line.find("]") + 1 :]
        csv_line = csv_line[: csv_line.find("[")]

        tokens = csv_line.split(",")
        identifier = tokens[0]
        return timestamp_ms, identifier, csv_line

    def to_gps_dataframe(self):
        """
        Convert gps data to pandas dataframe
        :return: dataframe
        """
        # Get the length of all the data Series, put this into a set and check
        # if any of the lengths are different.
        lists_have_different_size = (
            len(set([len(x) for x in self._gpsdata.values()])) > 1
        )
        if lists_have_different_size:
            print(
                "Series of input data -GPSDateTime, -GpsLatitude, -GpsLatitude  are not of the same length."
            )
            # return an empty data frame
            return pd.DataFrame(
                columns=[
                    "GPSDateTime",
                    "GPSLatitude",
                    "GPSLongitude",
                    "GPSTrack",
                    "GPSSpeed",
                    "SampleTime",
                ]
            )

        df = pd.DataFrame(self._gpsdata)
        df.loc[:, "GPSDateTime"] = pd.to_datetime(
            df["GPSDateTime"], errors="coerce", format="%Y:%m:%d-%H:%M:%S"
        )

        if len(df) == 0:
            # if empty just return it
            return df

        df["SampleTime"] = df.apply(
            lambda x: self.parse_exiftool_sample_time(x["SampleTime"]), axis=1
        )

        df.loc[:, "GPSLatitude"] = (
            df.loc[:, "GPSLatitude"].str.replace("N", "").astype(float)
        )
        df.loc[:, "GPSLongitude"] = (
            df.loc[:, "GPSLongitude"].str.replace("E", "").astype(float)
        )
        return df

    def to_nmea_dataframe(self):
        # Create dataframe of all NMEA messages
        df = pd.DataFrame(
            {
                "timestamp_ms": self._timestamp_ms_list,
                "identifier": self._identifier_list,
                "nmea_string": self._nmea_line_list,
            }
        )
        if len(self._timestamp_ms_list) == 0:
            return df

        # Parse timestamps to datetime
        df["timestamp_ms"] = pd.to_datetime(
            df["timestamp_ms"], errors="coerce", unit="ms"
        )

        df = df.set_index("timestamp_ms")

        # Pivot the dataframe so that the NMEA identifiers are not the column names
        df = df.pivot_table(
            index=["timestamp_ms"],
            values=["nmea_string"],
            columns=["identifier"],
            aggfunc=lambda x: x[0],
        )
        df = df["nmea_string"]

        return df

    def parse_gps_txt_file(self, gps_txt_file):
        # clear the dict
        self._gpsdata.update((key, []) for key in self._gpsdata)
        # parses the GPS file
        with open(gps_txt_file) as gps_file:
            gps_info = gps_file.readlines()

        for line in gps_info:
            line = (
                line.replace(" : ", "=")
                .replace(" ", "")
                .replace("\n", "")
                .replace("/", "")
            )
            tag, value = re.split("=", line)
            if tag != "StartTime":
                self._gpsdata[tag].append(value)

    def parse_nmea_txt_file(self, nmea_txt_file):
        # parses the GPS file
        with open(nmea_txt_file) as nmea_file:
            nmea_info = nmea_file.readlines()

        for line in nmea_info:
            if "$G" in line:
                timestamp_ms, identifier, nmea_line = self.parse_nmea_line(line)
                self._timestamp_ms_list.append(timestamp_ms)
                self._identifier_list.append(identifier)
                self._nmea_line_list.append(nmea_line)

    # def dataframe_operations(self, temp_gps_run, temp_nmea_run, csv_gps_run, csv_nmea_run):
    def dataframe_operations(
        self, temp_gps_run, csv_gps_run, temp_nmea_run, csv_nmea_run
    ):
        self.parse_gps_txt_file(gps_txt_file=temp_gps_run)
        df_gps_run = self.to_gps_dataframe()
        df_gps_run = df_gps_run.drop_duplicates(subset=["GPSDateTime"])

        self.parse_nmea_txt_file(nmea_txt_file=temp_nmea_run)
        df_nmea_run = self.to_nmea_dataframe()

        df_gps_run.to_csv(csv_gps_run, date_format="%Y:%m:%d %H:%M:%S:%f", index=False)
        df_nmea_run.to_csv(csv_nmea_run, date_format="%Y:%m:%d %H:%M:%S:%f")

    def black_vue_parser_main(self, gps_file_list):
        """
        Controlling function for blackvue parsing
        :param gps_file_list: txt and csv file path
        :return: None
        """
        logging.info("Parsing GPS information as dataframe")
        temp_gps_run, csv_gps_run, temp_nmea_run, csv_nmea_run = gps_file_list
        self.dataframe_operations(
            temp_gps_run, csv_gps_run, temp_nmea_run, csv_nmea_run
        )
        # next we delete the temp txt file (as they are no longer needed)
        os.remove(temp_gps_run)
        os.remove(temp_nmea_run)
        return
