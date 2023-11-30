# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
import os
import uuid
import logging
import pandas as pd
from video_preprocessing.process_frames import process_frame


class DataFrameOperations:
    def __init__(self):
        logging.info("Interpolating GPS dataframes")

    def interpolate_frames(self, df, frame_times_in):
        """
        :param df:              dataframe of gps data
        :param frame_times_in:    list of frame times
        :return: dataframe with position for each frame
        """
        if len(df) < 4:
            # Need more than 3 for cubic interpolation
            return df
        frame_times = [
            t
            for t in frame_times_in
            if df["SampleTime"].min() < t < df["SampleTime"].max()
        ]
        frame_times.sort()
        df["GPSDateTime"] = pd.to_datetime(
            df["GPSDateTime"], errors="coerce", format="%Y:%m:%d %H:%M:%S:%f"
        )
        start_datetime = df["GPSDateTime"][0]
        df = df.set_index("SampleTime")
        new_index = pd.Index(
            list(sorted(set(frame_times_in + df.index.tolist()))), name="SampleTime"
        )
        try:
            df = df[~df.index.duplicated()]
            df = df.reindex(new_index, axis=0)
        except Exception as e:
            logging.exception("Failed to reindex dataframe.")
            logging.error("New index:\n" + str(new_index.tolist()))
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                logging.error("Dataframe:\n" + str(df))
            return []

        df = df.sort_index()
        df.loc[frame_times, "GPSDateTime"] = start_datetime + pd.to_timedelta(
            frame_times, unit="s"
        )

        try:
            df = df.interpolate(method="cubic")
        except Exception as e:
            logging.exception("Failed to interpolate frame times.")
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                logging.error("Dataframe:\n" + str(df))
            # return df because it might contain timestamps of frames that we can still use
            return df

        df_interp = df.loc[frame_times, :]

        # Remove all duplicate frame sample times
        df_interp = df_interp.loc[~df_interp.index.duplicated(keep="first")]

        # Remove all rows that contain NaN in longitude or latitude
        df_interp = df_interp[
            ~df_interp["GPSLongitude"].isna() & ~df_interp["GPSLatitude"].isna()
        ]

        df_interp = df_interp.fillna(method="ffill")
        return df_interp

    def combine_gps_nmea(self, df_interpolated, df_nmea):
        """
        Function to combine GPS and NMEA data
        :param df_interpolated: the interpolated GPS data frame
        :param df_nmea: the non-interpolated NMEA data frame
        :return: combined and interpolated GPS+NMEA frame (picking out the most imp factors)
        """

        def extract_date(row):
            # internal function to extract date
            date = (row["timestamp_ms"].split(" ")[0]).split(":")
            date = "%04d:%02d:%02d" % (int(date[0]), int(date[1]), int(date[2]))
            return date

        def extract_time(row):
            creation_date = str(row["GPSDateTime"]).split(" ")[0]
            creation_time = (str(row["GPSDateTime"]).split(" ")[1]).split(":")
            seconds_array = creation_time[-1].split(".")
            if len(seconds_array) > 1:
                second, microsecond = seconds_array
                creation_time = "%02d%02d%02d%06d" % (
                    int(creation_time[0]),
                    int(creation_time[1]),
                    int(second),
                    int(microsecond[:6]),
                )
            else:
                microsecond = "000000"
                creation_time = "%02d%02d%02d%06d" % (
                    int(creation_time[0]),
                    int(creation_time[1]),
                    int(seconds_array[0]),
                    int(microsecond[:6]),
                )
            creation_date_time = creation_date + creation_time
            return creation_date_time

        # Create GPSDateTime column in nmea
        df_nmea = df_nmea.dropna(subset=["$GPGGA"])
        stpd_time = df_nmea["$GPGGA"].str.split(",", expand=True)[1]
        hours = stpd_time.str.slice(0, 2)
        minutes = stpd_time.str.slice(2, 4)
        seconds = stpd_time.str.slice(4, 6)

        date_col = df_nmea.apply(extract_date, axis=1)

        stpd_time_str = (
            date_col
            + " "
            + hours.map(str)
            + ":"
            + minutes.map(str)
            + ":"
            + seconds.map(str)
        )

        df_nmea.loc[:, "GPSDateTime"] = pd.to_datetime(
            stpd_time_str, errors="coerce", format="%Y:%m:%d %H:%M:%S"
        )
        df_nmea = df_nmea.set_index("GPSDateTime")

        # Create output dataframe
        df_out = pd.DataFrame(
            columns=[
                "GUID",
                "FILE_URL",
                "STRING",
                "LONGITUDE",
                "LATITUDE",
                "ALTITUDE",
                "DIRECTION",
                "DEVICE_NUM",
                "CREATE",
            ]
        )

        df_out["GUID"] = df_interpolated.index.to_series().map(lambda x: uuid.uuid4())

        # Write filepaths
        df_out["FILE_URL"] = df_interpolated["Filepath"]

        # Write NMEA $GPGGA messages
        df_nmea = df_nmea.loc[~df_nmea.index.duplicated(keep="first")]
        locs = df_interpolated.apply(
            lambda row: df_nmea.index.get_loc(row.name, method="nearest"), axis=1
        )
        df_out["STRING"] = df_nmea.loc[df_nmea.index[locs], "$GPGGA"].values

        # Write longitude
        df_out["LONGITUDE"] = df_interpolated["GPSLongitude"]

        # Write latitude
        df_out["LATITUDE"] = df_interpolated["GPSLatitude"]

        # Write altitude NOT AVAILABLE
        df_out["ALTITUDE"] = 0

        # Write heading
        df_out["DIRECTION"] = df_interpolated["GPSTrack"]

        # Write device num (we shall change this later)
        df_out["DEVICE_NUM"] = "DR900S-2CH"

        # Write creation date
        df_out["CREATE"] = df_interpolated.apply(extract_time, axis=1)

        # Final manipulations (drop the 1st column and change all paths of FILE_URL to \
        df_out = df_out.set_index("GUID")

        return df_out

    def process_dataframes(
        self,
        src_video,
        output_dir,
        drive_dir,
        csv_gps,
        csv_nmea,
        frame_times,
        frame_format,
    ):
        # processing the drive name
        drive_name = os.path.basename(src_video).split(sep=".")[0]

        # ----------------Processing the GPS file (interpolation)------------------------#
        df_gps = pd.read_csv(csv_gps)
        df_interp = self.interpolate_frames(df_gps, frame_times)

        # ----------------processing the frames ----------------------------------#
        for i, frame_sample_time in enumerate(frame_times):
            process_frame(
                df_interp=df_interp,
                index=i,
                output_dir=output_dir,
                drive_dir=drive_dir,
                frame_sample_time=frame_sample_time,
                frame_format=frame_format,
            )

        # ----------------Processing the NMEA file (not-interpolated)
        df_nmea = pd.read_csv(csv_nmea)
        df_combined = self.combine_gps_nmea(df_interpolated=df_interp, df_nmea=df_nmea)

        # -----------------Writing the CSV file ----------------------------------#
        combined_csv_path = os.path.join(output_dir, drive_name + "_gps_nmea.csv")
        logging.info("Writing interpolated csv data: " + combined_csv_path)
        df_combined.to_csv(combined_csv_path, date_format="%Y:%m:%d %H:%M:%S:%f")
