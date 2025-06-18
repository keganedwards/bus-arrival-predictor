# historical_calculator.py
import pandas as pd
import numpy as np

import config
import utils
import data_manager


class HistoricalMetricsCalculator:
    def __init__(self, stops_info_df, routes_dict):
        self.stops_info_df = stops_info_df
        self.routes_dict = routes_dict
        self.radius_correction_factor = 0.1

    def _stop_update(
        self, all_data_df, bus_snapshot_idx, prev_stop_idx, distance_between_stops
    ):
        if pd.notnull(prev_stop_idx):
            all_data_df.at[bus_snapshot_idx, "toStop"] = utils.get_next_stop(
                all_data_df.loc[bus_snapshot_idx], self.routes_dict
            )
            from_stop_curr = all_data_df.at[bus_snapshot_idx, "fromStop"]
            from_stop_prev = all_data_df.at[prev_stop_idx, "fromStop"]

            dist = utils.distance_between_coordinates(
                self.stops_info_df.loc[from_stop_curr, "longitude"],
                self.stops_info_df.loc[from_stop_prev, "longitude"],
                self.stops_info_df.loc[from_stop_curr, "latitude"],
                self.stops_info_df.loc[from_stop_prev, "latitude"],
            )

            if dist > (
                self.stops_info_df.loc[from_stop_curr, "radius"]
                * config.PASSIO_RADIUS_CORRECTION_FACTOR
            ):
                try:
                    key = (from_stop_curr, all_data_df.at[bus_snapshot_idx, "toStop"])
                    distance = all_data_df.at[bus_snapshot_idx, "distanceFromStop"] + (
                        self.radius_correction_factor
                        * self.stops_info_df.loc[from_stop_curr, "radius"]
                        * config.PASSIO_RADIUS_CORRECTION_FACTOR
                    )
                    distance_between_stops[key].append(distance)
                finally:
                    return bus_snapshot_idx, distance_between_stops
        return bus_snapshot_idx, distance_between_stops

    def _calculate_distance_columns(self, all_data_df, distance_between_stops):
        prev_stop_idx = None
        max_time_threshold = 60000
        for i in all_data_df.index:
            if (
                all_data_df.at[i, "routeId"] not in self.routes_dict
                or all_data_df.at[i, "routeId"] in config.EXCLUDED_ROUTES
            ):
                continue
            all_data_df.at[i, "fromStop"] = utils.is_at_stop(
                all_data_df.loc[i], self.stops_info_df, self.routes_dict
            )

            if i == 0:
                all_data_df.loc[i] = utils.reset_bus_data(all_data_df.loc[i])
                if pd.notnull(all_data_df.at[i, "fromStop"]):
                    prev_stop_idx, distance_between_stops = self._stop_update(
                        all_data_df, i, prev_stop_idx, distance_between_stops
                    )
                all_data_df.loc[i] = utils.reset_bus_data(all_data_df.loc[i])
                continue

            if all_data_df.at[i, "busNumber"] != all_data_df.at[i - 1, "busNumber"]:
                all_data_df.at[i, "newBus"] = True
                all_data_df.at[i, "distanceFromStop"] = 0
                prev_stop_idx = None
            else:
                if pd.notnull(prev_stop_idx):
                    all_data_df.at[i, "distDiff"] = utils.distance_between_coordinates(
                        all_data_df.at[i, "lng"],
                        all_data_df.at[i - 1, "lng"],
                        all_data_df.at[i, "lat"],
                        all_data_df.at[i - 1, "lat"],
                    )
                    all_data_df.at[i, "timeDiff"] = (
                        all_data_df.at[i - 1, "last_updated"]
                        - all_data_df.at[i, "last_updated"]
                    ).total_seconds() * config.SECONDS_TO_MILLISECONDS
                    if all_data_df.at[i, "timeDiff"] > max_time_threshold:
                        prev_stop_idx = None
                    all_data_df.loc[i] = utils.update_distance_from_last_stop(
                        all_data_df.loc[i], all_data_df.loc[i - 1]
                    )

            if pd.notnull(all_data_df.at[i, "fromStop"]):
                prev_stop_idx, distance_between_stops = self._stop_update(
                    all_data_df, i, prev_stop_idx, distance_between_stops
                )
                all_data_df.loc[i] = utils.reset_bus_data(all_data_df.loc[i])
        return distance_between_stops

    def _calculate_median_distances(self, distances_map):
        valid_medians = [
            np.median(dist_list) for dist_list in distances_map.values() if dist_list
        ]
        overall_median = np.median(valid_medians) if valid_medians else None

        for key, value in distances_map.items():
            if value:
                distances_map[key] = np.median(value)
            else:
                distances_map[key] = overall_median
        return distances_map

    def _to_dataframe(self, distances_map):
        from_stop, to_stop = list(map(list, zip(*distances_map.keys())))
        values = list(distances_map.values())
        data = {
            "fromStop": from_stop,
            "toStop": to_stop,
            "distances_between_stops": values,
        }
        return pd.DataFrame(data)

    def run_distance_calculation(self, all_data_df, timings_df):
        distance_between_stops = get_stop_combinations(timings_df, self.routes_dict)
        distances_map = self._calculate_distance_columns(
            all_data_df, distance_between_stops
        )
        distances_map = self._calculate_median_distances(distances_map)
        distance_df = self._to_dataframe(distances_map)
        data_manager.upload_dataframe(distance_df, config.AVG_DISTANCES_COLLECTION)


def get_stop_combinations(timings_df, routes_dict):
    distance_between_stops = {}
    for i, row in timings_df.iterrows():
        if row["routeId"] not in config.EXCLUDED_ROUTES:
            key = (row["fromStop"], row["toStop"])
            if key not in distance_between_stops:
                if (
                    utils.get_next_stop(row, routes_dict, timings_called=True)
                    == row["toStop"]
                ):
                    distance_between_stops[key] = []
    return distance_between_stops


def filter_time_outliers(timings_df):
    time_by_hour_df = timings_df.groupby(
        ["fromStop", "toStop", "hour_of_day", "day_of_week"],
        as_index=False,
        dropna=False,
    )["timeTaken"].median()
    time_by_hour_df.set_index(
        ["fromStop", "toStop", "hour_of_day", "day_of_week"], inplace=True
    )

    timings_df_list = []
    for index in time_by_hour_df.index:
        hour_and_day_df = timings_df[
            (timings_df["fromStop"] == index[0])
            & (timings_df["toStop"] == index[1])
            & (timings_df["hour_of_day"] == index[2])
            & (timings_df["day_of_week"] == index[3])
            & (~timings_df["routeId"].isin(config.EXCLUDED_ROUTES))
        ]
        if hour_and_day_df.empty:
            continue

        upper_limit = np.percentile(hour_and_day_df["timeTaken"], 40)
        lower_limit = np.percentile(hour_and_day_df["timeTaken"], 5)
        iqr = upper_limit - lower_limit

        if iqr == 0:
            time_by_hour_df.loc[index, "timeTaken"] = np.median(
                hour_and_day_df["timeTaken"]
            )
        else:
            filtered = hour_and_day_df[
                (hour_and_day_df["timeTaken"] < upper_limit)
                & (hour_and_day_df["timeTaken"] > lower_limit)
            ]
            if not filtered.empty:
                time_by_hour_df.loc[index, "timeTaken"] = np.median(
                    filtered["timeTaken"]
                )
                timings_df_list.append(filtered)

    time_by_hour_df.reset_index(inplace=True)
    return (
        pd.concat(timings_df_list) if timings_df_list else pd.DataFrame()
    ), time_by_hour_df


def process_and_upload_timings(timings_df):
    timings_df["hour_of_day"] = timings_df["timeStamp"].dt.floor("H").dt.hour
    timings_df["day_of_week"] = timings_df["timeStamp"].dt.day_name()

    filtered_df, contextual_timings_df = filter_time_outliers(timings_df)

    global_timings_df = filtered_df.groupby(
        ["fromStop", "toStop"], as_index=False, dropna=False
    )[["timeTaken"]].median()

    data_manager.upload_dataframe(global_timings_df, config.GLOBAL_TIMINGS_COLLECTION)
    data_manager.upload_dataframe(contextual_timings_df, config.AVG_TIMINGS_COLLECTION)


def initialize_historical_data(stops_info_df, routes_dict):
    try:
        # A simple check to see if data exists
        if config.MONGO_DB[config.AVG_DISTANCES_COLLECTION].count_documents({}) > 0:
            print("Historical data found.")
            return
        else:
            raise Exception("No historical data in DB.")
    except Exception as e:
        print(f"Historical data not found ({e}). Calculating from scratch...")

        # Calculate Distances
        snapshots_df = data_manager.download_snapshots(limit=999999)
        timings_df = data_manager.download_timings_data()

        # Prepare snapshot data
        snapshots_df = snapshots_df[
            ~snapshots_df["routeId"].isin(config.EXCLUDED_ROUTES)
        ].copy()
        snapshots_df["last_updated"] = pd.to_datetime(
            pd.Series(snapshots_df["last_updated"]), format="%Y-%m-%dT%H:%M:%S.000Z"
        ) - timedelta(hours=5)
        snapshots_df = snapshots_df.assign(
            distDiff=None,
            timeDiff=None,
            distanceFromStop=None,
            fromStop=None,
            toStop=None,
            newBus=False,
        )
        snapshots_df.sort_values(
            by=["busNumber", "last_updated"], ascending=[False, False], inplace=True
        )
        snapshots_df.reset_index(drop=True, inplace=True)

        metrics_calculator = HistoricalMetricsCalculator(stops_info_df, routes_dict)
        metrics_calculator.run_distance_calculation(snapshots_df, timings_df)

        # Calculate Timings
        process_and_upload_timings(timings_df)
        print("Historical data calculation complete.")
