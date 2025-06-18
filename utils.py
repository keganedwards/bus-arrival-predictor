# utils.py
from geopy import distance
import pandas as pd
import config


def distance_between_coordinates(lon1, lat1, lon2, lat2):
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    return distance.distance(point1, point2).feet


def is_at_stop(df_row, stops_info_df, routes_dict):
    potential_stop = None
    route_id = df_row["routeId"]
    if route_id in routes_dict:
        for stop_id in routes_dict[route_id]:
            stop_info = stops_info_df.loc[stop_id]
            dist = distance_between_coordinates(
                df_row["lng"],
                stop_info["longitude"],
                df_row["lat"],
                stop_info["latitude"],
            )
            radius = stop_info["radius"] * config.PASSIO_RADIUS_CORRECTION_FACTOR
            if dist <= radius:
                potential_stop = stop_id
                break
    return potential_stop


def get_next_stop(df_row, routes_dict, timings_called=False):
    route_id = df_row["routeId"]
    from_stop = df_row["fromStop"]

    if timings_called and (
        route_id not in routes_dict or str(from_stop) not in routes_dict[route_id]
    ):
        return False

    if (routes_dict[route_id].index(from_stop) + 1) == len(routes_dict[route_id]):
        next_stop_id = routes_dict[route_id][0]
    else:
        next_stop_id = routes_dict[route_id][routes_dict[route_id].index(from_stop) + 1]
    return next_stop_id


def reset_bus_data(df_row):
    df_row["distanceFromStop"] = 0
    df_row["timeFromStop"] = 0
    df_row["distDiff"] = 0
    df_row["timeDiff"] = 0
    return df_row


def update_distance_from_last_stop(df_row, previous_df_row):
    if df_row["newBus"]:
        previous_df_row["timeFromStop"] = 0
        previous_df_row["distanceFromStop"] = 0
    df_row["timeFromStop"] = previous_df_row["timeFromStop"] + df_row["timeDiff"]
    df_row["distanceFromStop"] = (
        previous_df_row["distanceFromStop"] + df_row["distDiff"]
    )
    return df_row


def get_number_of_stops_away(route, to_stop, target_stop, routes_dict):
    stops_away = routes_dict[route].index(target_stop) - routes_dict[route].index(
        to_stop
    )
    if stops_away < 0:
        stops_away += len(routes_dict[route])
    return stops_away
