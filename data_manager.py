# data_manager.py
from datetime import timedelta
import pandas as pd
import requests

import config


def get_routes_and_stops():
    response = requests.post(
        config.PASSIO_API_URL, data=config.PASSIO_PAYLOAD, headers=config.PASSIO_HEADERS
    )
    response.raise_for_status()
    data = response.json()

    stops_info_df = pd.DataFrame.from_dict(data["stops"], orient="index")
    stops_info_df.set_index("stopId", inplace=True)
    stops_info_df.index = stops_info_df.index.astype("string")

    routes_dict = {}
    for index, route_data in data["routes"].items():
        list_of_stops = [stop_info[1] for stop_info in route_data[3:]]
        routes_dict[int(index)] = list_of_stops

    return stops_info_df, routes_dict


def download_snapshots(limit):
    snapshot_collection = config.MONGO_DB[config.SNAPSHOTS_COLLECTION]
    records = (
        snapshot_collection.find(
            {}, {"_id": 0, "course": 0, "maxLoad": 0, "currentLoad": 0, "__v": 0}
        )
        .sort("_id", -1)
        .limit(limit)
    )
    return pd.DataFrame(list(records))


def download_timings_data():
    collection = config.MONGO_DB[config.TIMINGS_COLLECTION]
    records = collection.find({}, {"_id": 0, "__v": 0}).sort("_id", -1)
    timings_df = pd.DataFrame(list(records))
    timings_df["timeStamp"] -= timedelta(hours=5)
    return timings_df


def download_distance_data():
    collection = config.MONGO_DB[config.AVG_DISTANCES_COLLECTION]
    return pd.DataFrame(list(collection.find({}, {"_id": 0})))


def download_global_timings():
    collection = config.MONGO_DB[config.GLOBAL_TIMINGS_COLLECTION]
    return pd.DataFrame(list(collection.find({})))


def download_contextualized_timings():
    collection = config.MONGO_DB[config.AVG_TIMINGS_COLLECTION]
    return pd.DataFrame(list(collection.find({}, {"_id": 0})))


def upload_dataframe(df, collection_name):
    collection = config.MONGO_DB[collection_name]
    collection.delete_many({})
    if not df.empty:
        collection.insert_many(df.to_dict("records"))


def upload_etas(time_map_list):
    collection = config.MONGO_DB[config.ETA_COLLECTION]
    for bus in time_map_list:
        query = {"stop": bus["stop"], "routeId": bus["routeId"]}
        update = {"$set": bus}
        collection.update_one(query, update, upsert=True)
