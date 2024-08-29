#!/usr/bin/env python
# coding: utf-8
from datetime import timedelta
import pandas as pd
from geopy import distance
from pymongo import MongoClient
import numpy as np
import requests
import concurrent.futures
import os

pd.options.mode.chained_assignment = None
# seconds to milliseconds
SECONDS_TO_MILLISECONDS: int = 1000
CLUSTER = MongoClient(os.environ["mongoCredential"])
DB = CLUSTER['busforce']
EXCLUDED_ROUTES: list[int] = [26294, 3406]
PASSIO_RADIUS_CORRECTION_FACTOR: int = 0.85

# returns busroutes and the coordinates of stops in a dataframe and a map respectively
def get_routes_and_stops():
    global ROUTES_DICT, STOPS_INFO_DF
    url = "https://passio3.com/www/mapGetData.php?getStops=2&deviceId=1720493&withOutdated=1&wBounds=1&showBusInOos=0&lat=35.3083779&lng=-80.7325179&wTransloc=1"
    headers = {'accept': "application/json, text/javascript, */*; q=0.01", "accept-language": "en-US,en;q=0.9",
               "content-type": "application/x-www-form-urlencoded; charset=UTF-8"}
    data = "json=%7B%22s0%22%3A%221053%22%2C%22sA%22%3A1%2C%22rA%22%3A8%2C%22r0%22%3A%223201%22%2C%22r1%22%3A%2222940%22%2C%22r2%22%3A%2226308%22%2C%22r3%22%3A%223406%22%2C%22r4%22%3A%223474%22%2C%22r5%22%3A%2216380%22%2C%22r6%22%3A%2226294%22%2C%22r7%22%3A%2235130%22%7D"
    response = requests.post(url, data=data, headers=headers)
    route_data = response.json()['routes']
    stops_info_JSON = response.json()['stops']
    STOPS_INFO_DF = pd.DataFrame.from_dict(stops_info_JSON, orient='index')
    STOPS_INFO_DF.set_index('stopId', inplace=True)
    STOPS_INFO_DF.index = STOPS_INFO_DF.index.astype('string')
    ROUTES_DICT = {}
    for index in route_data:
        list_of_stops_in_route = []
        start_index = 3
        for i in range(start_index, len(route_data[index])):
            list_of_stops_in_route.append(route_data[index][i][1])
        ROUTES_DICT[int(index)] = list_of_stops_in_route
    return

# class for calculating distances between stops
class BusStopMetrics:
    weekly_record_count: int = 999999
    radius_correction_factor = 0.1

    def __init__(self, data):
        self.stop_groupby_df = pd.DataFrame(data)

    # when a bus gets to a stop, this adds that data to a map and resets all attributes to zero
    def stop_update(self, all_data_df, bus_snapshot, prev_stop_index, distance_between_stops):
        # do we have a previous stop for the bus
        if pd.notnull(prev_stop_index):
            # calc toStop
            all_data_df['toStop'][bus_snapshot] = next_stop(all_data_df.loc[bus_snapshot])
            # checking for issues with it registering the same stop twice
            if distance_between_buses(
                    STOPS_INFO_DF.loc[(all_data_df['fromStop'][bus_snapshot]), 'longitude'],
                    STOPS_INFO_DF.loc[(all_data_df['fromStop'][prev_stop_index]), 'longitude'],
                    STOPS_INFO_DF.loc[(all_data_df['fromStop'][bus_snapshot]), 'latitude'],
                    STOPS_INFO_DF.loc[(all_data_df['fromStop'][prev_stop_index]), 'latitude']) > (
                    STOPS_INFO_DF.loc[(all_data_df['fromStop'][
                        bus_snapshot]), 'radius'] * PASSIO_RADIUS_CORRECTION_FACTOR):
                try:
                    distance_between_stops[(all_data_df['fromStop'][bus_snapshot], all_data_df['toStop'][bus_snapshot])].append(
                        all_data_df['distanceFromStop'][bus_snapshot] + (stopTimingsAndDistances.radius_correction_factor * (
                                STOPS_INFO_DF.loc[(all_data_df['fromStop'][bus_snapshot]), 'radius'] * PASSIO_RADIUS_CORRECTION_FACTOR)))
                finally:
                    prev_stop_index = bus_snapshot
                    return prev_stop_index, distance_between_stops
        # set where we are coming from
        prev_stop_index = bus_snapshot
        return prev_stop_index, distance_between_stops

    # calculates distance values
    def calcDistCol(self, filtered_timings_df, all_data_df, distance_between_stops):
        prev_stop_index = None
        maximum_time_threshold: int = 60000
        for bus_snapshot in all_data_df.index:
            # if its invalid, continue and reset
            if all_data_df['routeId'][bus_snapshot] not in ROUTES_DICT.keys() or all_data_df['routeId'][bus_snapshot] in EXCLUDED_ROUTES:
                continue
            # classify if it's a stop
            all_data_df['fromStop'][bus_snapshot] = is_stop(all_data_df.loc[bus_snapshot])
            # 0 cond
            if bus_snapshot == 0:
                all_data_df.loc[bus_snapshot] = reset_data(all_data_df.loc[bus_snapshot])
                # bus_snapshot is a stop, reset
                if pd.notnull(all_data_df['fromStop'][bus_snapshot]):
                    prev_stop_index, distance_between_stops = self.stop_update(
                        all_data_df,
                        bus_snapshot,
                        prev_stop_index,
                        distance_between_stops
                    )
                all_data_df.loc[bus_snapshot] = reset_data(all_data_df.loc[bus_snapshot])
                continue
            # making sure it's the same bus
            if all_data_df['busNumber'][bus_snapshot] != all_data_df['busNumber'][bus_snapshot - 1]:
                # we are not going to find what stop this bus is going to
                all_data_df['newBus'][bus_snapshot] = True
                all_data_df['distanceFromStop'][bus_snapshot] = 0
                prev_stop_index = None
            else:
                # pass in current timestamps cord and previous time stamps cord to distance function
                if pd.notnull(prev_stop_index):
                    all_data_df['distDiff'][bus_snapshot] = distance_between_buses(all_data_df['lng'][bus_snapshot],
                                                                        all_data_df['lng'][bus_snapshot - 1],
                                                                        all_data_df['lat'][bus_snapshot],
                                                                        all_data_df['lat'][bus_snapshot - 1])
                    all_data_df['timeDiff'][bus_snapshot] = (((all_data_df['last_updated'][bus_snapshot - 1] - all_data_df['last_updated'][
                        bus_snapshot]).total_seconds()) * SECONDS_TO_MILLISECONDS)
                    # if it's more than 1 minute, its invalid.
                    if all_data_df['timeDiff'][bus_snapshot] > maximum_time_threshold:
                        prev_stop_index = None
                    all_data_df.loc[bus_snapshot] = distance_from_last_stop(all_data_df.loc[bus_snapshot], all_data_df.loc[bus_snapshot - 1])
            # bus_snapshot is a stop
            if pd.notnull(all_data_df['fromStop'][bus_snapshot]):
                prev_stop_index, distance_between_stops = self.stop_update(
                    all_data_df, bus_snapshot,
                    prev_stop_index,
                    distance_between_stops
                )
                all_data_df.loc[bus_snapshot] = reset_data(all_data_df.loc[bus_snapshot])
        return filtered_timings_df, distance_between_stops

    # takes the median of all of the distance values for a to-stop from-stop combination
    def calculate_median(self, distances_between_stops_map):
        # fill in missing stops with something
        valid_stops = []
        for key in distances_between_stops_map.keys():
            if distances_between_stops_map[key]:
                distances_between_stops_map[key] = np.median(distances_between_stops_map[key])
                valid_stops.append(distances_between_stops_map[key])
            else:
                distances_between_stops_map[key] = None
        for key in distances_between_stops_map.keys():
            if pd.isnull(distances_between_stops_map[key]):
                # if we did not find an instance of a to-stop from-stop combination, we sub the median median
                try:
                    distances_between_stops_map[key] = np.median(valid_stops)
                # of we use none, if a median cannot be found
                except:
                    distances_between_stops_map[key] = None
        return distances_between_stops_map

    def upload_distance_data(self, distance_groupby_df):
        collection = DB['averageDistancesBetweenStops']
        collection.delete_many({})
        collection.insert_many(distance_groupby_df.to_dict('records'))
        return

    def to_dataframe(self, distances_between_stops):
        index = list(distances_between_stops.keys())
        values = (distances_between_stops.values())
        fromStop, toStop = list(map(list, zip(*index)))
        data = {'fromStop': fromStop, 'toStop': toStop, 'dinstances_between_stops': values}
        distance_groupby_df = pd.DataFrame(data)
        return distance_groupby_df

    # main driver of the dist calculations
    def to_groupby(self, filtered_timings_df, distance_between_stops):
        all_data_df = download_data(stopTimingsAndDistances.weekly_record_count)
        all_data_df = prepare_data(all_data_df)
        filtered_timings_df, distances_between_stops_map = self.calcDistCol(filtered_timings_df, all_data_df,
                                                                       distance_between_stops)
        distances_between_stops_map = self.calculate_median(distances_between_stops_map)
        distance_groupby_df = self.to_dataframe(distances_between_stops_map)
        self.upload_distance_data(distance_groupby_df)
        return distance_groupby_df


# calls in the timings data for the time calculations
def get_timings_data():
    timings_collection = DB['stoptimings']
    # newest to oldest
    timings_records = timings_collection.find({}, {'_id': 0, '__v': 0}).sort("_id", -1)
    timings_curser = list(timings_records)
    timings_df = pd.DataFrame(timings_curser)
    # accounting for timezone
    timings_df['timeStamp'] = timings_df['timeStamp'] - timedelta(hours=5)
    return timings_df


# calls in the distance data for the distance calculations
def get_distance_data():
    global distance_groupby_df
    distance_collection = DB['averageDistancesBetweenStops']
    distance_records = distance_collection.find({}, {'_id': 0})
    distance_curser = list(distance_records)
    distance_groupby_df = pd.DataFrame(distance_curser)
    return

# this gives us unique combinations of to-stop and fromstop, to be used as the keys in our distmap
def calculate_stop_combinations(timings_df):
    timings_called = True
    distance_between_stops = {}
    for i, row in timings_df.iterrows():
        if row['routeId'] not in EXCLUDED_ROUTES:
            if (row['fromStop'], row['toStop']) not in distance_between_stops.keys():
                # making sure it's a valid from/toStop combination
                if next_stop(row, timings_called) == row['toStop']:
                    distance_between_stops[(row['fromStop'], row['toStop'])] = []
    return distance_between_stops


def filter_time_outliers(timings):
    timings_df_list = []
    # eg (fromstop, tostop)
    for index in time_by_hour_groupby_dataframe.index:
        # filtering for instances of the from-stop to stop combo, and filtering out excluded routes
        hour_and_day_df = timings[
            (timings['fromStop'] == index[0]) &
            (timings['toStop'] == index[1]) &
            (timings['hour_of_day'] == index[2]) &
            (timings['day_of_week'] == index[3]) &
            (~timings['routeId'].isin(EXCLUDED_ROUTES))
        ]
        if hour_and_day_df.empty:
            continue
        lower_limit_percentile, upper_limit_percentile, range_factor = 5, 40, 0
        upper_limit = np.percentile(hour_and_day_df['timeTaken'], upper_limit_percentile)
        lower_limit = np.percentile(hour_and_day_df['timeTaken'], lower_limit_percentile)
        iqr = upper_limit - lower_limit
        # if there is 1 entry, the iqr will be zero
        if iqr == 0:
            time_by_hour_groupby_dataframe.loc[[index], ['timeTaken']] = np.median(hour_and_day_df[['timeTaken']])
        else:
            filtered_times_df = hour_and_day_df[
                (hour_and_day_df['timeTaken'] < upper_limit + (range_factor * iqr)) &
                (hour_and_day_df['timeTaken'] > lower_limit - (range_factor * iqr))
            ]
            if not filtered_times_df.empty:
                time_by_hour_groupby_dataframe.loc[[index], ['timeTaken']] = np.median(hour_and_day_df[['timeTaken']])
                timings_df_list.append(filtered_times_df)
    process_global_times_df(timings_df_list)
    return


def process_global_times_df(global_timings_df):
    global global_times_groupby_df
    global_timings_df[['fromStop', 'toStop']] = global_timings_df[['fromStop', 'toStop']].astype('string')
    global_times_groupby_df = global_timings_df.groupby(['fromStop', 'toStop'], as_index=False, dropna=False)[
        ['timeTaken']].median()
    return


def add_day_and_hour(filtered_timings_):
    filtered_timings_df['hour_of_day'] = filtered_timings_df["timeStamp"].dt.floor("H").dt.hour
    filtered_timings_df['day_of_week'] = filtered_timings_df['timeStamp'].dt.day_name()
    return filtered_timings_df


def create_time_by_hour_df(filtered_timings_):
    global time_by_hour_groupby_dataframe
    time_by_hour_groupby_dataframe = \
        filtered_timings_df.groupby(['fromStop', 'toStop', 'hour_of_day', 'day_of_week'], as_index=False, dropna=False)[
            'timeTaken'].median()
    return


def upload_average_timings():
    time_by_hour_groupby_dataframe.reset_index(inplace=True)
    collection = DB['averageTimingsBetweenStops']
    collection.delete_many({})
    collection.insert_many(time_by_hour_groupby_dataframe.to_dict('records'))
    return


def upload_global_timings():
    collection = DB['globalAverageTimings']
    collection.delete_many({})
    collection.insert_many(time_by_hour_groupby_dataframe.to_dict('records'))
    return


def process_stop_group_by():
    distance_groupby_df[['fromStop', 'toStop']] = distance_groupby_df[['fromStop', 'toStop']].astype('string')
    distance_groupby_df.set_index(['fromStop', 'toStop'], inplace=True)
    time_by_hour_groupby_dataframe[['fromStop', 'toStop']] = time_by_hour_groupby_dataframe[['fromStop', 'toStop']].astype('string')
    time_by_hour_groupby_dataframe.set_index(['fromStop', 'toStop', 'hour_of_day', 'day_of_week'], inplace=True)
    return


def download_global_timings():
    collection = DB['globalAverageTimings']
    timings_records = collection.find({})
    timings_curser = list(timings_records)
    global_timings = pd.DataFrame(timings_curser)
    process_global_times_df(global_timings)
    return


# this considers time of day and day of week
def download_contextualized_timings():
    global time_by_hour_groupby_dataframe
    collection = DB['averageTimingsBetweenStops']
    timings_records = collection.find({}, {'_id': 0})
    timings_curser = list(timings_records)
    time_by_hour_groupby_dataframe = pd.DataFrame(timings_curser)
    return

# we try to download the data, if it doesn't exist, we calculate it
def download_historical_data():
    global distance_groupby_df
    # checking to make sure that dist exists
    try:
        # creates global vars
        get_distance_data()
        download_global_timings()
        download_contextualized_timings()
        global_times_groupby_df.set_index(['fromStop', 'toStop'], inplace=True)
    # if distData doesnt exist, calculate it
    except:
        timings_df = get_timings_data()
        stop_object = stopTimingsAndDistances(None)
        distances_between_stops = calculate_stop_combinations(timings_df)
        stop_object.to_groupby(timings_df, distances_between_stops)
        filtered_timings_df = add_day_and_hour(timings_df)
        create_time_by_hour_df(filtered_timings_df)
        filter_time_outliers(timings_df)
        upload_global_timings()
        upload_average_timings()
        global_times_groupby_df.set_index(['fromStop', 'toStop'], inplace=True)
        time_by_hour_groupby_dataframe.set_index(['fromStop', 'toStop', 'hour_of_day', 'day_of_week'], inplace=True)
    process_stop_group_by()
    return

# data for time_to_stop
def download_data(weekly_record_count):
    snapshot_collection = DB['bussnapshots']
    snapshot_records = snapshot_collection.find({},
                                                {'_id': 0, 'course': 0, 'maxLoad': 0, 'currentLoad': 0, '__v': 0}).sort(
        "_id", -1).limit(weekly_record_count)
    snapshot_curser = list(snapshot_records)
    all_data_df = pd.DataFrame(snapshot_curser)
    return all_data_df

# prep data for time to Stop
def prepare_data(all_data_df):
    global BUSES_LIST
    # removing unwanted routes
    all_data_df = all_data_df[~all_data_df['routeId'].isin(EXCLUDED_ROUTES)]
    all_data_df['last_updated'] = pd.to_datetime(pd.Series(all_data_df['last_updated']), format='%Y-%m-%dT%H:%M:%S.000Z')
    # accounting for timezone
    all_data_df['last_updated'] = all_data_df['last_updated'] - timedelta(hours=5)
    all_data_df = all_data_df.assign(
    combined=None,
    distDiff=None,
    timeDiff=None,
    distanceFromStop=None,
    speedFromStop=None,
    fromStop=None,
    currentStop=None,
    toStop=None,
    timeFromStop=np.nan,
    newBus=False
)
    all_data_df.sort_values(by=['busNumber', 'last_updated'], ascending=[False, False], inplace=True)
    BUSES_LIST = pd.unique(all_data_df['busNumber'])
    return all_data_df


def distance_between_buses(longitude_1, longitude_2, latitude_1, latitude_2):
    point1 = [longitude_1, latitude_1]
    point2 = [longitude_2, latitude_2]
    return distance.distance(point1, point2).feet


def is_stop(df):
    potential_stop = None
    # for some old data, old routes that no longer exist were used, so we need to check if the route is valid
    if df['routeId'] in ROUTES_DICT.keys():
        for stops in range(len(ROUTES_DICT[df['routeId']])):
            # routes[route][stop] is a stop, it is the index of stop
            if (distance_between_buses(df['lng'], STOPS_INFO_DF.loc[(ROUTES_DICT[df['routeId']][stops]), 'longitude'],
                                      df['lat'], STOPS_INFO_DF.loc[(ROUTES_DICT[df['routeId']][stops]), 'latitude']) <= (
                    (STOPS_INFO_DF.loc[(ROUTES_DICT[df['routeId']][stops]), 'radius']) * PASSIO_RADIUS_CORRECTION_FACTOR)):
                # set the stop column equal to bus_snapshot of the stop at index stops
                potential_stop = ROUTES_DICT[df['routeId']][stops]
                break
    return potential_stop


def next_stop(df, timings_called=False):
    # this is for when this function is called by timings_df
    if timings_called and (
            df['routeId'] not in ROUTES_DICT.keys() or
            str(df['fromStop']) not in ROUTES_DICT[df['routeId']]
    ):
        return False
    else:
        if (ROUTES_DICT[df['routeId']].index(df['fromStop']) + 1) == len(ROUTES_DICT[df['routeId']]):
            next_stop = ROUTES_DICT[df['routeId']][0]
        else:
            next_stop = ROUTES_DICT[df['routeId']][ROUTES_DICT[df['routeId']].index(df['fromStop']) + 1]
    return next_stop


# if there are previous runs of the time_to_stop, use that context to reduce necessary calculations
def combine_with_previous_iterations(all_data_df, previous_buses_df, all_data_previous_index, is_first_index):
    # making sure it's not empty
    # if busnumber and bus_snapshot are the same, we can set the columns from the previous run equal
    if not previous_buses_df.empty and all_data_df['busNumber'] not in NEW_BUSES_LIST and \
            previous_buses_df.loc[(all_data_df['busNumber']), 'last_updated'] == all_data_df['last_updated']:
        all_data_df['fromStop'] = previous_buses_df.loc[(all_data_df['busNumber']), 'fromStop']
        all_data_df['combined'] = True
        if is_first_index:
            all_data_df['timeFromStop'] = previous_buses_df.loc[(all_data_df['busNumber']), 'timeFromStop']
            all_data_df['distanceFromStop'] = previous_buses_df.loc[(all_data_df['busNumber']), 'distanceFromStop']
            all_data_df['timeDiff'] = previous_buses_df.loc[(all_data_df['busNumber']), 'timeDiff']
            all_data_df['distDiff'] = previous_buses_df.loc[(all_data_df['busNumber']), 'distDiff']
        else:
            # pass in current timestamps cord and previous time stamps cord to distance function
            all_data_df['distDiff'] = distance_between_buses(all_data_df['lng'], all_data_previous_index['lng'],
                                                          all_data_df['lat'], all_data_previous_index['lat'])
            # time diff is the difference between the ith and the index-1 bus_snapshot
            all_data_df['timeDiff'] = (((all_data_previous_index['last_updated'] - all_data_df[
                'last_updated']).total_seconds()) * SECONDS_TO_MILLISECONDS)
            all_data_df['timeFromStop'] = all_data_previous_index['timeFromStop'] + all_data_df['timeDiff'] + \
                                        previous_buses_df.loc[(all_data_df['busNumber']), 'timeFromStop']
            all_data_df['distanceFromStop'] = all_data_previous_index['distanceFromStop'] + all_data_df['distDiff'] + \
                                            previous_buses_df.loc[(all_data_df['busNumber']), 'distanceFromStop']
    return all_data_df


def reset_data(df):
    df['distanceFromStop'] = 0
    df['timeFromStop'] = 0
    df['distDiff'] = 0
    df['timeDiff'] = 0
    return df


def distance_from_last_stop(df, previous_df):
    if df['newBus']:
        previous_df['timeFromStop'] = 0
        previous_df['distanceFromStop'] = 0
    df['timeFromStop'] = previous_df['timeFromStop'] + df['timeDiff']
    df['distanceFromStop'] = previous_df['distanceFromStop'] + df['distDiff']
    return df


def find_time_stamp(df, first_index_of_new_bus_df, new_bus=False):
    # this happens in the case that we never arrive at a stop, but we run into a new bus
    if new_bus:
        df['toStop'] = None
    else:
        # new bus cond
        if pd.isnull(df['fromStop']):
            df['toStop'] = None
        else:
            df['toStop'] = next_stop(df)
    df['last_updated'] = first_index_of_new_bus_df['last_updated']
    return df


# find bus distance and time relative to its last stop, as well as what stop it is going to
def find_bus_information(all_data_df, previous_buses_df, first_time_run):
    # buses added since the last running
    global NEW_BUSES_LIST
    # this is a df of the most recent instance of all buses to be used as reference for future runs
    repeated_buses_df = all_data_df.iloc[:0, :].copy()
    firstIndexOfBus = 0
    if not first_time_run:
        NEW_BUSES_LIST = list(set(BUSES_LIST) - set(previous_buses_df.index))
    for index in range(len(all_data_df.index)):
        if set(repeated_buses_df['busNumber']) == set(BUSES_LIST):
            break
        # check to make sure we don't already know where the bus is going
        if all_data_df['busNumber'][index] in list(repeated_buses_df['busNumber']):
            continue
        if not first_time_run:
            if index == 0:
                all_data_df.loc[index] = combine_with_previous_iterations(all_data_df.loc[index], previous_buses_df, None, True)
            else:
                if all_data_df['busNumber'][index] != all_data_df['busNumber'][index - 1]:
                    all_data_df.loc[index] = combine_with_previous_iterations(all_data_df.loc[index], previous_buses_df, None, True)
                    firstIndexOfBus = index
                else:
                    all_data_df.loc[index] = combine_with_previous_iterations(all_data_df.loc[index], previous_buses_df,
                                                                   all_data_df.loc[index - 1], False)
            if all_data_df['combined'][index]:
                all_data_df.loc[index] = find_time_stamp(all_data_df.loc[index], all_data_df.loc[firstIndexOfBus])
                repeated_buses_df = pd.concat([repeated_buses_df, all_data_df.loc[[index]]])
                previous_buses_df.drop(all_data_df['busNumber'][index])
                continue
        # classify if stop
        all_data_df['fromStop'][index] = is_stop(all_data_df.loc[index])
        # making sure it's the same bus
        if index == 0:
            all_data_df.loc[index] = reset_data(all_data_df.loc[index])
            # bus_snapshot is a stop
            if pd.notnull(all_data_df['fromStop'][index]):
                all_data_df.loc[index] = find_time_stamp(all_data_df.loc[index], all_data_df.loc[firstIndexOfBus])
                repeated_buses_df = pd.concat([repeated_buses_df, all_data_df.loc[[index]]])
            continue
        if all_data_df['busNumber'][index] != all_data_df['busNumber'][index - 1]:
            # we are not going to find what stop this bus is going to
            all_data_df['newBus'][index] = True
            if all_data_df['busNumber'][index - 1] not in list(repeated_buses_df['busNumber']):
                all_data_df.loc[index - 1] = find_time_stamp(all_data_df.loc[index - 1],
                                                                  all_data_df.loc[firstIndexOfBus],
                                                                  all_data_df['newBus'][index])
                repeated_buses_df = pd.concat([repeated_buses_df, all_data_df.loc[[index - 1]]])
                firstIndexOfBus = index
            # if it's not the same bus, reset
            all_data_df.loc[index] = reset_data(all_data_df.loc[index])
        else:
            # pass in current timestamps cord and previous time stamps cord to distance function
            all_data_df['distDiff'][index] = distance_between_buses(all_data_df['lng'][index], all_data_df['lng'][index - 1],
                                                                 all_data_df['lat'][index], all_data_df['lat'][index - 1])
            # time diff is the difference between the ith and the index-1 bus_snapshot
            all_data_df['timeDiff'][index] = (((all_data_df['last_updated'][index - 1] - all_data_df['last_updated'][
                index]).total_seconds()) * SECONDS_TO_MILLISECONDS)
            all_data_df.loc[index] = distance_from_last_stop(all_data_df.loc[index], all_data_df.loc[index - 1])
        # bus_snapshot is a stop
        if pd.notnull(all_data_df['fromStop'][index]):
            all_data_df.loc[index] = find_time_stamp(all_data_df.loc[index], all_data_df.loc[firstIndexOfBus])
            repeated_buses_df = pd.concat([repeated_buses_df, all_data_df.loc[[index]]])
    return repeated_buses_df


def process_distance_data(repeated_buses_df):
    repeated_buses_df['toStop'] = repeated_buses_df['toStop'].astype("string")
    previous_buses_df = repeated_buses_df.copy()
    # set fromStop to null for buses not at a stop
    for bus in repeated_buses_df.index:
        try:
            if repeated_buses_df.loc[bus, 'distanceFromStop'] <= (STOPS_INFO_DF.loc[
                                                                    (repeated_buses_df.loc[
                                                                        (bus, 'fromStop')]), 'radius'] * PASSIO_RADIUS_CORRECTION_FACTOR):
                repeated_buses_df['currentStop'][bus] = repeated_buses_df['fromStop'][bus]
        # if one of the fromStops is null
        finally:
            continue
    repeated_buses_df.set_index('routeId', inplace=True)
    previous_buses_df.set_index('busNumber', inplace=True)
    return repeated_buses_df, previous_buses_df


def find_routes_per_stop():
    all_routes_going_to_stop = {}
    # Iterate over the key set
    for key in ROUTES_DICT.keys():
        # Pull the hashmap list bus_snapshot via the key
        for bus_snapshot in range(len(ROUTES_DICT[key])):
            if ROUTES_DICT[key][bus_snapshot] in all_routes_going_to_stop.keys():
                all_routes_going_to_stop[ROUTES_DICT[key][bus_snapshot]].append(key)
            else:
                all_routes_going_to_stop[ROUTES_DICT[key][bus_snapshot]] = [key]
    return all_routes_going_to_stop


def number_of_stops_away(route, toStop, target_stop):
    stops_away: int = ROUTES_DICT[route].index(target_stop) - ROUTES_DICT[route].index(toStop)
    if stops_away < 0:
        stops_away += len(ROUTES_DICT[route])
    return stops_away


# fastest bus to each stop for each route
def next_bus(repeated_buses_df, all_routes_going_to_stop):
    next_bus_map = {}
    # key = stop
    for key in all_routes_going_to_stop:
        for bus_snapshot in range(len(all_routes_going_to_stop[key])):
            next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])] = float("inf")
            # check to make sure a bus is on the route
            try:
                instance = repeated_buses_df.loc[[all_routes_going_to_stop[key][bus_snapshot]]]
            except:
                # if not, skip it
                continue
            else:
                # index = route
                for index, row in instance.iterrows():
                    # if its at a stop
                    if pd.isnull(row['toStop']):
                        continue
                    if (
                            pd.notnull(row['currentStop']) and
                            row['currentStop'] == key and
                            [index] == all_routes_going_to_stop[key]
                    ):
                        stops_away = None
                        next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])] = [index, key, row['last_updated'],
                                                                                    row['timeFromStop'],
                                                                                    stops_away,
                                                                                    row['distanceFromStop'],
                                                                                    row['lng'],
                                                                                    row['lat'],
                                                                                    row['toStop']]
                        continue
                    stops_away: int = number_of_stops_away(index, row['toStop'], key)
                    # if this is the first potential bus being evaluated, its the best one so far
                    if next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])] == float("inf"):
                        next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])] = [index, key, row['last_updated'],
                                                                                    row['timeFromStop'],
                                                                                    stops_away,
                                                                                    row['distanceFromStop'], row['lng'],
                                                                                    row['lat'],
                                                                                    row['toStop']]
                        continue
                    # if a bus is already at the stop, stop looking for new buses
                    if pd.isnull(next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])][4]):
                        continue
                    # if the bus is less stops away
                    if next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])][4] > stops_away:
                        next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])] = [index, key,
                                                                                    row['last_updated'],
                                                                                    row['timeFromStop'],
                                                                                    stops_away,
                                                                                    row['distanceFromStop'],
                                                                                    row['lng'], row['lat'],
                                                                                    row['toStop']]
                        continue
                    # if the bus is the same amount of stops away, but it is further away from its last stop
                    if next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])][4] == stops_away and \
                            next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])][5] < row['distanceFromStop']:
                        next_bus_map[(key, all_routes_going_to_stop[key][bus_snapshot])] = [index, key,
                                                                                    row['last_updated'],
                                                                                    row['timeFromStop'],
                                                                                    stops_away,
                                                                                    row['distanceFromStop'],
                                                                                    row['lng'], row['lat'],
                                                                                    row['toStop']]
    return next_bus_map


# calculates the average time and distance from a stop to a target stop
def previous_stop_to_target_stop(bus_route, target_stop, last_updated, stops_left):
    # the index in the order that we want to get to
    target_index = ROUTES_DICT[bus_route].index(target_stop)
    hour_of_day = last_updated.floor("H").hour
    day_of_week = last_updated.day_name()
    if pd.isnull(stops_left):
        # average_time_to_target, dist_to_target_minus_dist_to_next_stopdistance_to_target_minus_distance_to_next_stop timeToTarget, timeFromStop, distFromStop, distance_to_next_stop
        return None, None, None, None,
    contextualized_time_to_target, average_time_to_target, distance_to_target_minus_distance_to_next_stop = 0, 0, 0
    # the point we are coming from
    previous_index = target_index - stops_left - 1
    while previous_index < target_index:
        # we are looking in our historical data for the median amount of time for from PI to PI +1
        average_time_to_target += (
            global_times_groupby_df.loc[
                (ROUTES_DICT[bus_route][previous_index], ROUTES_DICT[bus_route][previous_index + 1]), 'timeTaken'])
        contextualized_time_to_target += (time_by_hour_groupby_dataframe.loc[(ROUTES_DICT[bus_route][previous_index],
                                                           ROUTES_DICT[bus_route][previous_index + 1], hour_of_day,
                                                           day_of_week), 'timeTaken'])
        if previous_index == target_index - stops_left - 1:
            distance_to_next_stop = (distance_groupby_df.loc[
                (ROUTES_DICT[bus_route][previous_index], ROUTES_DICT[bus_route][previous_index + 1]), 'dinstances_between_stops'])
        else:
            dist_to_target_minus_dist_to_next_stop += (distance_to_target_minus_distance_to_next_stop.loc[
                (ROUTES_DICT[bus_route][previous_index], ROUTES_DICT[bus_route][previous_index + 1]), 'dinstances_between_stops'])
        previous_index += 1
    return average_time_to_target, dist_to_target_minus_dist_to_next_stopdistance_to_target_minus_distance_to_next_stop contextualized_time_to_target, distance_to_next_stop


# time left until a bus of a route comes to the stop
def calculate_time_left(next_bus_map):
    all_stops = []
    for key in next_bus_map:
        if next_bus_map[key] == float('inf'):
            all_stops.append(
                {'stop': key[0], 'routeId': key[1], 'timeLeft': -1, 'busHowLate': -1, 'traffic_global_ratio': -1,
                 'timeFromStop': -1, 'distanceFromStop': -1})
            continue
        else:
            # param: bus_route, target_stop, last_updated, stops_left,
            average_time_to_target, dist_to_target_minus_dist_to_next_stopdistance_to_target_minus_distance_to_next_stop contextualized_time_to_target, \
                distance_to_next_stop = previous_stop_to_target_stop(
                next_bus_map[key][0], next_bus_map[key][1], next_bus_map[key][2], next_bus_map[key][4])
            # only null if its already at a stop
            if pd.isnull(average_time_to_target) | pd.isnull(dist_to_target_minus_dist_to_next_stopdistance_to_target_minus_distance_to_next_stop | pd.isnull(
                    contextualized_time_to_target) | pd.isnull(
                next_bus_map[key][3]) | pd.isnull(next_bus_map[key][5]):
                milliSecondsLate, traffic_global_ratio, millisecondsUntilStop = 0, 0, 0
                all_stops.append(
                    {'stop': key[0], 'routeId': key[1], 'timeLeft': millisecondsUntilStop,
                     'milliSecondsLate': milliSecondsLate,
                     'traffic_global_ratio': traffic_global_ratio})
                continue
            # we are checking the raw distance to the stop, and comparing it to the distance along the road. if it is
            # shorter, we go to the backup, because our distance prediction is invalid
            total_distance_to_target = distance_to_next_stop + dist_to_target_minus_dist_to_next_stop
            distance_to_target_minus_distance_to_next_stop = distance_between_buses(STOPS_INFO_DF.loc[(next_bus_map[key][8]), 'longitude'],
                                                             next_bus_map[key][6],
                                                             STOPS_INFO_DF.loc[(next_bus_map[key][8]), 'latitude'],
                                                             next_bus_map[key][7])
            distance_left_to_next_stop = distance_to_next_stop - next_bus_map[key][5]
            if distance_left_to_next_stop_backup > distance_left_to_next_stop:
                distance_left_to_next_stop = distance_left_to_next_stop_backup
            totalDistLeft = distance_left_to_next_stop + dist_to_target_minus_dist_to_next_stop
            distance_to_target_minus_distance_to_next_stop = total_distance_to_target / contextualized_time_to_target
            # assuming traffic will be normal after instance of traffic
            traffic_global_ratio = contextualized_time_to_target / average_time_to_target
            millisecondsUntilStop = totalDistLeft / speedFromStopToTarget
            # expected time - realTime
            milliSecondsLate = (next_bus_map[key][5] / speedFromStopToTarget) - next_bus_map[key][3]  # timeFromStop
        all_stops.append(
            {'stop': key[0], 'routeId': key[1], 'timeLeft': millisecondsUntilStop, 'milliSecondsLate': milliSecondsLate,
             'traffic_global_ratio': traffic_global_ratio})
    return all_stops


def upload_times(time_map_list):
    collection = DB['timetostop']
    for bus in range(len(time_map_list)):
        if not collection.find_one({"stop": time_map_list[bus]['stop'], "routeId": time_map_list[bus]['routeId']}):
            collection.insert_one(time_map_list[bus])
            continue
        collection.update_one({"stop": time_map_list[bus]['stop'], "routeId": time_map_list[bus]['routeId']},
                              {"$set": time_map_list[bus]})
    return


def main():
    previous_buses_df = None
    first_time_run: bool = True
    while True:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if first_time_run:
                thread_1 = executor.submit(get_routes_and_stops)
                # returns global stopInfoDf and ROUTES_DICT
                thread_2 = executor.submit(download_data, 5000)
                thread_3 = executor.submit(download_historical_data)
                # returns routes
                thread_1.result()
                # returns stopGroupBy
                thread_1 = executor.submit(find_routes_per_stop)
            else:
                thread_2 = executor.submit(download_data, 1250)
            all_data_df = thread_2.result()
            all_data_df = prepare_data(all_data_df)
            repeated_buses_df = find_bus_information(all_data_df, previous_buses_df, first_time_run)
            repeated_buses_df, previous_buses_df = process_distance_data(repeated_buses_df)
            routes_and_stops_map = thread_1.result()
            next_bus_map = next_bus(repeated_buses_df, routes_and_stops_map)
            # returns HistoricalData
            thread_3.result()
            all_stops_maps_list = calculate_time_left(next_bus_map)
            upload_times(all_stops_maps_list)
            first_time_run = False


if __name__ == "__main__":
    main()
