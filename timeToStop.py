#!/usr/bin/env python
# coding: utf-8
from datetime import timedelta
import pandas as pd
from geopy import distance
from pymongo import MongoClient
import numpy as np
# this is a library for calculating distance based on lng and lat
import requests
import concurrent.futures
import os
pd.options.mode.chained_assignment = None
# seconds to milliseconds
secondsToMilliseconds: int = 1000
cluster = MongoClient(os.environ["mongoCredential"])
db = cluster['busforce']
feetToMiles: float = 0.000189
# conversion from passio radius to something more practical
timeTooLong: int = 60000
excludedRoutes: list[int] = [26294]


def getRoutesAndStops():
    global routesDict, stopsInfoDf
    url = "https://passio3.com/www/mapGetData.php?getStops=2&deviceId=1720493&withOutdated=1&wBounds=1&showBusInOos=0&lat=35.3083779&lng=-80.7325179&wTransloc=1"
    headers = {'accept': "application/json, text/javascript, */*; q=0.01", "accept-language": "en-US,en;q=0.9",
               "content-type": "application/x-www-form-urlencoded; charset=UTF-8"}
    data = "json=%7B%22s0%22%3A%221053%22%2C%22sA%22%3A1%2C%22rA%22%3A8%2C%22r0%22%3A%223201%22%2C%22r1%22%3A%2222940%22%2C%22r2%22%3A%2226308%22%2C%22r3%22%3A%223406%22%2C%22r4%22%3A%223474%22%2C%22r5%22%3A%2216380%22%2C%22r6%22%3A%2226294%22%2C%22r7%22%3A%2235130%22%7D"
    response = requests.post(url, data=data, headers=headers)
    route_data = response.json()['routes']
    stopsInfoJson = response.json()['stops']
    stopsInfoDf = pd.DataFrame.from_dict(stopsInfoJson, orient='index')
    stopsInfoDf.set_index('stopId', inplace=True)
    stopsInfoDf.index = stopsInfoDf.index.astype('string')
    routesDict = {}
    for index in route_data:
        list_of_stops_in_route = []
        for x in range(3, len(route_data[index])):
            list_of_stops_in_route.append(route_data[index][x][1])
        routesDict[int(index)] = list_of_stops_in_route
    return


class stopTimingsAndDistances:
    # about a months worth of data
    numRecords: int = 555555

    def __init__(self, data):
        self.stopGroupByDf = pd.DataFrame(data)

    def atAStopUpdate(self, entireDf, value, prevStopIndex, timingsDf, distanceBetweenStops):
        # do we have a previous stop
        if pd.notnull(prevStopIndex):
            # calc toStop
            entireDf['toStop'][value] = nextStop(entireDf.loc[value])
            if entireDf['currentStop'][value] != entireDf['currentStop'][value - 1]:
                # append distance value
                try:
                    distanceBetweenStops[(entireDf['currentStop'][value], entireDf['toStop'][value])].append(
                        entireDf['distanceBetweenStops'][value])
                finally:
                    prevStopIndex = value
                    return entireDf, timingsDf, prevStopIndex, distanceBetweenStops
        # set where we are coming from
        prevStopIndex = value
        return entireDf, timingsDf, prevStopIndex, distanceBetweenStops

    def calcDistCol(self, filteredTimingsDf, entireDf, distanceBetweenStops):
        prevStopIndex = None
        for value in entireDf.index:
            # if its invalid, continue and reset
            if entireDf['routeId'][value] not in routesDict.keys() | entireDf['routeId'] in excludedRoutes:
                entireDf.loc[value] = resetData(entireDf.loc[value])
                continue
            # classify if stop
            entireDf['currentStop'][value] = isItAStop(entireDf.loc[value])
            # 0 cond
            if value == 0:
                entireDf.loc[value] = resetData(entireDf.loc[value])
                # timestamp is a stop, reset
                if pd.notnull(entireDf['currentStop'][value]):
                    entireDf, filteredTimingsDf, prevStopIndex, distanceBetweenStops = self.atAStopUpdate(entireDf,
                                                                                                          value,
                                                                                                          prevStopIndex,
                                                                                                          filteredTimingsDf,
                                                                                                          distanceBetweenStops)
                entireDf.loc[value] = resetData(entireDf.loc[value])
                continue
            # making sure it's the same bus
            if entireDf['busNumber'][value] != entireDf['busNumber'][value - 1]:
                # we are not going to find what stop this bus is going to
                entireDf['newBus'][value] = True
                entireDf['distanceBetweenStops'][value] = 0
                prevStopIndex = None
            else:
                # pass in current timestamps cord and previous time stamps cord to distance function
                if pd.notnull(prevStopIndex):
                    entireDf['distDiff'][value] = distanceBetweenPoints(entireDf['lng'][value],
                                                                        entireDf['lng'][value - 1],
                                                                        entireDf['lat'][value],
                                                                        entireDf['lat'][value - 1])
                    entireDf['timeDiff'][value] = (((entireDf['lastUpdated'][value - 1] - entireDf['lastUpdated'][
                        value]).total_seconds()) * secondsToMilliseconds)
                    # if it's more than 1 minute, its invalid.
                    if entireDf['timeDiff'][value] > timeTooLong:
                        prevStopIndex = None
                    entireDf.loc[value] = distTimeFromLastStop(entireDf.loc[value], entireDf.loc[value - 1])
            # timestamp is a stop
            if pd.notnull(entireDf['currentStop'][value]):
                entireDf, filteredTimingsDf, prevStopIndex, distanceBetweenStops = self.atAStopUpdate(entireDf, value,
                                                                                                      prevStopIndex,
                                                                                                      filteredTimingsDf,
                                                                                                      distanceBetweenStops)
                entireDf.loc[value] = resetData(entireDf.loc[value])
        return filteredTimingsDf, distanceBetweenStops

    def takeMedian(self, distancesBetweenStopsMap):
        # fill in missing stops with something
        validStopsDistsList = []
        for key in distancesBetweenStopsMap.keys():
            if distancesBetweenStopsMap[key]:
                distancesBetweenStopsMap[key] = np.median(distancesBetweenStopsMap[key])
                validStopsDistsList.append(distancesBetweenStopsMap[key])
            else:
                distancesBetweenStopsMap[key] = None
        for key in distancesBetweenStopsMap.keys():
            if pd.isnull(distancesBetweenStopsMap[key]):
                try:
                    distancesBetweenStopsMap[key] = np.median(validStopsDistsList)
                except:
                    distancesBetweenStopsMap[key] = None
        return distancesBetweenStopsMap

    def stopGroupByMaker(self, filteredTimingsDf, distanceBetweenStops):
        entireDf = callInData(stopTimingsAndDistances.numRecords)
        entireDf = prepareData(entireDf)
        filteredTimingsDf, distancesBetweenStopsMap = self.calcDistCol(filteredTimingsDf, entireDf,
                                                                       distanceBetweenStops)
        distancesBetweenStopsMap = self.takeMedian(distancesBetweenStopsMap)
        distGBDf = self.toDataFrame(distancesBetweenStopsMap)
        self.distToDataBase(distGBDf)
        return

    def distToDataBase(self, distGBDf):
        collection = db['averageDistancesBetweenStops']
        collection.delete_many({})
        collection.insert_many(distGBDf.to_dict('records'))
        return

    def toDataFrame(self, distancesBetweenStops):
        index = list(distancesBetweenStops.keys())
        values = (distancesBetweenStops.values())
        fromStop, toStop = list(map(list, zip(*index)))
        d = {'fromStop': fromStop, 'toStop': toStop, 'distBetweenStops': values}
        distGBDf = pd.DataFrame(d)
        return distGBDf


def callInTimingsData():
    timingsCollection = db['stoptimings']
    # newest to oldest
    timingsRecords = timingsCollection.find({}, {'_id': 0, '__v': 0}).sort("_id", -1)
    timingsCurser = list(timingsRecords)
    timingsDf = pd.DataFrame(timingsCurser)
    # accounting for timezone
    timingsDf['timeStamp'] = timingsDf['timeStamp'] + timedelta(hours=5)
    return timingsDf


def callInDistanceData():
    global distGBDf
    distanceCollection = db['averageDistancesBetweenStops']
    distanceRecords = distanceCollection.find({}, {'_id': 0})
    distanceCurser = list(distanceRecords)
    distGBDf = pd.DataFrame(distanceCurser)
    return distGBDf


def uniqueStopsCombinations(timingsDf):
    timingsCalled = True
    distanceBetweenStops = {}
    for i, row in timingsDf.iterrows():
        if row['routeId'] not in excludedRoutes:
            if (row['fromStop'], row['toStop']) not in distanceBetweenStops.keys():
                # making sure it's a valid from/toStop combination
                if nextStop(row, timingsCalled) == row['toStop']:
                    distanceBetweenStops[(row['fromStop'], row['toStop'])] = []
    return distanceBetweenStops


def filterTimeOutliers(timings, distanceBetweenStops):
    timingsDfsList = []
    # eg (fromstop, tostop)
    for key in distanceBetweenStops.keys():
        # filtering for instances of the fromstop to stop combo, and filtering out excluded routes
        toFromStopDf = timings[(timings['fromStop'] == key[0]) & (timings['toStop'] == key[1]) & (
            ~timings['routeId'].isin(excludedRoutes))]
        if toFromStopDf.empty:
            continue
        # we need a smaller range than the typical 25,75 * 1.5*iqr
        q1, q3 = np.percentile(toFromStopDf['timeTaken'], [25, 75])
        filtUl = toFromStopDf['timeTaken'] < q3
        filtLl = toFromStopDf['timeTaken'] > q1
        toFromStopDf = toFromStopDf[filtUl]
        toFromStopDf = toFromStopDf[filtLl]
        timingsDfsList.append(toFromStopDf)
    # after taking out outliers, recombine
    filteredTimingsDf = pd.concat(timingsDfsList)
    filteredTimingsDf.reset_index(inplace=True, drop=True)
    return filteredTimingsDf


def makeGroupBy(filteredTimingsDf):
    global timeByHourGBDf, globalTimeGBDf
    filteredTimingsDf['hourOfDay'] = filteredTimingsDf["timeStamp"].dt.floor("H").dt.hour
    filteredTimingsDf['dayOfWeek'] = filteredTimingsDf['timeStamp'].dt.day_name()
    globalTimeGBDf = filteredTimingsDf.groupby(['fromStop', 'toStop'], as_index=False, dropna=False)[
        ['timeTaken']].median()
    timeByHourGBDf = \
        filteredTimingsDf.groupby(['fromStop', 'toStop', 'hourOfDay', 'dayOfWeek'], as_index=False, dropna=False)[
            'timeTaken'].median()
    return timeByHourGBDf, globalTimeGBDf


def timingsToDb():
    collection = db['averageTimingsBetweenStops']
    collection.delete_many({})
    collection.insert_many(timeByHourGBDf.to_dict('records'))
    return


def processStopGroupBy():
    distGBDf[['fromStop', 'toStop']] = distGBDf[['fromStop', 'toStop']].astype('string')
    timeByHourGBDf[['fromStop', 'toStop']] = timeByHourGBDf[['fromStop', 'toStop']].astype('string')
    globalTimeGBDf[['fromStop', 'toStop']] = globalTimeGBDf[['fromStop', 'toStop']].astype('string')
    distGBDf.set_index(['fromStop', 'toStop'], inplace=True)
    timeByHourGBDf.set_index(['fromStop', 'toStop', 'hourOfDay', 'dayOfWeek'], inplace=True)
    globalTimeGBDf.set_index(['fromStop', 'toStop'], inplace=True)
    return


def tryDownloadingDistancesGroupBy(timingsDf):
    # checking to make sure that dist exists
    try:
        # returns global vars
        callInDistanceData()
    # if distData doesnt exist, calc it
    except:
        stopObj = stopTimingsAndDistances(None)
        stopObj.stopGroupByMaker(timingsDf)
    processStopGroupBy()
    return


def callInData(numRecords):
    snapshot_collection = db['bussnapshots']
    snapshot_records = snapshot_collection.find({},
                                                {'_id': 0, 'course': 0, 'maxLoad': 0, 'currentLoad': 0, '__v': 0}).sort(
        "_id", -1).limit(numRecords)
    snapshot_curser = list(snapshot_records)
    allDataDf = pd.DataFrame(snapshot_curser)
    return allDataDf


def prepareData(allDataDf):
    global uniqueBusesList
    # removing unwanted routes
    allDataDf = allDataDf[~allDataDf['routeId'].isin(excludedRoutes)]
    allDataDf['lastUpdated'] = pd.to_datetime(pd.Series(allDataDf['lastUpdated']), format='%Y-%m-%dT%H:%M:%S.000Z')
    # accounting for timezone
    allDataDf['lastUpdated'] = allDataDf['lastUpdated'] + timedelta(hours=5)
    allDataDf['combined'] = [None] * len(allDataDf.index)
    allDataDf['distDiff'] = [None] * len(allDataDf.index)
    # time diff is the difference between the time of a time stamp, and the next time stamp
    allDataDf['timeDiff'] = [None] * len(allDataDf.index)
    # this is the Dist the bus has traveled since it was last at a stop
    allDataDf['distanceFromStop'] = [None] * len(allDataDf.index)
    allDataDf['speedFromStop'] = [None] * len(allDataDf.index)
    allDataDf['currentStop'] = [None] * len(allDataDf.index)
    allDataDf['toStop'] = [None] * len(allDataDf.index)
    allDataDf['timeFromStop'] = np.empty(len(allDataDf.index), dtype=np.float64)
    allDataDf['newBus'] = [False] * len(allDataDf.index)
    allDataDf['newBus'][0] = True
    allDataDf.sort_values(by=['busNumber', 'lastUpdated'], ascending=False, ignore_index=True, inplace=True)
    uniqueBusesList = pd.unique(allDataDf['busNumber'])
    return allDataDf


def distanceBetweenPoints(lng1, lng2, lat1, lat2):
    point1 = [lng1, lat1]
    point2 = [lng2, lat2]
    return distance.distance(point1, point2).miles


def isItAStop(df):
    potentialStop = None
    # for some old data, old routes that no longer exist were used, so we need to check if the route is valid
    if df['routeId'] in routesDict.keys():
        for stops in range(len(routesDict[df['routeId']])):
            # routes[route][stop] is a stop, it is the index of stop
            if (distanceBetweenPoints(df['lng'], stopsInfoDf.loc[(routesDict[df['routeId']][stops]), 'longitude'],
                                      df['lat'], stopsInfoDf.loc[(routesDict[df['routeId']][stops]), 'latitude']) <= (
                    (stopsInfoDf.loc[(routesDict[df['routeId']][stops]), 'radius']) * feetToMiles)):
                # set the stop column equal to value of the stop at index stops
                potentialStop = routesDict[df['routeId']][stops]
                break
    return potentialStop


def nextStop(df, timingsCalled=False):
    # when it is called by allDataDf
    if not timingsCalled:
        if (routesDict[df['routeId']].index(df['currentStop']) + 2) <= len(routesDict[df['routeId']]):
            nextStopString = routesDict[df['routeId']][routesDict[df['routeId']].index(df['currentStop']) + 1]
        else:
            nextStopString = routesDict[df['routeId']][0]
    # this is for when this function is called by timingsDf
    else:
        if df['routeId'] not in routesDict.keys():
            nextStopString = False
        else:
            if str(df['fromStop']) not in routesDict[df['routeId']]:
                nextStopString = False
            else:
                if (routesDict[df['routeId']].index(df['fromStop']) + 2) <= len(routesDict[df['routeId']]):
                    nextStopString = routesDict[df['routeId']][routesDict[df['routeId']].index(df['fromStop']) + 1]
                else:
                    nextStopString = routesDict[df['routeId']][0]
    return nextStopString


def combineWithPreviousRuns(allDataDf, previousBusesDf, allDataPrevIndex, firstIndexCond):
    # making sure it's not empty
    # if busnumber and timestamp are the same, we can set the columns from the previous run equal
    if not previousBusesDf.empty and allDataDf['busNumber'] not in newBusesList and previousBusesDf.loc[
        (allDataDf['busNumber']), 'lastUpdated'] == allDataDf['lastUpdated']:
        allDataDf['currentStop'] = previousBusesDf.loc[(allDataDf['busNumber']), 'currentStop']
        allDataDf['combined'] = True
        if firstIndexCond:
            allDataDf['timeFromStop'] = previousBusesDf.loc[(allDataDf['busNumber']), 'timeFromStop']
            allDataDf['distanceFromStop'] = previousBusesDf.loc[(allDataDf['busNumber']), 'distanceFromStop']
            allDataDf['timeDiff'] = previousBusesDf.loc[(allDataDf['busNumber']), 'timeDiff']
            allDataDf['distDiff'] = previousBusesDf.loc[(allDataDf['busNumber']), 'distDiff']
        else:
            # pass in current timestamps cord and previous time stamps cord to distance function
            allDataDf['distDiff'] = distanceBetweenPoints(allDataDf['lng'], allDataPrevIndex['lng'],
                                                          allDataDf['lat'], allDataPrevIndex['lat'])
            # time diff is the difference between the ith and the index-1 timestamp
            allDataDf['timeDiff'] = (((allDataPrevIndex['lastUpdated'] - allDataDf[
                'lastUpdated']).total_seconds()) * secondsToMilliseconds)
            allDataDf['timeFromStop'] = allDataPrevIndex['timeFromStop'] + allDataDf['timeDiff'] + \
                                        previousBusesDf.loc[(allDataDf['busNumber']), 'timeFromStop']
            allDataDf['distanceFromStop'] = allDataPrevIndex['distanceFromStop'] + allDataDf['distDiff'] + \
                                            previousBusesDf.loc[(allDataDf['busNumber']), 'distanceFromStop']
    return allDataDf


def resetData(df):
    df['distanceFromStop'] = 0
    df['timeFromStop'] = 0
    df['distDiff'] = 0
    df['timeDiff'] = 0
    return df


def distTimeFromLastStop(df, dfMinus1):
    if df['newBus']:
        dfMinus1['timeFromStop'] = 0
        dfMinus1['distanceFromStop'] = 0
    df['timeFromStop'] = dfMinus1['timeFromStop'] + df['timeDiff']
    df['distanceFromStop'] = dfMinus1['distanceFromStop'] + df['distDiff']
    return df


def finalRelaventTimeStamp(df, dfFirstIndexOfBus, newBusPreviousIndex=False):
    # this happens in the case that we never arrive at a stop, but we run into a new bus
    if newBusPreviousIndex:
        df['toStop'] = None
    else:
        # newbus cond
        if pd.isnull(df['currentStop']):
            df['toStop'] = None
        else:
            df['toStop'] = nextStop(df)
    df['lastUpdated'] = dfFirstIndexOfBus['lastUpdated']
    return df


def findBusInformation(allDataDf, previousBusesDf, firstTimeRun):
    # buses added since the last running
    global newBusesList
    # this is a df of the most recent instance of all buses to be used as reference for future runs
    repeatedBusesDf = allDataDf.iloc[:0, :].copy()
    firstIndexOfBus = 0
    if not firstTimeRun:
        newBusesList = list(set(uniqueBusesList) - set(previousBusesDf.index))
    for index in range(len(allDataDf.index)):
        if set(repeatedBusesDf['busNumber']) == set(uniqueBusesList):
            break
        # check to make sure we don't already know where the bus is going
        if allDataDf['busNumber'][index] in list(repeatedBusesDf['busNumber']):
            continue
        if not firstTimeRun:
            if index == 0:
                allDataDf.loc[index] = combineWithPreviousRuns(allDataDf.loc[index], previousBusesDf, None, True)
            else:
                if allDataDf['busNumber'][index] == allDataDf['busNumber'][index - 1]:
                    allDataDf.loc[index] = combineWithPreviousRuns(allDataDf.loc[index], previousBusesDf,
                                                                   allDataDf.loc[index - 1], False)
                else:
                    allDataDf.loc[index] = combineWithPreviousRuns(allDataDf.loc[index], previousBusesDf, None, True)
            if allDataDf['combined'][index]:
                allDataDf.loc[index] = finalRelaventTimeStamp(allDataDf.loc[index], allDataDf.loc[firstIndexOfBus])
                repeatedBusesDf = pd.concat([repeatedBusesDf, allDataDf.loc[[index]]])
                previousBusesDf.drop(allDataDf['busNumber'][index])
                continue
        # classify if stop
        allDataDf['currentStop'][index] = isItAStop(allDataDf.loc[index])
        # making sure it's the same bus
        if index == 0:
            allDataDf.loc[index] = resetData(allDataDf.loc[index])
            # timestamp is a stop
            if pd.notnull(allDataDf['currentStop'][index]):
                allDataDf.loc[index] = finalRelaventTimeStamp(allDataDf.loc[index], allDataDf.loc[firstIndexOfBus])
                repeatedBusesDf = pd.concat([repeatedBusesDf, allDataDf.loc[[index]]])
            continue
        if allDataDf['busNumber'][index] != allDataDf['busNumber'][index - 1]:
            # we are not going to find what stop this bus is going to
            allDataDf['newBus'][index] = True
            if allDataDf['busNumber'][index - 1] not in list(repeatedBusesDf['busNumber']):
                allDataDf.loc[index - 1] = finalRelaventTimeStamp(allDataDf.loc[index - 1],
                                                                  allDataDf.loc[firstIndexOfBus],
                                                                  allDataDf['newBus'][index])
                repeatedBusesDf = pd.concat([repeatedBusesDf, allDataDf.loc[[index - 1]]])
                firstIndexOfBus = index
            # if it's not the same bus, reset
            allDataDf.loc[index] = resetData(allDataDf.loc[index])
        else:
            # pass in current timestamps cord and previous time stamps cord to distance function
            allDataDf['distDiff'][index] = distanceBetweenPoints(allDataDf['lng'][index], allDataDf['lng'][index - 1],
                                                                 allDataDf['lat'][index], allDataDf['lat'][index - 1])
            # time diff is the difference between the ith and the index-1 timestamp
            allDataDf['timeDiff'][index] = (((allDataDf['lastUpdated'][index - 1] - allDataDf['lastUpdated'][
                index]).total_seconds()) * secondsToMilliseconds)
            allDataDf.loc[index] = distTimeFromLastStop(allDataDf.loc[index], allDataDf.loc[index - 1])
        # timestamp is a stop
        if pd.notnull(allDataDf['currentStop'][index]):
            allDataDf.loc[index] = finalRelaventTimeStamp(allDataDf.loc[index], allDataDf.loc[firstIndexOfBus])
            repeatedBusesDf = pd.concat([repeatedBusesDf, allDataDf.loc[[index]]])
    return repeatedBusesDf


def processDistanceData(repeatedBusesDf):
    repeatedBusesDf['toStop'] = repeatedBusesDf['toStop'].astype("string")
    previousBusesDf = repeatedBusesDf.copy()
    repeatedBusesDf.set_index('routeId', inplace=True)
    previousBusesDf.set_index('busNumber', inplace=True)
    return repeatedBusesDf, previousBusesDf


def findRoutesPerStop():
    allRoutesThatGoToStops = {}
    # Iterate over the key set
    for key in routesDict.keys():
        # Pull the hashmap list value via the key
        for value in range(len(routesDict[key])):
            if routesDict[key][value] in allRoutesThatGoToStops.keys():
                allRoutesThatGoToStops[routesDict[key][value]].append(key)
            else:
                allRoutesThatGoToStops[routesDict[key][value]] = [key]
    return allRoutesThatGoToStops


def numStopsAway(route, toStop, targetStop):
    stopsAway: int = routesDict[route].index(targetStop) - routesDict[route].index(toStop)
    if stopsAway < 0:
        stopsAway += len(routesDict[route])
    return stopsAway


def fastestBus(repeatedBusesDf, allRoutesThatGoToStops):
    fastestBusMap = {}
    # key = stop
    for key in allRoutesThatGoToStops:
        for value in range(len(allRoutesThatGoToStops[key])):
            fastestBusMap[(key, allRoutesThatGoToStops[key][value])] = float("inf")
            # check to make sure a bus is on the route
            try:
                instance = repeatedBusesDf.loc[[allRoutesThatGoToStops[key][value]]]
            except:
                # if not, skip it
                continue
            else:
                # index = route
                for index, row in instance.iterrows():
                    if pd.isnull(row['toStop']):
                        continue
                    stopsAway: int = numStopsAway(index, row['toStop'], key)
                    if fastestBusMap[(key, allRoutesThatGoToStops[key][value])] == float("inf"):
                        fastestBusMap[(key, allRoutesThatGoToStops[key][value])] = [index, key, row['lastUpdated'],
                                                                                    row['timeFromStop'],
                                                                                    stopsAway,
                                                                                    row['distanceFromStop'], row['lng'],
                                                                                    row['lat']]
                    else:
                        if fastestBusMap[(key, allRoutesThatGoToStops[key][value])][4] > stopsAway:
                            fastestBusMap[(key, allRoutesThatGoToStops[key][value])] = [index, key,
                                                                                        row['lastUpdated'],
                                                                                        row['timeFromStop'],
                                                                                        stopsAway,
                                                                                        row['distanceFromStop'],
                                                                                        row['lng'], row['lat']]
                        if fastestBusMap[(key, allRoutesThatGoToStops[key][value])][4] == stopsAway and \
                                fastestBusMap[(key, allRoutesThatGoToStops[key][value])][5] < row['timeFromStop']:
                            fastestBusMap[(key, allRoutesThatGoToStops[key][value])] = [index, key,
                                                                                        row['lastUpdated'],
                                                                                        row['timeFromStop'],
                                                                                        stopsAway,
                                                                                        row['distanceFromStop'],
                                                                                        row['lng'], row['lat']]

    return fastestBusMap


def prevStopToTargetStop(busRoute, targetStop, lastUpdated, timeFromStop, stopsLeft, distFromStop):
    timeToTarget, avgTimeToTarget, distToTarget = 0, 0, 0
    # the index in the order that we want to get to
    targetIndex = routesDict[busRoute].index(targetStop)
    hourOfDay = lastUpdated.floor("H").hour
    dayOfWeek = lastUpdated.day_name()
    # the point we are coming from
    previousIndex = targetIndex - stopsLeft
    while previousIndex < targetIndex:
        # we are looking in our historical data for the median amount of time for from PI to PI +1
        avgTimeToTarget += (
            globalTimeGBDf.loc[
                (routesDict[busRoute][previousIndex], routesDict[busRoute][previousIndex + 1]), 'timeTaken'])
        distToTarget += (distGBDf.loc[
            (routesDict[busRoute][previousIndex], routesDict[busRoute][previousIndex + 1]), 'distBetweenStops'])
        timeToTarget += (timeByHourGBDf.loc[(routesDict[busRoute][previousIndex],
                                             routesDict[busRoute][previousIndex + 1], hourOfDay,
                                             dayOfWeek), 'timeTaken'])
        previousIndex += 1
    return avgTimeToTarget, distToTarget, timeToTarget, timeFromStop, distFromStop


def calculateTimeLeft(fastestBusMap):
    allStopsMapsList = []
    for key in fastestBusMap:
        if fastestBusMap[key] == float('inf'):
            allStopsMapsList.append(
                {'stop': key[0], 'routeId': key[1], 'timeLeft': -1, 'busHowLate': -1, 'trafficRatioGlobal': -1,
                 'timeFromStop': -1, 'distanceFromStop': -1})
            continue
        else:
            # param: busRoute, targetStop, currentStop, lastUpdated, timeFromStop, stopsLeft, distanceFromStop
            avgTimeToTarget, distToTarget, timeToTarget, timeFromStop, distFromStop = prevStopToTargetStop(
                fastestBusMap[key][0], fastestBusMap[key][1], fastestBusMap[key][2],
                fastestBusMap[key][3], fastestBusMap[key][4], fastestBusMap[key][5])
            distLeft = distToTarget - distFromStop
            # we are checking the raw distance to the stop, and comparing it to the distance along the road. if it is shorter, we go to the backup, because our distance prediction is invalid
            distLeftBackup = distanceBetweenPoints(stopsInfoDf.loc[(fastestBusMap[key][1]), 'longitude'],
                                                   fastestBusMap[key][6],
                                                   stopsInfoDf.loc[(fastestBusMap[key][1]), 'latitude'],
                                                   fastestBusMap[key][7])
            if distLeftBackup > distLeft:
                distLeft = distLeftBackup
            # if it's at a stop currently
            if timeToTarget == 0:
                milliSecondsLate = 0
                trafficRatioGlobal = 0
                millisecondsUntilStop = 0
            else:
                # assuming traffic will be normal after instance of traffic
                trafficRatioGlobal = timeToTarget / avgTimeToTarget
                millisecondsUntilStop = distLeft / (distToTarget / timeToTarget)
                # expected time - realTime
                milliSecondsLate = (distFromStop / (distToTarget / timeToTarget)) - timeFromStop
            # the last two parameters are meant to be passed back for the createARoute
        allStopsMapsList.append(
            {'stop': key[0], 'routeId': key[1], 'timeLeft': millisecondsUntilStop, 'milliSecondsLate': milliSecondsLate,
             'trafficRatioGlobal': trafficRatioGlobal, 'timeFromStop': timeFromStop, 'distanceFromStop': distFromStop})
    return allStopsMapsList


def timesToMongoDb(timeMapList):
    collection = db['timetostop']
    collection.delete_many({})
    for bus in range(len(timeMapList)):
        collection.insert_one(timeMapList[bus])
    return


def main():
    previousBusesDf = None
    numRecords = 5000
    firstTimeRun: bool = True
    while True:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if firstTimeRun:
                thread1 = executor.submit(callInTimingsData)
                thread2 = executor.submit(getRoutesAndStops)
                # returns global stopInfoDf and routesDict
                thread3 = executor.submit(callInData, numRecords)
                timingsDf = thread1.result()
                thread2.result()
                uniqueCombinations = uniqueStopsCombinations(timingsDf)
                filteredTimingsDf = filterTimeOutliers(timingsDf, uniqueCombinations)
                # makes global groupBy variables
                thread1 = executor.submit(makeGroupBy, filteredTimingsDf)
                thread2 = executor.submit(findRoutesPerStop)
                thread1.result()
                thread1 = executor.submit(tryDownloadingDistancesGroupBy, filteredTimingsDf)
                thread4 = executor.submit(timingsToDb)
            else:
                thread3 = executor.submit(callInData, numRecords)
            allDataDf = thread3.result()
            allDataDf = prepareData(allDataDf)
            repeatedBusesDf = findBusInformation(allDataDf, previousBusesDf, firstTimeRun)
            repeatedBusesDf, previousBusesDf = processDistanceData(repeatedBusesDf)
            allRoutesThatGoToStopsMap = thread2.result()
            fastestBusMap = fastestBus(repeatedBusesDf, allRoutesThatGoToStopsMap)
            # returns routesPerStop
            thread2.result()
            # returns stopGroupBy
            thread1.result()
            allStopsMapList = calculateTimeLeft(fastestBusMap)
            timesToMongoDb(allStopsMapList)
            firstTimeRun = False
            thread4.result()

            print(allStopsMapList)


if __name__ == "__main__":
    main()
