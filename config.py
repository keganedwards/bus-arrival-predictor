# config.py
import os
from pymongo import MongoClient

# --- General Constants ---
SECONDS_TO_MILLISECONDS: int = 1000
EXCLUDED_ROUTES: list[int] = [26294, 3406]
PASSIO_RADIUS_CORRECTION_FACTOR: float = 0.85

# --- PassioGo API Configuration ---
PASSIO_API_URL = "https://passio3.com/www/mapGetData.php?getStops=2&deviceId=1720493&withOutdated=1&wBounds=1&showBusInOos=0&lat=35.3083779&lng=-80.7325179&wTransloc=1"
PASSIO_HEADERS = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
}
PASSIO_PAYLOAD = "json=%7B%22s0%22%3A%221053%22%2C%22sA%22%3A1%2C%22rA%22%3A8%2C%22r0%22%3A%223201%22%2C%22r1%22%3A%2222940%22%2C%22r2%22%3A%2226308%22%2C%22r3%22%3A%223406%22%2C%22r4%22%3A%223474%22%2C%22r5%22%3A%2216380%22%2C%22r6%22%3A%2226294%22%2C%22r7%22%3A%2235130%22%7D"

# --- MongoDB Configuration ---
MONGO_CREDENTIAL = os.environ.get("mongoCredential", "mongodb://localhost:27017/")
MONGO_CLIENT = MongoClient(MONGO_CREDENTIAL)
MONGO_DB = MONGO_CLIENT["busforce"]

# --- Collection Names ---
SNAPSHOTS_COLLECTION = "bussnapshots"
TIMINGS_COLLECTION = "stoptimings"
AVG_DISTANCES_COLLECTION = "averageDistancesBetweenStops"
AVG_TIMINGS_COLLECTION = "averageTimingsBetweenStops"
GLOBAL_TIMINGS_COLLECTION = "globalAverageTimings"
ETA_COLLECTION = "timetostop"
