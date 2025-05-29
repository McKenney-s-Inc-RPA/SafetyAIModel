import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import googlemaps
from progress_bar import ProgressBar
from file_creator import driving_distance_folder
from overall_helpers import VERBOSE_MODE


# API key from Lloyd Anders. Restricted to api calls using the distance matrix.
google_maps_api_key = "AIzaSyAwAMzgkD33U82NMWzfLN1u2Ep0WN9bGTs"
gmaps = googlemaps.Client(key=google_maps_api_key)


# class to store the past distances
class DistanceLookup:
    originColName = 'Home Address'
    destinationColName = 'Project Address'
    distanceColName = 'Distance'
    durationColName = 'Duration'
    col_names = [originColName, destinationColName, distanceColName, durationColName]
    saveAfterNApiCalls = 50

    def __init__(self, file_name):
        # Empty Dictionary to store unique set of addresses
        self.lookup: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.api_calls_since_last_save = 0
        self.new_additions: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.file_name = file_name
        self.file_handler = driving_distance_folder.get_data_source(self.file_name, creator_func=DistanceLookup.col_names)
        self.use_api_to_lookup = False
        # If File exists, then read in.
        if VERBOSE_MODE:
            print("Reading File")
        loaded_file = self.file_handler.get_data()
        self.distances_df = loaded_file.df
        for index, row in self.distances_df.iterrows():
            origin = row[DistanceLookup.originColName]
            destination = row[DistanceLookup.destinationColName]
            distance = row[DistanceLookup.distanceColName]
            duration = row[DistanceLookup.durationColName]
            self._add_address_to_lookup(origin, destination, distance, duration)

    def _add_address_to_lookup(self, origin, destination, distance, duration):
        self.lookup.setdefault(origin, {})[destination] = (distance, duration)

    def add_address(self, origin, destination, distance, duration):
        # add to the lookup
        self._add_address_to_lookup(origin, destination, distance, duration)
        # add to the new additions
        self.new_additions.setdefault(origin, {})[destination] = (distance, duration)

    def save_to_file(self):
        # append the new additions to the distances dataframe
        if VERBOSE_MODE:
            print("Saving Prior Distances File")
        if len(self.new_additions) > 0:
            addition_dict: Dict[str, List] = {DistanceLookup.originColName: [],
                             DistanceLookup.destinationColName: [],
                             DistanceLookup.distanceColName: [],
                             DistanceLookup.durationColName: []}
            for origin, destination_lookup in self.new_additions.items():
                for destination, (distance, duration) in destination_lookup.items():
                    addition_dict[DistanceLookup.originColName].append(origin)
                    addition_dict[DistanceLookup.destinationColName].append(destination)
                    addition_dict[DistanceLookup.distanceColName].append(distance)
                    addition_dict[DistanceLookup.durationColName].append(duration)
            new_addition_df = pd.DataFrame(addition_dict)
            self.distances_df = pd.concat([self.distances_df, new_addition_df])
        self.file_handler.set_data(self.distances_df, build_files=True)
        self.new_additions = {}
        self.api_calls_since_last_save = 0
        if VERBOSE_MODE:
            print("Prior Distances File Saved")

    def find_metrics(self, origin, destination):
        if lookup_by_destination := self.lookup.get(origin, None):
            if trip_data := lookup_by_destination.get(destination, None):
                return trip_data
        # use the api to find the data
        if self.use_api_to_lookup:
            distance_result = gmaps.distance_matrix(origin, destination)['rows'][0]['elements'][0]  # Calling API
            distance_meters = distance_result['distance']['value']  # Specifying and sorting results
            duration_seconds = distance_result['duration']['value']
            distance_miles = distance_meters * 0.00062137
            duration_minutes = duration_seconds / 60
            # Assigns values (dist & dur) to the key in dict
            self.add_address(origin, destination, distance_miles, duration_minutes)
            self.api_calls_since_last_save += 1
            if self.api_calls_since_last_save > DistanceLookup.saveAfterNApiCalls:
                self.save_to_file()
            return distance_miles, duration_minutes
        else:
            return -1, -1


# Read the cached data
if VERBOSE_MODE:
    print("Reading the prior cached data")
past_distances: DistanceLookup = DistanceLookup("past_distances.csv")

# read in the data file
if VERBOSE_MODE:
    print("Reading the data")
df = pd.read_excel("Employee Working Distance.xlsx", header=0)
past_distances.use_api_to_lookup = False
page_size = 100 if past_distances.use_api_to_lookup else 1000
num_records = len(df)
# set up the columns to write to
distanceArr = -1 * np.ones(num_records)
durationArr = -1 * np.ones(num_records)

progress = ProgressBar(num_records)
originIndex = df.columns.get_loc('Home Address')
destIndex = df.columns.get_loc('Project Address')
for p in range(0, num_records, page_size):
    max_index = min(p+page_size, num_records)
    for i in range(p, min(p+page_size, num_records)):
        dist, dur = past_distances.find_metrics(df.iloc[i, originIndex], df.iloc[i, destIndex])
        distanceArr[i] = dist
        durationArr[i] = dur
    progress.moveTo(max_index)
progress.finish()
df['Distance'] = distanceArr
df["Duration"] = durationArr


def find_distance(row):
    return past_distances.find_metrics(row['Home Address'], row['Project Address'])


# Run functions to create dist and dur columns in df
# unpacks iterables from find_distance into df columns
# df['Distance'], df['Duration'] = zip(*df.apply(find_distance, axis=1))
if VERBOSE_MODE:
    print("Saving the distances driven")
df.to_excel("Employee Working Distances Updated.xlsx", index=False)
past_distances.save_to_file()

