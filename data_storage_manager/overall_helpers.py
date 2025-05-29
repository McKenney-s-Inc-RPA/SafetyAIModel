from typing import List, Type, Dict, Set, TypeVar, Any, Generic, TYPE_CHECKING, Callable, Iterable
from datetime import datetime
from pandas import DataFrame, Series
from bisect import bisect_right, bisect_left
from numpy import ndarray
# modules for profiling
from cProfile import Profile
from io import StringIO
from pstats import SortKey, Stats

VERBOSE_MODE = False

monthCol = "Month"
yearCol = "Year"
projectCol = "Project ID"
profitCenterTypeCol = "Profit Center Type"
profitCenterIdCol = "Profit Center ID"
branchCol = "Branch"
fiscalPeriodCol = "Fiscal Period"
weekCol = "Week"
dateCol = "Date"
craftCol = "Craft"
taskCodeIdCol = "Task ID"
taskCodeTypeIdCol = "Task Code Type"
hoursCol = "Hours"
numInjuriesCol = "# Injuries"
numRecordablesCol = "# Recordables"
employeeIdCol = "Employee ID"
foremanIdCol = "Foreman ID"
injuryRateCol = 'Risk per Man Year'
recordableRateCol = 'Recordable Risk per Man Year'
positionTypeCol = "Position Type"
positionCol = "Position"
possiblePredictors = [numInjuriesCol, numRecordablesCol, injuryRateCol, recordableRateCol]
# ignore anything that could indicate an injury (like master injury ID) or anything that could change (like employee IDs, projects, years, etc)
dimensionsToIgnoreInModel = [yearCol, projectCol, taskCodeIdCol, foremanIdCol, employeeIdCol, "Master Injury ID", "_Sources"]
commonDimensions = [monthCol, yearCol, projectCol, profitCenterTypeCol, branchCol, fiscalPeriodCol, weekCol, dateCol,
                    foremanIdCol, employeeIdCol, craftCol, taskCodeIdCol, taskCodeTypeIdCol, profitCenterIdCol]
commonNumericStringDimensions = [projectCol, foremanIdCol, employeeIdCol, taskCodeIdCol, taskCodeTypeIdCol]
profitCenterDesignationLookup = {'1': 'Service', '2': 'ACS', '3': 'CS', '4': 'ISP', '5': 'C15', '6': 'BPS', '7': 'ES',
                                 '8': 'C18', '9': 'HC', '0': "Unknown"}
defaultPercentiles = [10, 50, 90]
# declare the random seed, and allow to be released for random operation
# When set to null, allow all consumers of this seed to use their default value
as_deterministic = False
randomSeed = int(43) if as_deterministic else None

DECISION_TREE_max_depth = 5
DECISION_TREE_min_samples_leaf = 5
RANDOM_FOREST_num_trees = 11
RANDOM_FOREST_max_features = 0.7
RANDOM_FOREST_bootstrap = True
start_up_time = datetime.now()
log_as_elapsed_time = True
DEBUG_DEFAULT = True
REBUILD_CACHED_FILES = True
REBUILD_CACHED_FILES_BEFORE = datetime(2023, 7, 17)  # datetime.now()
CACHE_AS_PROCESS = True


def calculateRiskPerManYear(numInjuries, hours):
    return numInjuries / hours * 2000


def can_be_int(given_value: float):
    return (given_value % 1) == 0


class SingletonLatch:
    def __init__(self):
        self.needs_to_be_run = True

    def should_run(self):
        if self.needs_to_be_run:
            self.needs_to_be_run = False
            return True
        else:
            return False


def add_prefix_suffix_to_string(str_val, prefix: str | None, suffix: str | None):
    if prefix is not None and suffix is not None:
        return prefix + str_val + suffix
    elif prefix is not None:
        return prefix + str_val
    elif suffix is not None:
        return str_val + suffix
    else:
        return str_val


def convert_to_iterable(givenInput, underlyingType: Type) -> List:
    if givenInput is None:
        return []
    elif isinstance(givenInput, underlyingType):
        return [givenInput]
    elif isinstance(givenInput, list):
        return givenInput
    elif isinstance(givenInput, set) or isinstance(givenInput, tuple):
        return list(givenInput)
    else:
        raise NotImplementedError(f"Unknown item '{givenInput}' given to convert to list.")


def rename_key_in_dictionary(givenDictionary: Dict, oldKey, newKey):
    if oldKey in givenDictionary:
        if newKey in givenDictionary:
            raise ValueError(f"Key '{oldKey}' cannot be renamed to '{newKey}' as a key with that name already exists.")
        givenDictionary[newKey] = givenDictionary.pop(oldKey)


def rename_key_in_set(givenSet: Set, oldKey, newKey):
    if oldKey in givenSet:
        if newKey in givenSet:
            raise ValueError(f"Key '{oldKey}' cannot be renamed to '{newKey}' as an entry with that name already exists.")
        givenSet.remove(oldKey)
        givenSet.add(newKey)


T = TypeVar("T")
K = TypeVar("K")


def TryGetValue(givenDictionary: Dict[K, T], keyToFind: K) -> T:
    # returns None if not found
    # note that this can be combined with the walrus operator (:=) to get code that operates similarly
    # to the C# TryGetValue operator
    if keyToFind in givenDictionary:
        return givenDictionary[keyToFind]
    else:
        return None


def update_verbose_mode(newMode):
    global VERBOSE_MODE
    VERBOSE_MODE = newMode


def diagnostic_log_message(message):
    if VERBOSE_MODE:
        if log_as_elapsed_time:
            elapsed_time = datetime.now() - start_up_time
            total_seconds = elapsed_time.total_seconds()
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f'[{int(hours)}:{int(minutes):02}:{int(seconds):02}]: {message}')
        else:
            currTime = datetime.now().strftime("%H:%M:%S")
            print(f"[{currTime}]: {message}")


def group_data_frame(df: DataFrame, grouping_columns: List[str] | str, aggregations: Dict[str, Any], sort=False, dropna=False):
    # perform the group by and aggregation
    # sort is turned off to  increase efficiency of the code.
    # for more than one column, may need to turn sort off for deterministic behavior
    # don't drop na's, in case they are used later
    # use only observed items, so that if multiple grouping categorical columns are used, then doesn't create a cross-product of all options
    grouped_data = df.groupby(grouping_columns, dropna=dropna, sort=sort, observed=True)

    # next, perform the aggregation
    consolidated_data = grouped_data.agg(aggregations)

    # # if needed, then perform the sort now, before the index is reset
    # if sort:
    #     consolidated_data.sort_values(grouping_columns, inplace=True)

    # after the aggregation, will be a hierarchy index, so reset the index to split into the individual rows
    # reset the index to create the individual rows
    return consolidated_data.reset_index()


def order_data_frame_columns_alphabetically(df: DataFrame):
    return df.reindex(sorted(df.columns), axis=1)


HOURS_NORMALIZING_FACTOR = 200_000
hex_string = "0123456789ABCDEF"


def calculate_risk(num_injuries, num_hours):
    # normalize the risk over 100 full-time employees, similar to RIR or AICR
    return num_injuries / num_hours * HOURS_NORMALIZING_FACTOR


def calculate_injuries_from_risk(risk, num_hours):
    return risk * num_hours / HOURS_NORMALIZING_FACTOR


def convert_to_hex(value):
    # converts a 0 to 255 integer value to hexadecimal
    int_val = int(value)
    if int_val < 0 or int_val > 255:
        raise ValueError("Value outside of 0-255 range")
    ones_digit = int_val % 16
    sixteen_digit = int_val // 16
    return hex_string[sixteen_digit] + hex_string[ones_digit]


def range_one_based(total_length):
    return range(1, total_length + 1)


def profile_func_runtime(funcToRun: Callable[[], Any], file_name="training_runtime.csv", show_dirs=False):
    pr = Profile()
    pr.enable()
    funcResult = funcToRun()
    pr.disable()

    out_stream = StringIO()
    # get the statistics, and order by the calls done
    stats_data = Stats(pr, stream=out_stream)
    stats_data.sort_stats(SortKey.CALLS)
    if not show_dirs:
        stats_data.strip_dirs()
    stats_data.print_stats()
    analysisResult = out_stream.getvalue()
    # chop the string into a csv-like buffer
    analysisResult = 'ncalls' + analysisResult.split('ncalls')[-1]
    analysisResult = '\n'.join([','.join(line.rstrip().split(None, 5)) for line in analysisResult.split('\n')])

    with open(file_name, 'w') as f:
        f.write(analysisResult)
    return funcResult


class ModifySafeList(Generic[T]):
    def __init__(self, start_items: List[T] | None = None):
        # index for the iterator
        # start at -1, so that first iteration will move to the 0 index
        if start_items is None:
            start_items = []
        self._index = -1
        self._num_items = len(start_items)
        self._items = list(start_items)

    def _reset_index(self):
        self._index = -1

    def __iter__(self):
        self._reset_index()
        return self

    def __next__(self):
        # move forward to next index
        self._index += 1
        if self._index < self._num_items:
            # return the item at the current index
            item = self._items[self._index]
            return item
        else:
            # reset the index (in case loops through again), and tell to stop iterating
            self._reset_index()
            raise StopIteration

    def __len__(self):
        return self._num_items

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._items[item]
        else:
            raise KeyError(f"Index {item} not found in list...")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._items[key] = value
        else:
            raise KeyError(f"Index {key} not found in list...")

    def _apply(self, prior_state, apply_func: Callable[[T, Any], Any]):
        for v in self:
            prior_state = apply_func(v, prior_state)
        return prior_state

    def sum_using(self, attr_getter: Callable[[T], Any]):
        return self._apply(0, lambda v, s: attr_getter(v) + s)

    def __str__(self):
        return f"List of length {self._num_items}"

    def copy(self):
        return ModifySafeList(self._items)

    def remove(self, item):
        # assume that the item is within the list
        # remove from the list
        indexForRemoval = self._items.index(item)
        self._remove_at_index(indexForRemoval)

    def remove_where(self, func: Callable[[T], bool]):
        for s in self:
            if func(s):
                self.remove_current()

    def _remove_at_index(self, indexForRemoval):
        # if after the removed index, then subtract 1
        # if on the index, then move back one, so that next time will move to the
        #   next item in sequence (which is now at the same index)
        # if before the removed index, then nothing happens
        if self._index >= indexForRemoval:
            self._index -= 1
        self._num_items -= 1
        # pop out the item
        self._items.pop(indexForRemoval)

    def clear(self):
        self._num_items = 0
        self._items.clear()
        self._reset_index()

    def remove_current(self):
        if self._index >= 0:
            self._remove_at_index(self._index)
        else:
            raise KeyError(f"No items to remove")

    def append(self, item):
        # appends to the end of the list
        self._num_items += 1
        self._items.append(item)

    def insert(self, index, item):
        # inserts the given item at the given position
        # if are wanting to insert at or before, then move the index forward
        # note that this keeps the index at the current item, NOT the inserted item
        # if inserted after the current index, then don't need to do anything
        self._num_items += 1
        if index <= self._index:
            self._index += 1
        # insert within the list
        self._items.insert(index, item)

    def insert_after_current(self, item):
        # inserts the item as the next index
        # note that if the index is -1, then will insert at front
        # otherwise, will be able to insert anywhere in the list
        self.insert(self._index + 1, item)

    def insert_before_current(self, item):
        # insert before the current index
        # note that if the index is at the first item, or if the index is not being used, then insert location may be negative
        # if negative, then insert in the first position
        self.insert(max(0, self._index - 1), item)


class SortedList(Generic[T]):
    def __init__(self, heuristic: Callable[[T], float], is_ascending: bool = True, start_items: List[T] | None = None):
        # sorts items from lowest value to highest value, depending on the heuristic
        # assumes that no duplicates will be added, so therefore any duplicate heuristics will be added
        # if there is already a duplicate heuristic value, will be inserted to the right of (after) the current value,
        # so that follows fifo principle.
        # set up the underlying list
        self._items: list[T] = []
        # order of the heuristic values, so don't have to look up the same value every time
        self._heuristic_values: list[float] = []
        self._heuristic = heuristic
        self.is_ascending = is_ascending
        if start_items is not None:
            for s in start_items:
                self.add(s)

    def insert(self, insert_loc: int, value: T):
        # raise an error
        raise NotImplementedError()

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._items[item]
        else:
            raise KeyError(f"Index {item} not found in list...")

    def __len__(self):
        return len(self._items)

    def all_items(self):
        return self._items

    def _normalize_heuristic_value(self, given_value: float):
        # if is descending, then negate the values (so largest values are small, and smallest values are large)
        if self.is_ascending:
            return given_value
        else:
            return -given_value

    def add(self, item: T):
        # first, figure out the heuristic value
        heuristic_val = self._normalize_heuristic_value(self._heuristic(item))
        # next, figure out where should insert
        insert_loc = bisect_right(self._heuristic_values, heuristic_val)
        # insert at that location for the heuristic values
        self._heuristic_values.insert(insert_loc, heuristic_val)
        # insert at the location for the given item
        self._items.insert(insert_loc, item)

    def clear(self):
        # clear out the values and the heuristic
        self._items.clear()
        self._heuristic_values.clear()

    def remove(self, item: T):
        # find the first instance (will error out if not found, which is desired)
        found_index = self._items.index(item)
        # remove at the given position
        del self._heuristic_values[found_index]
        del self._items[found_index]

    def _remove_below_normalized(self, normalized_threshold: float):
        # the normalized threshold is the ascending heuristic value (same as given if ascending, negated if descending)
        remove_before = bisect_left(self._heuristic_values, normalized_threshold)
        if remove_before == len(self._items):
            # means that should just clear
            self.clear()
        elif remove_before > 0:
            self._items = self._items[remove_before:]
            self._heuristic_values = self._heuristic_values[remove_before:]

    def _remove_above_normalized(self, normalized_threshold: float):
        # the normalized threshold is the ascending heuristic value (same as given if ascending, negated if descending)
        remove_after = bisect_right(self._heuristic_values, normalized_threshold)
        if remove_after == 0:
            # means that should just clear
            self.clear()
        elif remove_after < len(self._items):
            self._items = self._items[:remove_after]
            self._heuristic_values = self._heuristic_values[:remove_after]

    def _get_below_normalized(self, normalized_threshold: float):
        # the normalized threshold is the ascending heuristic value (same as given if ascending, negated if descending)
        get_before = bisect_left(self._heuristic_values, normalized_threshold)
        # the value will be the index of the first item that is greater than the threshold (as can insert the threshold there)
        # therefore, select all items before it
        return self._items[:get_before]

    def _get_above_normalized(self, normalized_threshold: float):
        # the normalized threshold is the ascending heuristic value (same as given if ascending, negated if descending)
        get_after = bisect_right(self._heuristic_values, normalized_threshold)
        # the index will be the index of the first value greater than the threshold
        return self._items[get_after:]

    def remove_below(self, threshold_heuristic: float):
        # remove all items below the given value
        if self.is_ascending:
            # if ascending, then this means all values that occur before the given heuristic
            self._remove_below_normalized(threshold_heuristic)
        else:
            # if descending, then this means all values that occur after the given (negative) heuristic
            self._remove_above_normalized(-threshold_heuristic)

    def remove_above(self, threshold_heuristic: float):
        # remove all items below the given value
        if self.is_ascending:
            # if ascending, then this means all values that occur after the given heuristic
            self._remove_above_normalized(threshold_heuristic)
        else:
            # if descending, then this means all values that occur before the given (negative) heuristic
            self._remove_below_normalized(-threshold_heuristic)

    def get_above(self, threshold_heuristic: float):
        if self.is_ascending:
            return self._get_above_normalized(threshold_heuristic)
        else:
            return self._get_below_normalized(-threshold_heuristic)


class EasyDict(dict[K, T]):
    def __init__(self, initial_value: T | None = None):
        super().__init__()
        self._initial_value = initial_value

    def __getitem__(self, item):
        if item not in self:
            return self._initial_value
        else:
            return super().__getitem__(item)

    def add_to(self, key: K, value: T):
        if key in self:
            self[key] += value
        else:
            self[key] = value

    def __add__(self, other):
        if isinstance(other, dict):
            for k, v in other.items():
                self.add_to(k, v)
        return self

    def initialize(self, keys: Iterable[K]):
        for k in keys:
            if k not in self:
                self[k] = self._initial_value

    def print_by_value_desc(self):
        print("\t" + "\n\t".join([f"{k}: {round(v, 4)}"
                                  for k, v in sorted(self.items(),
                                                     key=lambda t: t[1],
                                                     reverse=True)]))


class ScoreValue:
    def __init__(self, starting_score=-100000, minimum_viable_score: float = 0, minimum_score_inclusive=False):
        self.score = starting_score
        self.minimum_viable_score = minimum_viable_score
        self.minimum_score_inclusive = minimum_score_inclusive

    def __str__(self):
        return str(self.score) if self.score > self.minimum_viable_score else "Empty"

    def update_score(self, new_score) -> bool:
        # now, check if is better than the current score
        is_better_than = self.can_update(new_score)
        if is_better_than:
            self.score = new_score
        return is_better_than

    def can_update(self, new_score):
        # check that the new score is better than the minimum
        if self.minimum_score_inclusive:
            # if inclusive, then can be greater than or equal to, but not less than
            if new_score < self.minimum_viable_score:
                return False
        else:
            # if not inclusive, then can be greater than, but not less than or equal to
            if new_score <= self.minimum_viable_score:
                return False

        # now, check if is better than the current score
        return new_score > self.score
