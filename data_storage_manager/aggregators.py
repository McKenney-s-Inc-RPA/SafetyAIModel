from __future__ import annotations
from pandas import Series, concat
from pandas.api.types import is_object_dtype, is_number


class ColumnGroupedValues:
    # retain values for a given column, even after aggregation
    # keeps the values until the data transformations are finalized, to keep data available until calculated
    # allows more complex aggregations, like percentile, to occur after multiple group by iterations
    def __init__(self, series_data: Series):
        self.series_data = series_data
        self.is_sorted = False

    @staticmethod
    def aggregate_series(given_series: Series) -> ColumnGroupedValues | int | float:
        # if there is only one value, then return the one value
        if len(given_series) == 1:
            first_index = given_series.index[0]
            return given_series.get(first_index)
        # since there is more than one value, compress into a column grouped values object
        elif is_object_dtype(given_series):
            # since is object type, then at least one of the values is the column grouped value
            # go through all values and combine into one series
            full_series_list = []
            # get all of the items that are stand-alone numbers
            numeric_mask = given_series.transform(is_number)
            numeric_series = given_series[numeric_mask]
            if len(numeric_series) > 0:
                full_series_list.append(numeric_series)
            # get all of the items that are not stand-alone numbers
            # these should be all instances of the grouped value class
            obj_indices = given_series.index[~numeric_mask]
            for i in obj_indices:
                obj_at_index = given_series.get(i)
                assert isinstance(obj_at_index, ColumnGroupedValues)
                full_series_list.append(obj_at_index)
            # now, combine into a full series, and set the value
            # set up the index to range from 0 to n
            return ColumnGroupedValues(concat(full_series_list, ignore_index=True))
        else:
            # means that is a series of integer or float types
            # reset the index of the given series to index by position
            # create a grouped value and return
            return ColumnGroupedValues(given_series.reset_index(drop=True))

    @staticmethod
    def calculate_percentile_for_row(row_value, percentile: float):
        if is_number(row_value):
            # if is a single number, then means that only has one record
            # because of this, return the value
            return row_value
        else:
            assert isinstance(row_value, ColumnGroupedValues)
            return row_value.series_data.quantile(percentile)


def string_minimum_function(given_series: Series) -> str:
    # if there is only one value, then return the one value
    if len(given_series) == 1:
        first_index = given_series.index[0]
        return given_series.get(first_index)
    # since there is more than one value, select the minimum value
    else:
        return given_series.min()


def string_maximum_function(given_series: Series) -> str:
    # if there is only one value, then return the one value
    if len(given_series) == 1:
        first_index = given_series.index[0]
        return given_series.get(first_index)
    # since there is more than one value, select the maximum value
    else:
        try:
            return given_series.fillna("").max()
        except Exception as ex:
            raise ex
