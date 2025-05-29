# this is to import the data type types without creating a circular import reference
from __future__ import annotations
from typing import TYPE_CHECKING, List, Set, Callable, Any
from enum import Enum
from pandas.api.types import is_numeric_dtype, is_string_dtype
from abc import ABC, abstractmethod
from aggregators import ColumnGroupedValues, string_minimum_function, string_maximum_function
import numpy as np

if TYPE_CHECKING:
    from data_table import DataTable
    from data_collection import DataCollection
    from pandas import DataFrame, Series


# note that the value is the aggregation type used by pandas
class DataColumnAggregation(Enum):
    SUM = "sum"
    AVG = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STDEV = "std"
    DISTINCT = "nunique"
    # group keeps all data, and groups the values per each grouping
    GROUP = "group"
    DEFAULT = None


def get_aggregation_function(aggregate_method: DataColumnAggregation,
                             col_dtype: np.dtype) -> Callable[[Series], Any] | str:
    # if grouping, then group via the aggregate series function
    if aggregate_method == DataColumnAggregation.GROUP:
        return ColumnGroupedValues.aggregate_series
    elif is_string_dtype(col_dtype):
        # if is a string, then can use custom min / max values
        # note that other values either don't make sense (like sum / avg)
        # or are already implemented (like count / distinct)
        if aggregate_method == DataColumnAggregation.MIN:
            return string_minimum_function
        elif aggregate_method == DataColumnAggregation.MAX:
            return string_maximum_function
    elif not is_numeric_dtype(col_dtype):
        # should something be done in this case?
        # for now, just run the default behavior
        pass
    # by default, just return the given value
    return aggregate_method.value


class DataColumnBase(ABC):
    def __init__(self, name: str, max_calculation_depth: int):
        super().__init__()
        self.name = name
        self._max_calculation_depth = max_calculation_depth

    # gets the maximum calculation depth before this column can be calculated
    @property
    def max_calculation_depth(self):
        return self._max_calculation_depth

    @abstractmethod
    def copy(self) -> DataColumnBase: ...


class DataColumnDimension(DataColumnBase):
    def __init__(self, name: str):
        # note that shou
        super().__init__(name, 0)

    def copy(self) -> DataColumnDimension:
        return DataColumnDimension(self.name)


class DataColumnMeasureBase(DataColumnBase, ABC):
    def __init__(self, name: str, max_calculation_depth: int):
        super().__init__(name, max_calculation_depth)
        self.used_by: List[DataColumnCalculated] = []


class DataColumn(DataColumnMeasureBase):
    # regular data column
    # assumes that when aggregates, will sum up the values
    def __init__(self, name, source: DataCollection, aggregation_method: DataColumnAggregation = DataColumnAggregation.DEFAULT):
        # since this column has data in it, then the max calculation depth is 0
        super().__init__(name, 0)
        self.source: DataCollection = source
        if aggregation_method == DataColumnAggregation.DEFAULT:
            # check the type of the given column. If is not number (i.e., text, date, etc), then return distinct count
            if source.df is not None and not is_numeric_dtype(source.df[name]):
                self.aggregation_method = DataColumnAggregation.DISTINCT
            else:
                self.aggregation_method = DataColumnAggregation.SUM
        else:
            self.aggregation_method = aggregation_method

    def copy(self) -> DataColumn:
        return DataColumn(self.name, self.source, self.aggregation_method)


class DataColumnCalculated(DataColumnMeasureBase):
    # calculated data column
    def __init__(self, name, input_columns: List[DataColumnBase], calc_func, replacement_value=0):
        # calculate the maximum depth for the input columns
        # add one since are dependent on the calculated max depth
        input_max_calculation_depth = max([c.max_calculation_depth for c in input_columns]) + 1
        super().__init__(name, input_max_calculation_depth)
        self._input_columns = input_columns
        self._calc_func = calc_func
        # values to replace after the calculation is complete
        self.replacement_values = {np.inf: replacement_value, -np.inf: replacement_value, np.nan: replacement_value}

    def calculate(self, source_data: DataFrame):
        # run the calculation
        # note that all input columns are guaranteed to be aggregated already
        # convert the columns (by name) over to the data frame column data, run the function, and set the output
        calcColData = self._calc_func(*[source_data[c.name] for c in self._input_columns])
        # replace any values (i.e., infinite, nan, or other values)
        if len(self.replacement_values) > 0:
            calcColData.replace(self.replacement_values, inplace=True)
        return calcColData

    def calculate_and_set(self, source_data: DataFrame):
        # run the calculation
        # note that all input columns are guaranteed to be aggregated already
        # convert the columns (by name) over to the data frame column data, run the function, and set the output
        source_data[self.name] = self.calculate(source_data)

    def copy(self) -> DataColumnCalculated:
        return DataColumnCalculated(self.name, self._input_columns, self._calc_func)


class DataColumnPercentile(DataColumnCalculated):
    def __init__(self, parent: DataColumn, percentile: int, replacement_value=0):
        assert 0 <= percentile <= 100, "Percentile must be an integer between 0 and 100"
        self.percentile = percentile
        self.percentile_float = percentile / 100
        self.parent = parent
        super().__init__(f"{parent.name} {percentile} Percentile", [parent], self.calculate_percentile, replacement_value)

    def calculate_percentile(self, col_value: Series):
        return col_value.transform(self.calculate_percentile_for_row)

    def calculate_percentile_for_row(self, row_val):
        return ColumnGroupedValues.calculate_percentile_for_row(row_val, self.percentile_float)
