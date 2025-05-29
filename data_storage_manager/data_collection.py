# the annotations and typing is to check for the data table without creating a import reference loop
from __future__ import annotations
from typing import List, Dict, Tuple, Set, TYPE_CHECKING, Iterable
from warnings import warn

import numpy as np

from progress_bar import ProgressBar
import pandas as pd
import pantab
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_number
from data_column_manager import DataColumnManager
from data_columns import *
from overall_helpers import *
from os.path import exists as path_exists, basename as file_basename


class DataCollectionPartitionJoinType(Enum):
    INTERSECT = 1
    UNION = 2


class JoinType(Enum):
    OUTER = "outer"
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"


def get_base_file_name_without_ext(file_path):
    # Get the base name of the file
    filename = file_basename(file_path)
    # Use slicing to remove the extension
    filename_without_ext = filename[:filename.rindex('.')]
    return filename_without_ext


class DimensionSet:
    def __init__(self, dimensions_sharing: DataCollectionPartitionJoinType = DataCollectionPartitionJoinType.INTERSECT):
        self._nothing_added_yet = True
        self.dimensions_sharing = dimensions_sharing
        self.dimensions: Set[str] = set()

    def add(self, other_set: Set[str]):
        # if nothing added, or if are unioning, then add in all items
        if self._nothing_added_yet or self.dimensions_sharing == DataCollectionPartitionJoinType.UNION:
            self.dimensions.update(other_set)
            self._nothing_added_yet = False
        else:
            # otherwise, run an intersection
            self.dimensions.intersection_update(other_set)


class DataCollection(DataColumnManager):
    def __init__(self, df: pd.DataFrame, name: str | None = None):
        super().__init__()
        self.df: pd.DataFrame = df
        self.name = name
        self.headers: Set[str]
        self._update_headers()

    def __str__(self):
        defaultStr = f"{super().__str__()}\nRows: {len(self.df)}\nSample: \n{self.df}"
        if self.name is not None:
            return f"{self.name}\n{defaultStr}"
        else:
            return defaultStr

    def copy(self, new_name: str | None = None):
        copyVal = DataCollection(self.df.copy(True), new_name if new_name is not None else self.name)
        copyVal._copy_from(self)
        return copyVal

    def _merge_from(self, other_data: DataCollection):
        # copies the data from the other collection onto this collection without changing the data
        # copy over the name
        self.name = other_data.name
        # copy the underlying data first, so can see if any columns aren't present when copying over
        self.df = other_data.df.copy(True)
        self._update_headers()
        # copy over the column information
        super()._clear()
        super()._copy_from(other_data)

    def _update_headers(self):
        self.headers = set(self.df.columns.values.tolist())

    def set_data_frame(self, new_df: pd.DataFrame):
        self.df = new_df
        self._update_headers()

    def return_columns_in_data_table(self, possible_cols: list[str] | str | None):
        return [c for c in convert_to_iterable(possible_cols, str) if c in self.headers]

    def get_part_of_table(self, possible_dimensions: List[str], possible_measures: List[str]):
        # create the new sub-section
        tablePart = DataCollection(self.df)

        # go through the dimensions, and add the columns if found
        any_dimensions_found = False
        for d in possible_dimensions:
            if self.has_dimension(d):
                any_dimensions_found = True
                tablePart.add_dimension_col(self.dimension_lookup[d])
        assert any_dimensions_found, "None of the given dimensions were found"

        # go through the measures, and add the columns if found
        any_measures_found = False
        for d in possible_measures:
            if self.has_measure(d):
                any_measures_found = True
                tablePart.add_measure_col(self.measures_lookup[d])
        assert any_measures_found, "None of the given measures were found"

        # build out the data frame
        tablePart.df = tablePart.consolidate_data()
        return tablePart

    def _check_if_column_in_data_set(self, col_name: str, warn_if_not_found: bool = True):
        if col_name not in self.headers:
            if warn_if_not_found:
                givenName = self.name if self.name is not None else "given data"
                warn(f"The given column '{col_name}' does not exist in {givenName}.", RuntimeWarning)
            return False
        return True

    def _add_dimension(self, col_name: str, warn_if_not_found: bool = True):
        if self._check_if_column_in_data_set(col_name, warn_if_not_found):
            super()._add_dimension(col_name)

    def convert_numeric_dimension_to_str(self, col_name: str, warn_if_not_found: bool = True):
        if self._check_if_column_in_data_set(col_name, warn_if_not_found):
            givenCol = self.df[col_name]
            if is_numeric_dtype(givenCol):
                self.df[col_name] = givenCol.astype(int).astype(str)

    def convert_numeric_dimensions_to_str(self, col_names: List[str], warn_if_not_found: bool = True):
        for d in col_names:
            self.convert_numeric_dimension_to_str(d, warn_if_not_found)

    def set_dimension_data_columns(self, data_col_names: str | List[str], df_cols):
        # set the column values
        self.df[data_col_names] = df_cols
        # add to the headers
        for n in convert_to_iterable(data_col_names, str):
            self._add_dimension_data_column_name(n)

    def add_percentiles(self, col_name: str, percentile_values: List[int] = defaultPercentiles,
                        aggregate_on_column_copy: bool = False):
        # check to see if the given column is marked to be aggregated yet
        sourceCol = self._get_raw_clustered_aggregation_column(col_name, aggregate_on_column_copy)

        # for each of the percentiles, add a calculated column that takes the aggregated data and returns a percentile
        for p in percentile_values:
            self.add_measure_col(DataColumnPercentile(sourceCol, p))

    def _add_dimension_data_column_name(self, data_col_name: str):
        self.headers.add(data_col_name)
        self._add_dimension(data_col_name)

    def set_measure_data_column(self, data_col_name: str, df_col, aggregation_method=DataColumnAggregation.DEFAULT):
        # set the column values
        self.df[data_col_name] = df_col
        # add to the headers
        self.headers.add(data_col_name)
        if data_col_name not in self.measure_names:
            # if not created, then set up as a data column
            self.add_measure(data_col_name, aggregation_method=aggregation_method)

    def set_measure_data_columns(self, data_col_names: List[str], df_cols: DataFrame, aggregation_method=DataColumnAggregation.DEFAULT):
        # set the column values
        self.df[data_col_names] = df_cols
        # add to the headers
        self.headers.update(data_col_names)
        for data_col_name in data_col_names:
            if data_col_name not in self.measure_names:
                # if not created, then set up as a data column
                self.add_measure(data_col_name, aggregation_method=aggregation_method)

    def add_measure(self, col_name: str, warn_if_not_found: bool = True,
                    aggregation_method=DataColumnAggregation.DEFAULT):
        # only allow to create raw data columns on a data table
        if self._check_if_column_in_data_set(col_name, warn_if_not_found):
            # all others will be copied over
            if self.has_measure(col_name):
                # set the aggregate type
                existingMeasCol = self.measures_lookup[col_name]
                if isinstance(existingMeasCol, DataColumn) and aggregation_method != DataColumnAggregation.DEFAULT:
                    existingMeasCol.aggregation_method = aggregation_method
                return existingMeasCol

        # create the column, and add
        measCol = DataColumn(col_name, self, aggregation_method)
        self._add_measure_column(measCol)
        return measCol

    def _get_raw_clustered_aggregation_column(self, col_name: str,
                                              aggregate_on_column_copy: bool = False) -> DataColumn:
        if self._check_if_column_in_data_set(col_name):
            if aggregate_on_column_copy:
                # means that will need to create a dummy copy of the column, if doesn't already exist
                if col_name in self.dummy_data_name_lookup:
                    return self.dummy_data_name_lookup[col_name]
                else:
                    dummy_col_name = col_name + " _GROUPED"
                    self.set_measure_data_column(dummy_col_name, self.df[col_name])
                    dummy_col = self.data_measures_lookup[dummy_col_name]
                    self.dummy_data_name_lookup[col_name] = dummy_col
                    return dummy_col
            else:
                # means that can override the value, if needed
                measCol = self.data_measures_lookup[col_name]
                measCol.aggregation_method = DataColumnAggregation.GROUP
                return measCol

    def add_dimension_col(self, given_col: DataColumnDimension):
        self._add_dimension(given_col.name)

    def add_measure_col(self, given_col: DataColumnMeasureBase):
        if isinstance(given_col, DataColumn):
            self.add_measure(given_col.name, aggregation_method=given_col.aggregation_method)
        elif isinstance(given_col, DataColumnCalculated):
            # add the calculated column to the lookups
            self._add_measure_column(given_col)
            # perform the calculation
            calcColData = given_col.calculate(self.df)
            # save off the result
            self.set_measure_data_column(given_col.name, calcColData)

    def add_dimensions(self, col_names: List[str] | str, warn_if_not_found: bool = True):
        for currCol in convert_to_iterable(col_names, str):
            self._add_dimension(currCol, warn_if_not_found)

    def add_measures(self, col_names: List[str], warn_if_not_found: bool = True,
                     aggregation_method=DataColumnAggregation.DEFAULT):
        for currCol in col_names:
            self.add_measure(currCol, warn_if_not_found, aggregation_method)

    def consolidate_data_and_replace(self):
        # replaces the existing data frame with a consolidated version, where all of the measures are aggregated
        self.df = self.consolidate_data()
        self._update_headers()

    def remove_data_columns(self, data_col_names: List[str], perform_consolidation=True):
        # drop the columns
        self.df.drop(columns=data_col_names, inplace=True)
        for n in data_col_names:
            self._remove_data_column__already_removed_from_data(n, False)
        if perform_consolidation:
            self.consolidate_data_and_replace()

    def keep_data_columns(self, data_col_names: List[str], perform_consolidation=True):
        # figure out the columns to remove
        colsToKeep = set(data_col_names)
        colsToRemove = [c for c in self.headers if c not in colsToKeep]
        self.remove_data_columns(colsToRemove, perform_consolidation)

    def keep_dimensions(self, data_col_names: List[str], perform_consolidation=True):
        colsToKeep = set(data_col_names)
        colsToRemove = [c for c in self.dimension_names if c not in colsToKeep]
        self.remove_data_columns(colsToRemove, perform_consolidation)

    def keep_measures(self, data_col_names: List[str]):
        colsToKeep = set(data_col_names)
        colsToRemove = [c for c in self.measure_names if c not in colsToKeep]
        self.remove_data_columns(colsToRemove, False)

    def pivot_dimension(self, dimension_to_pivot: str, value_measure: str):
        # converts one dimension column and one measure column into multiple measure columns,
        # each named after the unique values within the dimension
        # returns the names of the grouped values
        # group by the remaining dimensions
        assert self.has_dimension(dimension_to_pivot)
        assert self.has_measure(value_measure)
        remaining_dimensions = list(self.dimension_names.difference([dimension_to_pivot]))
        # pivot the data by the dimension and the values, and set all na's to zero
        result = self.df.pivot(columns=dimension_to_pivot, values=value_measure, index=remaining_dimensions)
        result.reset_index(inplace=True)
        result.replace(to_replace=np.nan, value=0, inplace=True)
        # remove the dimension and measure from the prior data set
        # don't consolidate yet, as will only need to consolidate if there are other measures
        self.remove_data_columns([value_measure, dimension_to_pivot], False)
        if len(self.measure_names) > 0:
            # consolidate the data
            self.consolidate_data_and_replace()
            self.df = self.df.merge(result, how="inner", on=remaining_dimensions)
        else:
            # since no other measures exists, then can set the data frame outright
            self.df = result
        # update the headers, and add the measures
        self._update_headers()
        new_measures = [c for c in result.columns if c not in self.dimension_names]
        self.add_measures(new_measures)
        return new_measures

    def _remove_data_column__already_removed_from_data(self, data_col_name: str, perform_consolidation=True):
        # remove from the headers
        self.headers.remove(data_col_name)
        # remove from the dimensions / measures
        if self.has_dimension(data_col_name):
            self._remove_dimension(data_col_name)
            # note that only need to consolidate if are removing a dimension
            if perform_consolidation:
                self.consolidate_data_and_replace()
        elif self.has_measure(data_col_name):
            self._remove_measure(data_col_name)

    def remove_data_column(self, data_col_name: str, perform_consolidation=True):
        # remove from the data
        self.df.drop(columns=data_col_name, inplace=True)
        self._remove_data_column__already_removed_from_data(data_col_name, perform_consolidation)

    def rename_measures(self, prefix: str | None = None, suffix: str | None = None):
        if prefix is not None or suffix is not None:
            self.rename_columns({m: add_prefix_suffix_to_string(m, prefix, suffix) for m in self.measure_names})

    def rename_columns(self, changes: Dict[str, str]):
        # key is the old name
        # value is the new name
        # check that the old name exists, the new name is not already used, and the old name is different from the new name
        validated_changes = {k: v for k, v in changes.items() if k in self.headers and k != v and v not in self.headers}
        if len(validated_changes) > 0:
            # renames in the data frame and on the column handler
            # rename the data frame
            self.df.rename(columns=validated_changes, inplace=True)
            for oldName, newName in validated_changes.items():
                self._rename_column(oldName, newName)

    def rename_column(self, old_name, new_name):
        self.rename_columns({old_name: new_name})

    def _rename_column(self, old_name, new_name):
        # rename within the column handler
        super()._rename_column(old_name, new_name)
        # rename in the headers
        rename_key_in_set(self.headers, old_name, new_name)

    def filter_remove_before_year(self, year_to_filter_after: int):
        if yearCol in self.headers:
            self.filter_where(self.df[yearCol] >= year_to_filter_after)

    def filter_where(self, true_values):
        self.df = self.df.loc[true_values]

    def filter(self, col_name: str):
        return DataCollectionFilter(self, col_name)

    def _reduce_dimensions_if_needed(self, new_dimension_set):
        # if all dimensions are included within the new dimension set, then return itself
        if self.dimension_names.issubset(new_dimension_set):
            return self
        # otherwise, pare down the dimensions, and return the partition
        dimensionsToKeep = self.dimension_names.intersection(new_dimension_set)
        return self.get_part_of_table(list(dimensionsToKeep), list(self.measure_names))

    def copy_measures_from_source(self, source_partition: DataCollection, suffix: str = ''):
        for measCol in source_partition.measures_lookup.values():
            copyMeasCol = measCol.copy()
            copyMeasCol.name += suffix
            self.add_measure_col(copyMeasCol)

    def _set_data_remove_extra_columns(self, df: DataFrame):
        # see which columns should keep
        colsToExclude = [c for c in df.columns if c not in self.headers]
        self.df = df.drop(columns=colsToExclude)

    def _drop_columns(self, columns: List[str] | str):
        self.df.drop(columns=[c for c in convert_to_iterable(columns, str) if c in self.df.columns], inplace=True)

    def join_with_fallbacks(self, other_partition: DataCollection, joining_dimensions: List[List[str]],
                            join_type: JoinType = JoinType.OUTER,
                            left_suffix: str = '', right_suffix: str = '',
                            extra_dimensions: List[str] | str | None = None,
                            extra_dimensions_left: List[str] | str | None = None,
                            extra_dimensions_right: List[str] | str | None = None,
                            source_indicator: bool = False,
                            left_source_name: None | str = None,
                            right_source_name: None | str = None) -> DataCollection:
        # joins the data together using the most strict matches, then works way down to least strict matches
        # for most strict matches, the join is outer joined,
        # and results pertaining to both are appended to the outer result
        # after that, the leftovers from the left and right side are isolated,
        # and then the join is performed again using the next most strict match
        # the final match is performed with the given join type
        match_columns_by_strictness: List[List[str]] = []
        # given a set of columns with increasing strictness (like [[A, B], [C], [D]]
        # make the sets of columns with decreasing strictness (like [[A, B, C, D], [A, B, C], [A, B]])
        columns_seen_so_far = set()
        for col_list in joining_dimensions:
            columns_seen_so_far.update(col_list)
            match_columns_by_strictness.insert(0, list(columns_seen_so_far))
        assert len(match_columns_by_strictness) > 0, "No Joining Dimensions given..."

        # create the joining helper
        join_helper = JoinHelper(self, other_partition, left_suffix, right_suffix, source_indicator, left_source_name, right_source_name)
        join_helper.set_up_extra_dimensions(extra_dimensions, extra_dimensions_left, extra_dimensions_right)

        # now, go through these columns and incrementally join the data together
        last_link_index = len(match_columns_by_strictness) - 1

        joined_df = DataFrame()
        for i in range(last_link_index):
            join_helper.set_shared_dimensions(match_columns_by_strictness[i])
            resulting_df = join_helper.join_data_tables(JoinType.OUTER, True)
            source_data = resulting_df[DataCollection.indicator_col]
            # add the items that were matched by both to the other data that has been joined
            from_both = DataCollection._is_from_both_sources(source_data)
            both_data = resulting_df.loc[from_both]
            joined_df = pd.concat([joined_df, both_data], axis=0)
            # set the unmatched values of the left and right partitions back to the partitions
            # remove any columns that were not a part of the original data sets
            join_helper.left.partition._set_data_remove_extra_columns(resulting_df.loc[DataCollection._is_from_left_source(source_data)])
            join_helper.right.partition._set_data_remove_extra_columns(resulting_df.loc[DataCollection._is_from_right_source(source_data)])

        # if don't want to use the source indicator, then drop the indicator column
        if not source_indicator:
            joined_df.drop(columns=[DataCollection.indicator_col], inplace=True)
            join_helper.left.partition._drop_columns(DataCollection.indicator_col)
            join_helper.right.partition._drop_columns(DataCollection.indicator_col)

        # join the last data frame
        join_helper.set_shared_dimensions(match_columns_by_strictness[last_link_index])
        resulting_df = join_helper.join_data_tables(join_type)
        joined_df = pd.concat([joined_df, resulting_df], axis=0)

        return join_helper.create_output_data(joined_df)

    right_source_col = "_Right Sources"
    left_source_col = "_Left Sources"
    source_col_name = "_Sources"
    indicator_col = "_merge"
    from_right_source = "right_only"
    from_left_source = "left_only"

    def join_simple(self, other_partition: DataCollection, join_type: JoinType = JoinType.OUTER,
                    left_suffix: str = '', right_suffix: str = '',
                    source_indicator: bool = False,
                    left_source_name: None | str = None,
                    right_source_name: None | str = None) -> DataCollection:
        # figure out which dimensions are shared
        joining_dimensions = list(self.dimension_names.intersection(other_partition.dimension_names))
        extra_dimensions_left = list(self.dimension_names.difference(joining_dimensions))
        extra_dimensions_right = list(other_partition.dimension_names.difference(joining_dimensions))
        return self.join(other_partition, join_type=join_type, left_suffix=left_suffix, right_suffix=right_suffix,
                         joining_dimensions=joining_dimensions, ignore_dimensions=None,
                         extra_dimensions_left=extra_dimensions_left, extra_dimensions_right=extra_dimensions_right,
                         source_indicator=source_indicator, left_source_name=left_source_name, right_source_name=right_source_name)

    def join_simple_inplace(self, other_partition: DataCollection, join_type: JoinType = JoinType.LEFT,
                            left_suffix: str = '', right_suffix: str = '',
                            source_indicator: bool = False,
                            left_source_name: None | str = None,
                            right_source_name: None | str = None):
        # warn on any mismatched dimensions that may duplicate rows
        # check that the kept destination (i.e., side of join kept) does not inflate from the joined-in data
        # note that should be joining data that is less specific or as specific as the left data
        # for example, the destination could have project and employee data, and the data being merged in may only have project data
        # however, if the data merged in has craft, project, and employee, then could cause bloat in the destination data set
        if join_type == JoinType.LEFT:
            # if merging left, then are joining data from right that is less specific or as specific as the left data
            dimsNotInThis = other_partition.dimension_names.difference(self.dimension_names)
            # warn if there are any dimensions in the other partition not in this dimension
            if len(dimsNotInThis) > 0:
                warn(f"Partition '{other_partition.name}' has dimensions {dimsNotInThis} that are not in '{self.name}'")
        elif join_type == JoinType.RIGHT:
            dimsNotInOther = self.dimension_names.difference(other_partition.dimension_names)
            # warn if there are any dimensions in this not in the other dimension
            if len(dimsNotInOther) > 0:
                warn(f"Partition '{self.name}' has dimensions {dimsNotInOther} that are not in '{other_partition.name}'")
        elif join_type == JoinType.OUTER or join_type == JoinType.INNER:
            # warn if there are any dimensions not in both
            allDims = other_partition.dimension_names.union(self.dimension_names)
            sharedDims = other_partition.dimension_names.intersection(self.dimension_names)
            nonSharedDims = allDims.difference(sharedDims)
            if len(nonSharedDims) > 0:
                warn(f"{nonSharedDims} not in both Partitions '{other_partition.name}' and '{self.name}'")
        # keeps all of the data from this collection (i.e., joins left by default)
        join_result = self.join_simple(other_partition, join_type, left_suffix, right_suffix, source_indicator, left_source_name, right_source_name)
        # copy into this result
        self._merge_from(join_result)

    def join(self, other_partition: DataCollection, join_type: JoinType = JoinType.OUTER,
             left_suffix: str = '', right_suffix: str = '',
             joining_dimensions: List[str] | str | None = None,
             ignore_dimensions: List[str] | str | None = None,
             extra_dimensions: List[str] | str | None = None,
             extra_dimensions_left: List[str] | str | None = None,
             extra_dimensions_right: List[str] | str | None = None,
             source_indicator: bool = False,
             left_source_name: None | str = None,
             right_source_name: None | str = None) -> DataCollection:
        # joins the two partitions together on shared columns
        # first, figures out what columns to share
        # note that any columns not shared will be dropped, and then the source will be aggregated
        # any measures will be carried over (as should be be on only measures)
        join_helper = JoinHelper(self, other_partition, left_suffix, right_suffix, source_indicator, left_source_name, right_source_name)
        join_helper.set_up_extra_dimensions(extra_dimensions, extra_dimensions_left, extra_dimensions_right)
        join_helper.set_shared_dimensions(joining_dimensions, ignore_dimensions)
        resultingDf = join_helper.join_data_tables(join_type)
        return join_helper.create_output_data(resultingDf)

    def _rename_unshared_dimensions_and_measures(self, shared_dimensions: Set[str], dimensions_to_keep: Set[str], prefix: str = "", suffix: str = ""):
        # creates a list to rename
        all_renames = {}
        for c in self.measure_names:
            all_renames[c] = prefix + c + suffix
        for c in self.dimension_names:
            if c not in shared_dimensions:
                new_dim_name = prefix + c + suffix
                all_renames[c] = new_dim_name
                # if is a dimension to keep, then update
                if c in dimensions_to_keep:
                    dimensions_to_keep.remove(c)
                    dimensions_to_keep.add(new_dim_name)
        self.rename_columns(all_renames)

    @staticmethod
    def _is_from_right_source(source_data: Series):
        return source_data == "right_only"

    @staticmethod
    def _is_from_left_source(source_data: Series):
        return source_data == "left_only"

    @staticmethod
    def _is_from_both_sources(source_data: Series):
        return source_data == "both"

    def rename_indicator_column(self):
        source_data = self.df[DataCollection.indicator_col]
        left_source_data = self.df[DataCollection.left_source_col]
        right_source_data = self.df[DataCollection.right_source_col]
        # by default, set up as both, and use comma separated to join together
        self.set_measure_data_column(DataCollection.source_col_name, left_source_data + ", " + right_source_data,
                                     aggregation_method=DataColumnAggregation.MIN)
        self.df[DataCollection.source_col_name].mask(DataCollection._is_from_left_source(source_data), left_source_data, inplace=True)
        self.df[DataCollection.source_col_name].mask(DataCollection._is_from_right_source(source_data), right_source_data, inplace=True)
        self.df.drop(columns=[DataCollection.indicator_col])
        self.remove_data_columns([DataCollection.right_source_col, DataCollection.left_source_col])

    def add_calculated_field(self, col_name: str, input_columns: List[DataColumnBase | str], calc_func, replacement_value=0):
        givenColumns = []
        for c in input_columns:
            if isinstance(c, DataColumnBase):
                givenColumns.append(c)
            elif isinstance(c, str):
                givenColumns.append(self.column_lookup[c])
            else:
                raise NotImplementedError("Unknown type given")

        # create the column
        newCalcCol = DataColumnCalculated(col_name, givenColumns, calc_func, replacement_value)
        # add to this calculated column set
        self.add_measure_col(newCalcCol)

    def create_lookup(self, key_columns: str, value_column: str):
        collectionCopy = self.copy()
        collectionCopy.keep_data_columns([key_columns, value_column])
        return {row[key_columns]: row[value_column] for idx, row in collectionCopy.df.iterrows()}

    def randomly_select_columns(self, proportion_to_select: float = 0.7):
        if not (0 < proportion_to_select <= 1):
            raise ValueError("The given proportion needs to be in the range (0, 1]")

    def create_common_calculated_dimensions(self, remove_dates=True, perform_consolidation=True):
        if self.has_dimension(dateCol):
            self.split_date(dateCol, remove_dates, False)
        if self.has_dimension(weekCol):
            self.split_date(weekCol, remove_dates, False)
        if self.has_dimension(fiscalPeriodCol):
            self.split_fiscal_period(fiscalPeriodCol, False)
        if self.has_dimension(projectCol):
            self.split_project_number()
        if self.has_dimension(taskCodeIdCol):
            self.split_task_code(False)
        if perform_consolidation:
            self.consolidate_data_and_replace()

    def _convert_column_to_date(self, col_name: str):
        date_data = self.df[col_name]
        # convert to a date/time column
        if not is_datetime64_any_dtype(date_data):
            date_data = pd.to_datetime(date_data)
            self.df[col_name] = date_data
        return date_data

    def convert_to_date(self, date_col: str | List[str]):
        for c in convert_to_iterable(date_col, str):
            self._convert_column_to_date(c)

    def convert_to_week(self, date_col: str, week_col_name: str = weekCol, remove_date: bool = True):
        # should convert over to the week starting on the Monday
        date_data = self._convert_column_to_date(date_col)
        # convert to a date, and subtract off the day of the week
        week_data = date_data - pd.to_timedelta(date_data.dt.dayofweek, unit='d')
        self.set_dimension_data_columns(week_col_name, week_data)
        if remove_date:
            self._remove_dimension(date_col)

    def split_date(self, date_col, remove_date: bool = True, perform_consolidation=True):
        # retrieves month and year from a date
        if self.has_dimension(date_col):
            self.convert_to_date(date_col)
            date_data = self.df[date_col]
            self.set_dimension_data_columns(monthCol, date_data.dt.month)
            self.set_dimension_data_columns(yearCol, date_data.dt.year)
            # remove the date column, if desired
            if remove_date:
                self.remove_data_column(date_col, perform_consolidation)

    def split_fiscal_period(self, fiscal_period_col: str = fiscalPeriodCol, perform_consolidation=True):
        # assumes the fiscal period is in YYYYMM format
        if self.has_dimension(fiscal_period_col):
            self.set_dimension_data_columns(yearCol,
                                            self.df[fiscal_period_col].astype(str).str.slice(stop=4).astype(int))
            self.set_dimension_data_columns(monthCol,
                                            self.df[fiscal_period_col].astype(str).str.slice(start=4).astype(int))
            self.remove_data_column(fiscal_period_col, perform_consolidation)

    def convert_ids_to_string(self, col_names: List[str], empty_value: str = ""):
        for n in col_names:
            self.convert_id_to_string(n, empty_value)

    def convert_common_ids_to_string(self):
        self.convert_ids_to_string([projectCol, employeeIdCol, taskCodeIdCol, foremanIdCol, branchCol])

    def convert_id_to_string(self, col_name: str, empty_value: str = ""):
        if self.has_dimension(col_name):
            idColData = self.df[col_name]
            emptyMask = idColData.isna() | idColData.isin([np.nan, np.inf, -np.inf, ""])
            # before setting the value, if is categorical and does NOT include the value, then do so here
            if self.is_categorical_column(idColData) and empty_value not in idColData.cat.categories:
                idColData = idColData.cat.add_categories(new_categories=empty_value)
                self._alphabetize_categories_for_column(col_name, idColData)
            # convert the numbers over to integers
            idColData = self.df[col_name]
            numeric_mask = idColData.transform(is_number)
            self.df.loc[numeric_mask, col_name] = idColData[numeric_mask].astype(int)
            # convert all values over to string
            self.df[col_name] = self.df[col_name].astype(str)
            # write the empty value at the mask positions
            self.df.loc[emptyMask, col_name] = empty_value

    def split_project_number(self, projectColName=projectCol):
        # breaks out the branch and profit center type from a project number
        # Add Branch (first digit of profit center) (keep as string)
        # first, create using the first digit as branch, and the second as profit center type
        default_project = "00000000"
        self.convert_id_to_string(projectColName, default_project)
        projectIdColData = self.df[projectColName]

        branchFromProject: Series = projectIdColData.str[0]  # Make Branch column the first digit of Project ID
        profitCenterFromProject: Series = projectIdColData.str[1]  # Make Profit Center column the second digit of Project

        # now, pull data for the second and third digit if the first digit is 7
        thirdDigitCol = projectIdColData.str[2]  # Use the third digit as Profit Center when the first digit is 7
        placesWhereFirstDigitIs7 = branchFromProject == '7'
        branchFromProject.mask(placesWhereFirstDigitIs7, profitCenterFromProject, inplace=True)
        profitCenterFromProject.mask(placesWhereFirstDigitIs7, thirdDigitCol, inplace=True)

        # Set 'Branch' and 'Profit Center' to zero if 'Project ID' contains letters
        branchFromProject.loc[branchFromProject.str.contains('[a-zA-Z]')] = '0'
        profitCenterFromProject.loc[profitCenterFromProject.str.contains('[a-zA-Z]')] = '0'
        profitCenterIdFromProject = branchFromProject + profitCenterFromProject

        # convert over to English
        branchFromProject.replace({"0": "Unknown", "1": "ATL", "2": "CLT", "4": "ATL"}, inplace=True)
        profitCenterFromProject.replace(profitCenterDesignationLookup, inplace=True)

        # only set the whole branch column if the branch doesn't exist yet
        # if the branch column does exist, then only set where the first digit of the project id is numeric
        can_set_from_project = projectIdColData.str.isnumeric() & (projectIdColData != default_project)
        if branchCol in self.df.columns:
            # since exists, then only write where the project id is a numeric value
            self.df.loc[can_set_from_project, branchCol] = branchFromProject.loc[can_set_from_project]
        else:
            # since does not exist yet, then create, and add as dimension
            self.set_dimension_data_columns(branchCol, branchFromProject)

        # similar for profit center type as with branch
        if profitCenterTypeCol in self.df.columns:
            # since exists, then only write where the project id is a numeric value
            self.df.loc[can_set_from_project, profitCenterTypeCol] = profitCenterFromProject.loc[can_set_from_project]
        else:
            # since does not exist yet, then create, and add as dimension
            self.set_dimension_data_columns(profitCenterTypeCol, profitCenterFromProject)

        # similar for profit center ID as with branch
        if profitCenterIdCol in self.df.columns:
            # since exists, then only write where the project id is a numeric value
            self.df.loc[can_set_from_project, profitCenterIdCol] = profitCenterIdFromProject.loc[can_set_from_project]
        else:
            # since does not exist yet, then create, and add as dimension
            self.set_dimension_data_columns(profitCenterIdCol, profitCenterIdFromProject)

    def split_task_code(self, perform_consolidation=True):
        # breaks out the craft and task code type
        taskCodeIdColData = self.df[taskCodeIdCol].astype(str)
        # Craft is the first digit of the task code
        self.set_dimension_data_columns(craftCol, taskCodeIdColData.str[0])
        # take the first 3 digits which will likely be shared across all data
        self.set_dimension_data_columns(taskCodeTypeIdCol, taskCodeIdColData.str[0:3])
        # now, remove the task code
        self.remove_data_column(taskCodeIdCol, perform_consolidation)

    @staticmethod
    def get_file_display_name(file_name: str, sheet_name: str | int | None):
        if file_name.endswith(".xlsx") and sheet_name is not None:
            return f"Sheet '{sheet_name}' of File '{file_name}'"
        else:
            return f"File '{file_name}'"

    def order_data_frame_columns_alphabetically(self):
        self.df = order_data_frame_columns_alphabetically(self.df)

    def save_data(self, file_name: str, sheet_name: str | int | None = None, excel_writer: pd.ExcelWriter | None = None, order_columns=True):
        # save off the file
        diagnostic_log_message(f"Saving {DataCollection.get_file_display_name(file_name, sheet_name)}")
        # order alphabetically to preserve order (to make for deterministic file)
        if order_columns:
            self.order_data_frame_columns_alphabetically()
        if file_name.endswith(".xlsx"):
            if sheet_name is None:
                sheet_name = "Sheet1"
            if isinstance(excel_writer, pd.ExcelWriter):
                self.df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
            elif path_exists(file_name):
                # since exists, then need to read the whole excel file to memory, then write
                with pd.ExcelWriter(file_name, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    # Now here add your new sheets
                    self.df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # since doesn't exist yet, then can write
                self.df.to_excel(file_name, sheet_name=sheet_name, index=False)
        elif file_name.endswith(".csv"):
            # if the data file is too large, then show progress bar while saving
            num_rows = len(self.df)
            if num_rows < 100_000:
                self.df.to_csv(file_name, index=False)
            else:
                # split into chunks
                num_sections = 100
                pb = ProgressBar(num_sections)
                chunks = np.array_split(self.df.index, num_sections)
                # save the first chunk
                self.df.loc[chunks[0]].to_csv(file_name, mode='w', index=False)
                pb.update()
                # save the remaining chunks
                for i in range(1, num_sections):
                    self.df.loc[chunks[i]].to_csv(file_name, header=None, mode='a', index=False)
                    pb.update()
                pb.finish()
        elif file_name.endswith(".hyper"):
            # set the sheet name as the name of the file
            # if a sheet name is not given, then use the name of the file
            table_name = get_base_file_name_without_ext(file_name) if sheet_name is None else sheet_name
            pantab.frame_to_hyper(self.df, file_name, table=table_name)
        else:
            raise ValueError("Unrecognized file type given")
        diagnostic_log_message(f"Finished saving {DataCollection.get_file_display_name(file_name, sheet_name)}")

    def auto_assign_measures(self, keep_dates=True):
        self.set_up_from_dimensions(commonDimensions, False)
        self.convert_common_ids_to_string()
        # create the common dimensions, but don't consolidate until later
        self.create_common_calculated_dimensions(not keep_dates, False)
        self.replace_dimension_nan()
        # after have done the replacements, then replace the data with the consolidated data
        self.consolidate_data_and_replace()
        self.convert_numeric_dimensions_to_str(commonNumericStringDimensions, False)

    def set_up_from_dimensions(self, dim_cols: List[str], warn_if_not_found: bool = True):
        # add the dimensions
        self.add_dimensions(dim_cols, warn_if_not_found)
        # add all other columns as measures
        for c in self.headers:
            if not self.has_dimension(c):
                self.add_measure(c)

    def replace_dimension_nan(self, numeric_placeholder=-1, other_placeholder=""):
        for d in self.dimension_names:
            dim_col: pd.Series = self.df[d]
            if is_numeric_dtype(dim_col):
                # replace with a negative 1
                dim_col.replace([np.inf, -np.inf, np.nan], numeric_placeholder, inplace=True)
            else:
                # replace with an empty string
                dim_col.replace([np.inf, -np.inf, np.nan], other_placeholder, inplace=True)

    def consolidate_data(self) -> DataFrame:
        if len(self.data_measures_lookup) > 0:
            # perform the group by and aggregation
            # figure out the aggregation methods
            measureAggr = {m.name: get_aggregation_function(m.aggregation_method, self.df[m.name].dtype)
                           for m in self.data_measures_lookup.values()}

            consolidated_data = group_data_frame(self.df, list(self.dimension_names), measureAggr)
        else:
            # since there are no data measure columns, then remove duplicates
            # remove any calculated fields, as these will be added back later
            consolidated_data = self.df.drop(columns=[k for k in self.calculated_columns_lookup.keys()]) \
                .drop_duplicates()

        # next, update all of the calculated fields
        self._recalculate_fields(consolidated_data)

        # return the output data frame
        return consolidated_data

    def round_columns(self, decimals: int, col_names: list[str]):
        for c in col_names:
            if c in self.df.columns:
                self.df[c] = self.df[c].round(decimals=decimals)

    def round_columns_containing(self, decimals: int, col_name_sub_str: str, exclude_sub_str: str = None):
        for c in self.df.columns:
            # find columns which contain the given substring, but don't include columns containing the exclusion sub string
            is_pertinent_column = col_name_sub_str in c
            if is_pertinent_column and exclude_sub_str is not None:
                is_pertinent_column = is_pertinent_column and exclude_sub_str not in c
            # if the column is pertinent, then round the column
            if is_pertinent_column:
                self.df[c] = self.df[c].round(decimals=decimals)

    @staticmethod
    def is_categorical_column(col):
        return isinstance(col.dtype, pd.CategoricalDtype)

    def _get_categorical_columns(self):
        # figure out which columns are set as categorical columns
        for colName in self.df.columns:
            col = self.df[colName]
            if self.is_categorical_column(col):
                yield colName, col

    def remove_unused_categories(self):
        # remove any unused categories from each of the categorical columns
        for (colName, col) in self._get_categorical_columns():
            self.df[colName] = col.cat.remove_unused_categories()

    def _alphabetize_categories_for_column(self, colName: str, col: pd.Series):
        currCategories: pd.Series = col.cat.categories
        self.df[colName] = col.cat.reorder_categories(new_categories=currCategories.sort_values(), ordered=True)

    def alphabetize_categories(self):
        # convert categories to categories ordered alphabetically
        for (colName, col) in self._get_categorical_columns():
            self._alphabetize_categories_for_column(colName, col)

    def _recalculate_fields(self, data_frame):
        # go through the calculated fields and determine which order will need to calculate
        calc_fields: List[DataColumnCalculated] = [c for c in self.calculated_columns_lookup.values()]
        calc_fields.sort(key=lambda c: c.max_calculation_depth)

        # lastly, create any of the calculated fields
        for c in calc_fields:
            c.calculate_and_set(data_frame)

    def cumsum(self, group_by_dims: List[str], order_by_dims: List[str],
               metrics_to_cumsum: List[str] | None, output_prefix: str | None = None, output_suffix: str | None = None):
        # calculates the cumulative sum, within each group, ordered by the dimensions listed to order
        # if the value is not given, then cumsum over all variables
        if metrics_to_cumsum is None:
            metrics_to_cumsum = [m for m in self.data_measures_lookup.keys()]

        # group by the given dimensions
        dimensions_to_keep = [d for d in group_by_dims + order_by_dims if d in self.dimension_names]

        # keep the relevant data columns, and perform the consolidation
        self.keep_data_columns(metrics_to_cumsum + dimensions_to_keep, True)

        # sort by the given dimensions
        if len(order_by_dims) > 0:
            self.df.sort_values(order_by_dims, ascending=True, inplace=True)

        # next, perform the cumsum
        grouped_rows = self.df.groupby(group_by_dims)
        grouped_values_to_cumsum = grouped_rows[metrics_to_cumsum]
        cumsum_result = grouped_values_to_cumsum.transform(Series.cumsum)
        output_columns = [add_prefix_suffix_to_string(m, output_prefix, output_suffix) for m in metrics_to_cumsum]
        self.set_measure_data_columns(output_columns, cumsum_result)

        # lastly, create any of the calculated fields
        self._recalculate_fields(self.df)

    def normalize(self, cols_to_normalize: List[str] | str, normalizing_col: str, remove_normalizing_col=True,
                  col_prefix="", col_suffix="", factor=1):
        # get the normalizing values
        # if wanting to apply a factor (like 100 to turn from 0-1 to 0-100), then divide here
        # this is the equivalent of multiplying the output by the factor
        normalizing_values = self.df[normalizing_col] / factor
        renamed_cols: Dict[str, str] = {}
        for c in convert_to_iterable(cols_to_normalize, str):
            self.df[c] /= normalizing_values
            renamed_cols[c] = col_prefix + c + col_suffix
        self.rename_columns(renamed_cols)
        if remove_normalizing_col:
            self.remove_data_column(normalizing_col, True)


class JoinHelperSide:
    # contains data for once side of the join
    def __init__(self, partition: DataCollection, suffix: str, source_name: str | None, source_indicator: bool, is_left: bool):
        # create a copy of the partition for joining
        self.partition = partition.copy()
        # set the suffix for any column renaming
        self.suffix = suffix
        # the name for the partition
        self.name: str | None = self.partition.name
        if source_name is not None:
            self.name = source_name
        if source_indicator:
            if is_left:
                source_col = DataCollection.left_source_col
                side_name = "Left"
            else:
                source_col = DataCollection.right_source_col
                side_name = "Right"
            assert self.name is not None, f"{side_name} source name not given or inferable"
            # create the source columns if doesn't exist
            if self.partition.has_measure(DataCollection.source_col_name):
                self.partition.rename_column(DataCollection.source_col_name, source_col)
            else:
                self.partition.set_measure_data_column(source_col, self.name, DataColumnAggregation.MIN)
        # the extra columns to be kept from this side
        self.extra_dimensions: Set[str] = set()
        # the data columns to keep for this side
        self.dimensions_to_keep: Set[str] = set()

    def rename_columns(self, cols_to_rename: Iterable[str]):
        # rename within the data frame and collection
        self.partition.rename_columns({m: m + self.suffix for m in cols_to_rename})
        # rename the extra dimensions and dimensions to keep
        for c in cols_to_rename:
            if c in self.extra_dimensions:
                self.extra_dimensions.remove(c)
                self.extra_dimensions.add(c + self.suffix)
            if c in self.dimensions_to_keep:
                self.dimensions_to_keep.remove(c)
                self.dimensions_to_keep.add(c + self.suffix)

    def set_shared_dimensions(self, new_shared_dimensions: Set[str]):
        # clear out the old values
        self.dimensions_to_keep.clear()
        # add in the shared dimensions
        self.dimensions_to_keep.update(new_shared_dimensions)
        # add in the extra dimensions
        self.dimensions_to_keep.update(self.extra_dimensions)
        # consolidate the dimensions down
        # if all dimensions are included within the new dimension set, then don't need to change anything
        if not self.partition.dimension_names.issubset(self.dimensions_to_keep):
            # otherwise, pare down the dimensions, and return the partition
            dimensionsToKeep = self.partition.dimension_names.intersection(self.dimensions_to_keep)
            self.partition.keep_dimensions(list(dimensionsToKeep))


class JoinHelper:
    # helps negotiate the join between two data collections
    # steps for a join:
    #   - Since all measures will be kept, rename any shared measures
    #       - If want all measures to be renamed, then should do before the join
    #   - For each of the join conditions:
    #       - Find the dimensions that are shared for the joining, and
    #         the dimensions that want to preserve (i.e., extra dimensions)
    #       - Rename any shared dimensions that are not going to be used for the joining
    #           - Note that this should only include extra dimensions, which at the time of writing
    #             were assumed to be constant throughout the join. Because of this,
    #             there should never be a time where an extra dimension is shared in a prior
    #             step (and thus renamed), but NOT shared in the next step.
    #             Because of this, don't need to have a special case for dimensions that were renamed
    #             in a prior step, and the renaming needs to persist.
    #             Instead, only rename the extra dimensions that are not shared.
    #             However, if a dimension is used as a shared dimension in a prior step,
    #             but is not used as a shared dimension in the current step, and is an
    #             extra dimension on both sides, then should be renamed from this point forward.
    #       - Perform the join
    def __init__(self, left_partition: DataCollection, right_partition: DataCollection,
                 left_suffix: str, right_suffix: str,
                 source_indicator: bool, left_source_name: str, right_source_name: str):
        diagnostic_log_message(f"Joining '{left_partition.name}' with '{right_partition.name}'")
        # create copies of the partitions for the joining
        self.left = JoinHelperSide(left_partition, left_suffix, left_source_name, source_indicator, True)
        self.right = JoinHelperSide(right_partition, right_suffix, right_source_name, source_indicator, False)

        # rename any measures which are shared
        shared_measures = self.left.partition.measure_names.intersection(self.right.partition.measure_names)
        if len(shared_measures) > 0:
            if len(left_suffix) == 0 and len(right_suffix) == 0:
                warn(f"The columns {shared_measures} are shared by both sides of the join. Consider renaming.")
            else:
                self.left.rename_columns(shared_measures)
                self.right.rename_columns(shared_measures)
        # set the output name
        self.output_name = None
        self.source_indicator = source_indicator
        if self.left.name is not None and self.right.name is not None:
            self.output_name = self.left.name + " & " + self.right.name

        # the columns that are shared by both data sources
        self.shared_dimensions: Set[str] = set()
        # keep track of the dimensions of the join result
        self.result_dimensions: Set[str] = set()

    def set_up_extra_dimensions(self, common_extra_dimensions: List[str] | str | None,
                                left_extra_dimensions: List[str] | str | None,
                                right_extra_dimensions: List[str] | str | None):
        # update the left and right with their values
        self.left.extra_dimensions.update(convert_to_iterable(left_extra_dimensions, str))
        self.right.extra_dimensions.update(convert_to_iterable(right_extra_dimensions, str))
        # update the left and right with the common values
        common_extra_dims_list = convert_to_iterable(common_extra_dimensions, str)
        self.left.extra_dimensions.update(common_extra_dims_list)
        self.right.extra_dimensions.update(common_extra_dims_list)

    def set_shared_dimensions(self, joining_dimensions: List[str] | str | None = None, ignore_dimensions: List[str] | str | None = None):
        # clear out the old values, and set the new values
        self.shared_dimensions.clear()
        # figure out the new dimensions to share
        sharedDimensions = self.left.partition.dimension_names.intersection(self.right.partition.dimension_names)
        # if the user wants only to look at specific dimensions, then whittle down the list
        if joining_dimensions is not None:
            specifiedDimensionsToJoin = set(convert_to_iterable(joining_dimensions, str))
            # require that all dimensions that the user specified are present
            if specifiedDimensionsToJoin.issubset(sharedDimensions):
                sharedDimensions = specifiedDimensionsToJoin
            else:
                missingDims = specifiedDimensionsToJoin - sharedDimensions
                raise ValueError(f"Dimensions {missingDims} are missing from both tables.")

        # if the user wants to ignore some dimensions, then remove
        for d in convert_to_iterable(ignore_dimensions, str):
            if d in sharedDimensions:
                sharedDimensions.remove(d)

        # now, update the values
        self.shared_dimensions.update(sharedDimensions)
        self.left.set_shared_dimensions(sharedDimensions)
        self.right.set_shared_dimensions(sharedDimensions)

        # Rename any dimensions that are kept on both sides but NOT shared.
        # Note that assumes that the extra dimensions never change,
        # and that the shared dimensions go from most restrictive to least restrictive.
        # Therefore, any overlapping dimensions that are unshared should be renamed
        # and will remain renamed
        overlapping_dims = self.left.dimensions_to_keep.intersection(self.right.dimensions_to_keep)
        overlapping_dims.difference_update(self.shared_dimensions)
        if len(overlapping_dims) > 0:
            self.left.rename_columns(overlapping_dims)
            self.right.rename_columns(overlapping_dims)

        # set the output dimensions as the kept dimensions from both sides
        self.result_dimensions.update(self.left.dimensions_to_keep)
        self.result_dimensions.update(self.right.dimensions_to_keep)

    def join_data_tables(self, join_type: JoinType, source_indicator: bool | None = None):
        # join the two data frames together on the shared dimensions
        # note that the two data frames should already be reduced down
        use_source_indicator = self.source_indicator if source_indicator is None else source_indicator
        joiningCols = list(self.shared_dimensions)
        resultingDf = self.left.partition.df.merge(self.right.partition.df, how=join_type.value, on=joiningCols,
                                                   indicator=use_source_indicator)

        # fill any created nans with 0
        for c in resultingDf.columns:
            colType = resultingDf.dtypes[c]
            if is_numeric_dtype(colType):
                resultingDf[c] = resultingDf[c].fillna(0)

        return resultingDf

    def create_output_data(self, joined_df: DataFrame):
        # set the partition, and then load in the dimensions and the non-calculated measures
        joinedPartition = DataCollection(joined_df)
        joinedPartition.name = self.output_name
        joinedPartition.add_dimensions(list(self.result_dimensions))
        joinedPartition.copy_measures_from_source(self.left.partition)
        joinedPartition.copy_measures_from_source(self.right.partition)

        # if a source indicator is desired, then rename the source types if needed
        if self.source_indicator:
            joinedPartition.rename_indicator_column()

        return joinedPartition


class DataCollectionFilter:
    def __init__(self, parent: DataCollection, column_name: str):
        self.parent = parent
        self.column_name = column_name
        self.column = parent.df[column_name]

    def is_greater_than(self, given_value):
        self.parent.filter_where(self.column > given_value)
        return self

    def is_greater_than_or_equal_to(self, given_value):
        self.parent.filter_where(self.column >= given_value)
        return self

    def is_less_than(self, given_value):
        self.parent.filter_where(self.column < given_value)
        return self

    def is_less_than_or_equal_to(self, given_value):
        self.parent.filter_where(self.column <= given_value)
        return self

    def is_between_exclusive(self, lower_bound, upper_bound):
        self.parent.filter_where(self.column.between(lower_bound, upper_bound, "neither"))
        return self

    def is_not_equal_to(self, value):
        self.parent.filter_where(self.column != value)
        return self

    def is_equal_to(self, value):
        self.parent.filter_where(self.column == value)
        return self

    def is_in(self, value_list: List):
        self.parent.filter_where(self.column.isin(value_list))


class DataCollectionPartitionSample(DataCollection):
    # a sampling of the data collection, with 1<=# files used <= total number files
    def __init__(self, df: pd.DataFrame,
                 dimensions_sharing: DataCollectionPartitionJoinType = DataCollectionPartitionJoinType.INTERSECT):
        super().__init__(df)
        self.dimensions_sharing = dimensions_sharing

        # determine the shared columns
        # note that when intersecting the dimensions, only keep the dimensions that are shared between all files
        # when unioning the dimensions, keep all dimensions, and instead do a full join
        if dimensions_sharing == DataCollectionPartitionJoinType.INTERSECT:
            # for all files, find the shared dimensions
            pass

    def _generate(self):
        # assumes that already have all data
        pass

    def generate_from_measures(self):
        # require there to be at least one measure column
        assert len(self.measures_lookup) > 0, "No measure columns provided"

        # now, go through all of the measure columns and determine what sources are used
        colListBySource: Dict[str, Tuple[DataTable, List[DataColumn]]] = {}

        # next, go through and determine which dimensions will be shared
        # note that when intersecting the dimensions, only keep the dimensions that are shared between all files
        # when unioning the dimensions, keep all dimensions, and instead do a full join
        dt: DataTable
        cl: List[DataColumn]
        usedDataTables = [dt for (dt, cl) in colListBySource.values()]
        # will be used for joining
        sharedDimensions = DimensionSet(DataCollectionPartitionJoinType.INTERSECT)
        # will be used for creating the dimensions on each table
        groupingDimensions = DimensionSet(self.dimensions_sharing)
        for dt in usedDataTables:
            sharedDimensions.add(dt.dimension_names)
            groupingDimensions.add(dt.dimension_names)
            sharedDimensions = sharedDimensions.intersection(dt.dimension_names)

        # next, build out a source collection for each data table
