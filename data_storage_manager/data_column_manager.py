from __future__ import annotations
from data_columns import *
from typing import Set, Dict
from overall_helpers import rename_key_in_dictionary, rename_key_in_set


class DataColumnManager:
    # a column manager will have the following types of columns
    #   - Dimension columns (potentially shared across multiple data sources)
    #   - Measure columns (data columns
    #       - Data Columns: Come from one data file
    #       - Calculated Columns: Use other measure columns to calculate the values
    def __init__(self):
        self.dimension_names: Set[str] = set()
        self.measure_names: Set[str] = set()
        self.dimension_lookup: Dict[str, DataColumnDimension] = {}
        self.measures_lookup: Dict[str, DataColumnMeasureBase] = {}
        self.data_measures_lookup: Dict[str, DataColumn] = {}
        self.calculated_columns_lookup: Dict[str, DataColumnCalculated] = {}
        self.dummy_data_name_lookup: Dict[str, DataColumn] = {}
        self.column_lookup: Dict[str, DataColumnBase] = {}

    def _clear(self):
        self.dimension_names.clear()
        self.measure_names.clear()
        self.dimension_lookup.clear()
        self.measures_lookup.clear()
        self.data_measures_lookup.clear()
        self.calculated_columns_lookup.clear()
        self.dummy_data_name_lookup.clear()
        self.column_lookup.clear()

    def __str__(self):
        return f"Dimensions: {self.dimension_names}\nMeasures: {self.measure_names}"

    def _add_column(self, new_col: DataColumnBase):
        self.column_lookup[new_col.name] = new_col

    def _add_dimension_column(self, new_dim: DataColumnDimension):
        # unignore the column, and remove as measure
        self._remove_measure(new_dim.name)

        # add as the dimension
        self._add_column(new_dim)
        self.dimension_lookup[new_dim.name] = new_dim
        self.dimension_names.add(new_dim.name)

    def _add_measure_column(self, new_meas: DataColumnMeasureBase):
        # unignore the column, and remove as dimension
        self._remove_dimension(new_meas.name)

        # add as the measure
        self._add_column(new_meas)
        self.measure_names.add(new_meas.name)
        self.measures_lookup[new_meas.name] = new_meas

        # add to the appropriate lookup
        # also, remove from any other lookups
        if isinstance(new_meas, DataColumn):
            self.data_measures_lookup[new_meas.name] = new_meas
            self.calculated_columns_lookup.pop(new_meas.name, None)
        elif isinstance(new_meas, DataColumnCalculated):
            self.calculated_columns_lookup[new_meas.name] = new_meas
            self.data_measures_lookup.pop(new_meas.name, None)
        else:
            raise NotImplementedError("Unknown measure type given")

    def _rename_column(self, old_name, new_name):
        # rename the column
        if old_name in self.column_lookup:
            self.column_lookup[old_name].name = new_name
        # replace all references within the lookups / sets
        rename_key_in_set(self.dimension_names, old_name, new_name)
        rename_key_in_set(self.measure_names, old_name, new_name)
        rename_key_in_dictionary(self.dimension_lookup, old_name, new_name)
        rename_key_in_dictionary(self.measures_lookup, old_name, new_name)
        rename_key_in_dictionary(self.data_measures_lookup, old_name, new_name)
        rename_key_in_dictionary(self.calculated_columns_lookup, old_name, new_name)
        rename_key_in_dictionary(self.column_lookup, old_name, new_name)

    def _add_dimension(self, new_dimension: str) -> DataColumnDimension:
        if self.has_dimension(new_dimension):
            return self.dimension_lookup[new_dimension]

        # add to the lookup
        dimCol = DataColumnDimension(new_dimension)
        self._add_dimension_column(dimCol)
        return dimCol

    def _remove_dimension(self, dimension_to_remove: str):
        if self.has_dimension(dimension_to_remove):
            self.dimension_names.remove(dimension_to_remove)
            self.dimension_lookup.pop(dimension_to_remove, None)
            self.column_lookup.pop(dimension_to_remove, None)

    def _remove_measure(self, measure_to_remove: str):
        if self.has_measure(measure_to_remove):
            self.measure_names.remove(measure_to_remove)
            self.measures_lookup.pop(measure_to_remove, None)
            self.column_lookup.pop(measure_to_remove, None)
            self.data_measures_lookup.pop(measure_to_remove, None)
            self.calculated_columns_lookup.pop(measure_to_remove, None)

    def has_dimension(self, dim_name: str):
        return dim_name in self.dimension_names

    def has_measure(self, meas_name: str):
        return meas_name in self.measure_names

    def __getitem__(self, key: str) -> DataColumnBase:
        # gets a column from the columns
        return self.column_lookup[key]

    def _copy_from(self, source_data: DataColumnManager):
        for dim_name in source_data.dimension_names:
            self._add_dimension(dim_name)
        for meas_col in source_data.measures_lookup.values():
            self._add_measure_column(meas_col.copy())
