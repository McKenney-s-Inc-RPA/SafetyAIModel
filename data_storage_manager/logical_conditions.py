from __future__ import annotations
from typing import Iterable
from enum import Enum
from abc import ABC, abstractmethod
from overall_helpers import *
from render_helpers import get_short_text_for_large_number
from pandas import DataFrame, Series
from numpy import logical_or as np_logical_or, logical_and as np_logical_and


# set of conditions for a given rule


class ConditionOperations(Enum):
    # these operators are checking if an item is inside / outside of a set
    INCLUDED_IN = "IN"
    NOT_IN = "NOT IN"
    # these comparisons are in regards to a threshold
    EQUALS = "="
    NOT_EQUALS = "!="
    LESS_THAN = "<"
    LESS_THAN_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_THAN_EQUAL = ">="
    # these comparisons are between two bounds
    # note that a "not between" operator has not been implemented, so all range operators will be between operators
    BETWEEN = "BETWEEN"
    # these are for checking boolean set operations
    OR = "OR"
    AND = "AND"
    NO_CONDITION = "NONE"


class ConditionBase(ABC):
    # stores a condition for a given property name
    # conditions are simple boolean expressions (like 'Col' < 5.4)
    def __init__(self, operator: ConditionOperations):
        self.operator = operator

    @property
    def _operator_text(self):
        return self.operator.value

    @property
    def is_condition_list(self):
        return self.operator in (ConditionOperations.AND, ConditionOperations.OR)

    @property
    def is_value_operator(self):
        return self.operator in (ConditionOperations.EQUALS, ConditionOperations.NOT_EQUALS, ConditionOperations.LESS_THAN_EQUAL,
                                 ConditionOperations.LESS_THAN, ConditionOperations.GREATER_THAN, ConditionOperations.GREATER_THAN_EQUAL)

    @property
    def is_list_operator(self):
        return self.operator in (ConditionOperations.INCLUDED_IN, ConditionOperations.NOT_IN)

    @property
    def is_range_operator(self):
        return self.operator == ConditionOperations.BETWEEN

    @property
    def is_empty(self):
        return self.operator == ConditionOperations.NO_CONDITION

    @abstractmethod
    def get_text(self):
        ...

    def get_columns(self) -> set[str]:
        return set()

    def get_dataset_mask(self, dataset: DataFrame) -> Series:
        return Series(data=False, index=dataset.index, dtype=bool)


class EmptyCondition(ConditionBase):
    def __init__(self):
        super().__init__(ConditionOperations.NO_CONDITION)

    def get_text(self):
        return ""


class ValueConditionBase(ConditionBase, ABC):
    def __init__(self, name: str, operator: ConditionOperations, wrap_value_in_quotes_if_str=True):
        super().__init__(operator)
        self.name = name
        self.wrap_value_in_quotes_if_str = wrap_value_in_quotes_if_str

    def _convert_to_text(self, given_item):
        # converts a string, numeric, or boolean value to text
        if isinstance(given_item, str):
            if self.wrap_value_in_quotes_if_str:
                return "'" + given_item + "'"
            else:
                return given_item
        elif isinstance(given_item, float) or isinstance(given_item, int):
            return str(get_short_text_for_large_number(given_item))
        else:
            return str(given_item)

    def get_columns(self) -> set[str]:
        name_set = set()
        name_set.add(self.name)
        return name_set


class ValueCondition(ValueConditionBase):
    # compares a value against a given condition
    def __init__(self, name: str, operator: ConditionOperations, comparison_value, wrap_value_in_quotes_if_str=True):
        super().__init__(name, operator, wrap_value_in_quotes_if_str)
        self.comparison_value = comparison_value

    def get_text(self):
        return f"'{self.name}' {self._operator_text} {self._convert_to_text(self.comparison_value)}"

    def get_dataset_mask(self, dataset: DataFrame):
        data_col = dataset[self.name]
        match self.operator:
            case ConditionOperations.LESS_THAN_EQUAL:
                return data_col <= self.comparison_value
            case ConditionOperations.LESS_THAN:
                return data_col < self.comparison_value
            case ConditionOperations.GREATER_THAN_EQUAL:
                return data_col >= self.comparison_value
            case ConditionOperations.GREATER_THAN:
                return data_col > self.comparison_value
            case ConditionOperations.EQUALS:
                return data_col == self.comparison_value
            case ConditionOperations.NOT_EQUALS:
                return data_col != self.comparison_value
            case _:
                return super().get_dataset_mask(dataset)


class ListCondition(ValueConditionBase):
    @staticmethod
    def get_conditions_for_list(name: str, items: Iterable, includes_items: bool, wrap_value_in_quotes_if_str=True):
        # check the length of the given sequence
        full_list = list(items)
        num_items = len(full_list)
        # if only one item, then convert over to an equals / not equals
        if num_items == 1:
            operator = ConditionOperations.EQUALS if includes_items else ConditionOperations.NOT_EQUALS
            return ValueCondition(name, operator, full_list[0], wrap_value_in_quotes_if_str)
        elif num_items > 1:
            operator = ConditionOperations.INCLUDED_IN if includes_items else ConditionOperations.NOT_IN
            return ListCondition(name, operator, full_list, wrap_value_in_quotes_if_str)
        else:
            return EmptyCondition

    # checks whether a value is in / out of a list
    def __init__(self, name: str, operator: ConditionOperations, full_list: List, wrap_value_in_quotes_if_str=True):
        super().__init__(name, operator, wrap_value_in_quotes_if_str)
        self.full_list = full_list

    def get_text(self):
        optionList = ", ".join([self._convert_to_text(c) for c in self.full_list])
        return f"'{self.name}' {self._operator_text} [{optionList}]"

    def get_dataset_mask(self, dataset: DataFrame) -> Series:
        data_col: Series = dataset[self.name]
        match self.operator:
            case ConditionOperations.INCLUDED_IN:
                return data_col.isin(self.full_list)
            case ConditionOperations.NOT_IN:
                return ~data_col.isin(self.full_list)
            case _:
                return super().get_dataset_mask(dataset)


class RangeCondition(ValueConditionBase):
    # checks whether a column is within / not within a range
    def __init__(self, name: str, lower_bound, lower_is_inclusive: bool, upper_bound, upper_is_inclusive: bool, wrap_value_in_quotes_if_str=True):
        super().__init__(name, ConditionOperations.BETWEEN, wrap_value_in_quotes_if_str)
        self.lower_bound = lower_bound
        self.lower_is_inclusive = lower_is_inclusive
        self.upper_bound = upper_bound
        self.upper_is_inclusive = upper_is_inclusive

    def get_text(self):
        lower_bound_text = self._convert_to_text(self.lower_bound)
        upper_bound_text = self._convert_to_text(self.upper_bound)
        if self.lower_is_inclusive and self.upper_is_inclusive:
            return f"{lower_bound_text} <= '{self.name}' <= {upper_bound_text}"
        elif self.lower_is_inclusive:
            return f"{lower_bound_text} <= '{self.name}' < {upper_bound_text}"
        elif self.upper_is_inclusive:
            return f"{lower_bound_text} < '{self.name}' <= {upper_bound_text}"
        else:
            return f"{lower_bound_text} < '{self.name}' < {upper_bound_text}"

    def get_dataset_mask(self, dataset: DataFrame) -> Series:
        data_col: Series = dataset[self.name]
        lower_bound_adhesion: Series = data_col >= self.lower_bound if self.lower_is_inclusive else data_col > self.lower_bound
        upper_bound_adhesion: Series = data_col <= self.upper_bound if self.upper_is_inclusive else data_col < self.upper_bound
        return lower_bound_adhesion & upper_bound_adhesion


class BooleanCondition(ConditionBase):
    @staticmethod
    def get_condition_list(conditions: Iterable[ConditionBase], all_required=True) -> ConditionBase:
        condition_list = [c for c in conditions if c is not None and not c.is_empty]
        if len(condition_list) == 0:
            return EmptyCondition()
        elif len(condition_list) == 1:
            return condition_list[0]
        else:
            return BooleanCondition(condition_list, all_required)

    # contains a list of conditions to be met
    # can be combined with an "and" or "or"
    def __init__(self, conditions: List[ConditionBase], all_required=True):
        super().__init__(ConditionOperations.AND if all_required else ConditionOperations.OR)
        self.conditions = conditions

    def get_text(self):
        joining_text = " " + self._operator_text + " "
        return joining_text.join(["(" + c.get_text() + ")" for c in self.conditions])

    def get_dataset_mask(self, dataset: DataFrame) -> Series:
        if len(self.conditions) == 2:
            sub_result_1 = self.conditions[0].get_dataset_mask(dataset)
            sub_result_2 = self.conditions[1].get_dataset_mask(dataset)
            match self.operator:
                case ConditionOperations.AND:
                    return sub_result_1 & sub_result_2
                case ConditionOperations.OR:
                    return sub_result_1 | sub_result_2
                case _:
                    return super().get_dataset_mask(dataset)
        else:
            sub_results = [c.get_dataset_mask(dataset) for c in self.conditions]
            sub_result_frame = DataFrame(data=sub_results, index=dataset.index)
            match self.operator:
                case ConditionOperations.AND:
                    return sub_result_frame.all(axis="columns")
                case ConditionOperations.OR:
                    return sub_result_frame.any(axis="columns")
                case _:
                    return super().get_dataset_mask(dataset)

    def get_columns(self) -> set[str]:
        names_used = set()
        for c in self.conditions:
            names_used.update(c.get_columns())
        return names_used
