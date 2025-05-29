# helpers for describing the set of criteria that must be met on a path for a decision tree
# for example, A BETWEEN 2 and 10 AND B <= 6 AND D IN (1,2,5)
from __future__ import annotations
from typing import Callable
from logical_conditions import *
from risk_data_column import *
from abc import ABC, abstractmethod
from overall_helpers import *


# contains a set of features, along with the criteria that must be met for each feature
# each node within a decision tree should contribute its criteria using this interface
class FeatureCriteriaSet:
    def __init__(self):
        self.feature_lookup: Dict[str, FeatureCriteriaBase] = {}
        self.base_value_contribution = 0

    def get_feature(self, feature: RiskDataColumn) -> FeatureCriteriaBase:
        if feature.name in self.feature_lookup:
            return self.feature_lookup[feature.name]
        else:
            new_feature = FeatureRangeCriteria(feature)
            self.feature_lookup[feature.name] = new_feature
            return new_feature

    def filter_with_threshold(self, feature: RiskDataColumn, threshold, is_less_than_or_equal_to: bool):
        # get the feature
        feature_criteria = self.get_feature(feature)
        # run the threshold, and change if needed
        if is_less_than_or_equal_to:
            new_feature_criteria = feature_criteria.keep_less_than_equal_to(threshold)
        else:
            new_feature_criteria = feature_criteria.remove_less_than_equal_to(threshold)
        if feature_criteria != new_feature_criteria:
            # if changed, then set
            self.feature_lookup[feature.name] = new_feature_criteria

    def filter_with_range(self, feature: RiskDataColumn, lower_bound, upper_bound, is_within: bool):
        # get the feature
        feature_criteria = self.get_feature(feature)
        # run the range criteria, and change if needed
        if is_within:
            new_feature_criteria = feature_criteria.keep_between(lower_bound, upper_bound)
        else:
            new_feature_criteria = feature_criteria.remove_between(lower_bound, upper_bound)
        if feature_criteria != new_feature_criteria:
            # if changed, then set
            self.feature_lookup[feature.name] = new_feature_criteria

    def filter_with_list(self, feature: RiskDataColumn, set_values: Iterable, is_included: bool):
        # get the feature
        feature_criteria = self.get_feature(feature)
        # run the inclusion, and change if needed
        if is_included:
            new_feature_criteria = feature_criteria.include_items(set_values)
        else:
            new_feature_criteria = feature_criteria.exclude_items(set_values)
        if feature_criteria != new_feature_criteria:
            # if changed, then set
            self.feature_lookup[feature.name] = new_feature_criteria

    def get_criteria(self, prettify=True) -> ConditionBase:
        # require all conditions to be met
        return BooleanCondition.get_condition_list([c.get_condition_representation(prettify) for c in self.feature_lookup.values()], True)

    def get_conditions_text(self, prettify=True):
        return self.get_criteria(prettify).get_text()

    @property
    def has_data(self):
        return len(self.feature_lookup) > 0


class ValueRange:
    # values (usually numbers, strings, or other comparable values)
    # defines inclusive or exclusive bounds for a single range of values
    def __init__(self, min_val=None, min_val_inclusive=False, max_val=None, max_val_inclusive=False):
        self.min_val = min_val
        self.min_val_inclusive = min_val_inclusive
        self.max_val = max_val
        self.max_val_inclusive = max_val_inclusive

    def constrain_to_range(self, constraining_range: ValueRange):
        self.constrain_minimum(constraining_range.min_val, constraining_range.min_val_inclusive)
        self.constrain_maximum(constraining_range.max_val, constraining_range.max_val_inclusive)
        return self.is_valid

    @property
    def is_valid(self):
        # will only be invalid if the minimum is greater than the maximum
        # thus, if doesn't have either a minimum or a maximum, then is valid
        if self.doesnt_have_min or self.doesnt_have_max:
            return True

        # set to whether the range is still applicable
        if self.min_val < self.max_val:
            # if the minimum is lower than the current value, then is valid
            return True
        elif self.min_val == self.max_val:
            # if the minimum and maximum range are the same, then
            return self.max_val_inclusive and self.min_val_inclusive

    @property
    def has_min(self):
        return self.min_val is not None

    @property
    def doesnt_have_min(self):
        return self.min_val is None

    @property
    def has_max(self):
        return self.max_val is not None

    @property
    def doesnt_have_max(self):
        return self.max_val is None

    def _change_minimum(self, new_minimum, new_minimum_inclusive):
        self.min_val = new_minimum
        self.min_val_inclusive = new_minimum_inclusive

    def _change_maximum(self, new_maximum, new_maximum_inclusive):
        self.max_val = new_maximum
        self.max_val_inclusive = new_maximum_inclusive

    def constrain_minimum(self, new_minimum, new_minimum_inclusive):
        # take the highest minimum value
        if new_minimum is None:
            return
        if self.doesnt_have_min or new_minimum > self.min_val:
            self._change_minimum(new_minimum, new_minimum_inclusive)
        elif new_minimum == self.min_val and not new_minimum_inclusive:
            # if starts at the same value, but is not inclusive, then set this value to not be inclusive
            self.min_val_inclusive = False

    def constrain_maximum(self, new_maximum, new_maximum_inclusive):
        # take the lower maximum value
        if new_maximum is None:
            return
        if self.doesnt_have_max or new_maximum < self.max_val:
            # if don't have an upper bound yet, of if the new maximum is less than the current value, then constrain
            self._change_maximum(new_maximum, new_maximum_inclusive)
        elif new_maximum == self.max_val and not new_maximum_inclusive:
            # if ends with the same value, but is not inclusive, then set to not be inclusive
            self.max_val_inclusive = False

    def includes_item(self, given_item):
        # make sure that the value is not higher than the max and not lower than the minimum
        return not self._value_falls_above_max(given_item) and not self._value_falls_below_min(given_item)

    def _value_falls_below_min(self, given_value):
        # checks if the given value is less than the minimum value of this range
        # if there is a minimum set, then check to see if it is beneath it
        if self.has_min:
            # if inclusive, then must be greater than or equal to the min to be included, so if less than, then is excluded
            # similarly, if not exclusive, then is excluded if less than or equal to min
            return given_value < self.min_val if self.min_val_inclusive else given_value <= self.min_val
        else:
            return False

    def _value_falls_above_max(self, given_value):
        # checks if the given value is higher than the max
        # if no max is specified, then means that extends to infinity
        # if there is a max set, then check to see if the value is above the max
        if self.has_max:
            # if inclusive, then must be less than or equal to the max to be included, so if greater than, then is excluded
            # similarly, if not exclusive, then is excluded if greater than or equal to max
            return given_value > self.max_val if self.max_val_inclusive else given_value >= self.max_val
        else:
            return False

    def modify_or_split_range(self, lower_bound, upper_bound):
        # removes any values (inclusive) of the lower bound and upper bound
        # 4 things can happen:
        # - the range exclusively covers the excluded range, in which case should be removed
        # - The range lowerbound is within the excluded range, which means that the minimum should be moved up
        # - The range upperbound is within the excluded range, which means that the maximum should be moved down
        # - The excluded range is contained within the range, which means that should split the range
        # Note that could either make the range invalid, modify this range, or split it
        # if splitting, then will return the split item as a new item to add to the list
        survives_below = not self._value_falls_below_min(lower_bound)
        survives_above = not self._value_falls_above_max(upper_bound)
        if survives_above and survives_below:
            # split into two ranges
            # create a new range for the part that survives above
            upper_range = ValueRange(upper_bound, False, self.max_val, self.max_val_inclusive)
            # modify this range for the lower range
            self._change_maximum(lower_bound, False)
            return upper_range

        # if doesn't survive above, then move the maximum to the lower bound (if applicable)
        if not survives_above:
            self.constrain_maximum(lower_bound, False)

        # if doesn't survive below, then move the minimum to the upper bound (if applicable)
        if not survives_below:
            self.constrain_minimum(upper_bound, False)
        return None


Tc = TypeVar("Tc")


# base for a feature criteria.
class FeatureCriteriaBase(ABC):
    def __init__(self, feature: RiskDataColumn):
        self.feature = feature

    # criteria about a collection of items
    # starts with the least restrictive set, and goes to most restrictive
    @abstractmethod
    # method for adding items to the criteria base
    def include_items(self, items_to_include: Iterable[Tc]) -> FeatureCriteriaBase:
        ...

    @abstractmethod
    # method for removing items from the criteria base
    def exclude_items(self, items_to_exclude: Iterable[Tc]) -> FeatureCriteriaBase:
        ...

    @abstractmethod
    # method for keeping values between two bounds
    def keep_between(self, lower_bound, upper_bound) -> FeatureCriteriaBase:
        ...

    @abstractmethod
    # method for removing values between two bounds
    def remove_between(self, lower_bound, upper_bound) -> FeatureCriteriaBase:
        ...

    @abstractmethod
    # method for keeping values at or below a certain threshold
    def keep_less_than_equal_to(self, threshold):
        ...

    @abstractmethod
    # method for removing values at or below a certain threshold
    def remove_less_than_equal_to(self, threshold):
        ...

    @abstractmethod
    def get_condition_representation(self, prettify=True) -> ConditionBase: ...


# most restrictive criteria, as says that must be within a given list.
class FeatureIncludeCriteria(FeatureCriteriaBase):
    # the include criteria is the most inclusive, and means that the the value follows from an IN statement
    # the value WILL only be one of a few values
    def __init__(self, feature: RiskDataColumn, includes: Iterable[Tc]):
        # don't need to start with the associated ranges, since are given the values included
        super().__init__(feature)
        self.includes = set(includes)

    def include_items(self, items_to_include: Iterable[Tc]) -> FeatureCriteriaBase:
        # only keep items that are within the original set and the new set
        self.includes.intersection_update(set(items_to_include))
        return self

    def _remove_items(self, items_to_remove: Iterable[Tc]) -> FeatureCriteriaBase:
        for item in items_to_remove:
            self.includes.discard(item)
        return self

    def _remove_items_meeting_criteria(self, criteria_func: Callable[[Tc], bool]):
        itemsToRemove = [item for item in self.includes if criteria_func(item)]
        return self._remove_items(itemsToRemove)

    def exclude_items(self, items_to_exclude: Iterable[Tc]) -> FeatureCriteriaBase:
        return self._remove_items(items_to_exclude)

    def keep_between(self, lower_bound, upper_bound) -> FeatureCriteriaBase:
        return self._remove_items_meeting_criteria(lambda item: item < lower_bound or item > upper_bound)

    def remove_between(self, lower_bound, upper_bound) -> FeatureCriteriaBase:
        return self._remove_items_meeting_criteria(lambda item: lower_bound <= item <= upper_bound)

    def keep_less_than_equal_to(self, threshold) -> FeatureCriteriaBase:
        return self._remove_items_meeting_criteria(lambda item: item > threshold)

    def remove_less_than_equal_to(self, threshold) -> FeatureCriteriaBase:
        return self._remove_items_meeting_criteria(lambda item: item <= threshold)

    def get_condition_representation(self, prettify=True) -> ConditionBase:
        # get the representation of the items
        # assumes that don't need to translate the value (since is likely not continuous, and therefore not rounded)
        list_texts = [self.feature.get_value_as_text(i, prettify) for i in self.includes]
        show_quotes = any([isinstance(i, str) for i in self.includes])
        return ListCondition.get_conditions_for_list(self.feature.name, list_texts, True, show_quotes)


# generic criteria, as says that either must be outside of a given list or outside of a given range
class FeatureRangeCriteria(FeatureCriteriaBase):
    # whether to allow excluded items to be relayed within the ranges
    exclude_items_from_ranges = False

    # class which can handle ranges and exclude requests
    def __init__(self, feature: RiskDataColumn):
        super(FeatureRangeCriteria, self).__init__(feature)
        self.ranges: ModifySafeList[ValueRange] = ModifySafeList()
        # whether or not can use ranges as a criteria
        self._uses_ranges = feature.can_range or feature.can_threshold
        if self._uses_ranges:
            base_range = ValueRange()
            if self.feature.has_min_value:
                base_range.constrain_minimum(self.feature.min_value, True)
            if self.feature.has_max_value:
                base_range.constrain_maximum(self.feature.max_value, True)
            self.ranges.append(base_range)
        self.excluded_items = set()

    def _includes_item(self, given_item):
        # check to see if it is in the excluded item set
        if given_item in self.excluded_items:
            return False
        # next, check to see if is included within any of the ranges
        if self._uses_ranges:
            # if including ranges, then ensure that is included within one of the ranges
            for r in self.ranges:
                if r.includes_item(given_item):
                    return True
            return False
        else:
            # if not including ranges, then probably means that just are checking the exclusion values
            # in this case, anything that is not excluded should be included
            return True

    def include_items(self, items_to_include: Iterable[Tc]) -> FeatureCriteriaBase:
        # since are only wanting to include specific items, then need to convert over to an Include Criteria
        # figure out which of the items should be included based on the given criteria, and include those
        return FeatureIncludeCriteria(self.feature, [i for i in items_to_include if self._includes_item(i)])

    def exclude_items(self, items_to_exclude: Iterable[Tc]) -> FeatureCriteriaBase:
        # mark the items to be excluded
        self.excluded_items.update(set(items_to_exclude))

        if FeatureRangeCriteria.exclude_items_from_ranges:
            # in addition, if using ranges, then break up any ranges that contain the items
            for item in items_to_exclude:
                for r in self.ranges:
                    if r.includes_item(item):
                        # the ranges includes the item, then modify or split the range
                        created_range = r.modify_or_split_range(item, item)
                        if created_range is not None:
                            self.ranges.insert_after_current(created_range)
                        if not r.is_valid:
                            self.ranges.remove_current()
                        # exit out of the ranges loop, since is guaranteed to only be in one range
                        break

        # finally, return this item
        return self

    def _modify_ranges(self, modify_func: Callable[[ValueRange], None]):
        # if uses ranges, then set up
        if self._uses_ranges:
            for r in self.ranges:
                # make the call to modify the range
                modify_func(r)
                # if is no longer valid, then remove from the list
                if not r.is_valid:
                    self.ranges.remove_current()
        return self

    def keep_between(self, lower_bound, upper_bound) -> FeatureCriteriaBase:
        # if uses ranges, then set up
        def constrain_ranges(r: ValueRange):
            r.constrain_minimum(lower_bound, True)
            r.constrain_maximum(upper_bound, True)

        return self._modify_ranges(constrain_ranges)

    def remove_between(self, lower_bound, upper_bound) -> FeatureCriteriaBase:
        if self._uses_ranges:
            for r in self.ranges:
                # the ranges includes the item, then modify or split the range
                created_range = r.modify_or_split_range(lower_bound, upper_bound)
                if created_range is not None:
                    self.ranges.insert_after_current(created_range)
                if not r.is_valid:
                    self.ranges.remove_current()

        # finally, return this item
        return self

    def keep_less_than_equal_to(self, threshold) -> FeatureCriteriaBase:
        # constrain to be less than or equal to (meaning including) the threshold
        return self._modify_ranges(lambda r: r.constrain_maximum(threshold, True))

    def remove_less_than_equal_to(self, threshold) -> FeatureCriteriaBase:
        # constrain to be greater than but not equal to (meaning not including) the threshold
        return self._modify_ranges(lambda r: r.constrain_minimum(threshold, False))

    def get_condition_representation(self, prettify=True) -> ConditionBase:
        # create a condition representation of the excluded items and the ranges
        # require that at least one of the ranges are met (i.e., OR)
        range_conditions = BooleanCondition.get_condition_list([self._convert_range_to_condition(r, prettify) for r in self.ranges], False)
        # require that both the exclusions and range criterias are met (i.e., AND)
        excluded_items_text = [self.feature.get_value_as_text(i, prettify) for i in self.excluded_items]
        return BooleanCondition.get_condition_list((ListCondition.get_conditions_for_list(self.feature.name, excluded_items_text, False),
                                                    range_conditions))

    def _convert_range_to_condition(self, given_range: ValueRange, prettify=True) -> ValueCondition | RangeCondition | EmptyCondition:
        # translate the values of the ranges back to the given values
        # if the range is not set, then return None, as don't need to include
        # check to see if the range is actually interesting
        # in order for the minimum value to be interesting, must actually have a minimum value,
        # and must be greater than (or equal to but non-inclusive) than the feature minimum value
        # if a feature minimum exists
        min_interesting = False
        if given_range.has_min:
            if self.feature.has_min_value:
                # must be greater than the feature minimum, or can be the same but non-inclusive
                min_interesting = given_range.min_val > self.feature.min_value or \
                                  (given_range.min_val == self.feature.min_value and not given_range.min_val_inclusive)
            else:
                # since has a minimum set, then is interesting
                min_interesting = True
        max_interesting = False
        if given_range.has_max:
            if self.feature.has_max_value:
                # must be less than the feature minimum, or can be the same but non-inclusive
                max_interesting = given_range.max_val < self.feature.max_value or \
                                  (given_range.max_val == self.feature.max_value and not given_range.max_val_inclusive)
            else:
                # since has a maximum set, then is interesting
                max_interesting = True

        # find the actual min / max value
        # note that the reported value could have been rounded, so un-round the value
        max_value_text = None
        wrap_max_in_quotes = False
        max_inclusive = False
        min_value_text = None
        wrap_min_in_quotes = False
        min_inclusive = None
        if max_interesting:
            if given_range.max_val_inclusive:
                # if is inclusive, then be less than or equal to as high as the feature can go
                actual_max_value, max_inclusive = self.feature.get_upper_bound(given_range.max_val)
            else:
                # means that goes up to the max value, but not including it
                actual_max_value, bound_is_inclusive = self.feature.get_lower_bound(given_range.max_val)
                max_inclusive = False
            max_value_text = self.feature.get_value_as_text(actual_max_value, prettify)
            wrap_max_in_quotes = not (isinstance(actual_max_value, float) or isinstance(actual_max_value, int))
        if min_interesting:
            if given_range.min_val_inclusive:
                # if is inclusive, then be greater than or equal to as low as the feature can go
                actual_min_value, bound_is_inclusive = self.feature.get_lower_bound(given_range.min_val)
                min_inclusive = given_range.min_val_inclusive
            else:
                # if is exclusive, then be greater than as high as the feature can go
                actual_min_value, bound_is_inclusive = self.feature.get_upper_bound(given_range.min_val)
                # only include the minimum if the bound is not inclusive
                min_inclusive = not bound_is_inclusive
            min_value_text = self.feature.get_value_as_text(actual_min_value, prettify)
            wrap_min_in_quotes = not (isinstance(actual_min_value, float) or isinstance(actual_min_value, int))

        # now, build the ranges
        if min_interesting and max_interesting:
            # since has both the upper and lower bound, then return a range
            return RangeCondition(self.feature.name, min_value_text, min_inclusive, max_value_text, max_inclusive,
                                  wrap_max_in_quotes or wrap_min_in_quotes)
        elif min_interesting:
            operator = ConditionOperations.GREATER_THAN_EQUAL if min_inclusive else ConditionOperations.GREATER_THAN
            return ValueCondition(self.feature.name, operator, min_value_text, wrap_min_in_quotes)
        elif max_interesting:
            operator = ConditionOperations.LESS_THAN_EQUAL if max_inclusive else ConditionOperations.LESS_THAN
            return ValueCondition(self.feature.name, operator, max_value_text, wrap_max_in_quotes)
        else:
            # since neither were interesting, return None
            return EmptyCondition()
