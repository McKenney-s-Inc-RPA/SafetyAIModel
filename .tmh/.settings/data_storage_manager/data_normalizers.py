from __future__ import annotations
import numpy as np
from json_manager import JsonBase
from numpy import floor as np_floor, round as np_round, ndarray, log10 as np_log10, float_power as np_float_power, \
    zeros as np_zeros, inf as np_inf, intc as np_int
from pandas import Series
from typing import List, Callable
from math import inf
from overall_helpers import can_be_int
from sortedcontainers import SortedDict

DEFAULT_MIN_VAL = -inf
DEFAULT_MAX_VAL = inf


def extract_exponent(given_value):
    return np_floor(np_log10(given_value))


class DataNormalizer(JsonBase):
    # by default, don't do anything, and just return the given value
    def normalize(self, given_value):
        # by default, return the value
        return given_value

    def get_lower_bound(self, normalized_value: float | ndarray | Series):
        return normalized_value, True

    # returns the upper bound, and whether or not it is inclusive
    def get_upper_bound(self, normalized_value):
        return normalized_value, True

    def explain_normalized_values(self, normalized_value: ndarray | Series):
        # explain the normalized value, like in a range or un-restricted value
        return normalized_value


class DataNormalizer_Bounded(DataNormalizer):
    def __init__(self, min_val: float = DEFAULT_MIN_VAL, max_val: float = DEFAULT_MAX_VAL):
        assert max_val > min_val, "Minimum value must be greater than the maximum value"
        self.min_val = min_val
        self.max_val = max_val
        if self.min_val > -inf:
            self._lower_than_min_val = self.min_val - 1
            self._higher_than_max_val = self.max_val + 1
        else:
            self._lower_than_min_val = self.min_val
            self._higher_than_max_val = self.max_val
        self._should_make_int = False
        self._includes_lower_bound = True
        self._includes_upper_bound = False

    def normalize(self, given_value: Series | ndarray):
        if isinstance(given_value, Series):
            normalized_values = self._normalize_vectorized(given_value.values)
            return Series(normalized_values, index=given_value.index)
        else:
            return self._normalize_vectorized(given_value)

    def _normalize_vectorized(self, given_value: ndarray):
        # create a zeros array
        output_value = np.zeros(given_value.shape)
        # set the values lower than the min value as the min value
        output_value[given_value < self.min_val] = self._lower_than_min_val
        # set the values higher than or equal to the max val as the max value
        output_value[given_value >= self.max_val] = self._higher_than_max_val
        # find the values within the bounds
        in_bounds_mask = (given_value >= self.min_val) & (given_value < self.max_val)
        output_value[in_bounds_mask] = self._normalize_inbound_values_vectorized(given_value[in_bounds_mask])
        if self._should_make_int:
            return output_value.astype(np_int)
        else:
            return output_value

    def _normalize_inbound_values_vectorized(self, given_value: ndarray):
        return given_value

    def get_lower_bound(self, normalized_value: float | ndarray | Series):
        if isinstance(normalized_value, Series):
            lower_bounds, is_inclusive = self._get_lower_bound_vectorized(normalized_value.values)
            return Series(lower_bounds, index=normalized_value.index), is_inclusive
        elif isinstance(normalized_value, ndarray):
            return self._get_lower_bound_vectorized(normalized_value)
        else:
            # assume that is a float or int
            if normalized_value == self._lower_than_min_val:
                return -inf, False
            elif normalized_value == self._higher_than_max_val:
                return self.max_val, self._includes_lower_bound
            else:
                # by default, assume that the vectorized solution will work for the float as well
                return self._get_inbound_lower_bound(normalized_value), self._includes_lower_bound

    def _get_lower_bound_vectorized(self, normalized_value: ndarray):
        # create a zeros array
        output_value = np.zeros(normalized_value.shape)
        # set the values lower than the minimum value as having no lower cutoff
        output_value[normalized_value == self._lower_than_min_val] = -inf
        # set the values higher than or equal to the max val as the max value
        output_value[normalized_value == self._higher_than_max_val] = self.max_val
        # find the values within the bounds
        in_bounds_mask = (normalized_value >= self.min_val) & (normalized_value < self.max_val)
        in_bounds_values = normalized_value[in_bounds_mask]
        lower_bound_values = self._get_inbound_lower_bound_vectorized(in_bounds_values)
        output_value[in_bounds_mask] = lower_bound_values
        return output_value, self._includes_lower_bound

    def _get_inbound_lower_bound_vectorized(self, normalized_value: ndarray | float):
        return normalized_value

    def _get_inbound_lower_bound(self, normalized_value: float):
        # by default, assume that the vectorized solution will work for the float as well
        return self._get_inbound_lower_bound_vectorized(normalized_value)

    def get_upper_bound(self, normalized_value: float | ndarray | Series):
        if isinstance(normalized_value, Series):
            upper_bounds, is_inclusive = self._get_upper_bound_vectorized(normalized_value.values)
            return Series(upper_bounds, index=normalized_value.index), is_inclusive
        elif isinstance(normalized_value, ndarray):
            return self._get_upper_bound_vectorized(normalized_value)
        else:
            # assume that is a float or int
            if normalized_value == self._lower_than_min_val:
                return self.min_val, self._includes_upper_bound
            elif normalized_value == self._higher_than_max_val:
                return inf, False
            else:
                # by default, assume that the vectorized solution will work for the float as well
                return self._get_inbound_upper_bound(normalized_value), self._includes_upper_bound

    def _get_upper_bound_vectorized(self, normalized_value: ndarray):
        # create a zeros array
        output_value = np.zeros(normalized_value.shape)
        # set the values lower than the minimum value as having the lower bound as the max value
        output_value[normalized_value == self._lower_than_min_val] = self.min_val
        # set the values higher than or equal to the max val as the max value
        output_value[normalized_value == self._higher_than_max_val] = inf
        # find the values within the bounds
        in_bounds_mask = (normalized_value >= self.min_val) & (normalized_value < self.max_val)
        output_value[in_bounds_mask] = self._get_inbound_upper_bound_vectorized(normalized_value[in_bounds_mask])
        return output_value, self._includes_upper_bound

    def _get_inbound_upper_bound_vectorized(self, normalized_value: ndarray | float):
        return normalized_value

    def _get_inbound_upper_bound(self, normalized_value: float):
        # by default, assume that the vectorized solution will work for the float as well
        return self._get_inbound_upper_bound_vectorized(normalized_value)


class DataNormalizer_RoundInteger(DataNormalizer_Bounded):
    def __init__(self, min_val: float = DEFAULT_MIN_VAL, max_val: float = DEFAULT_MAX_VAL):
        super().__init__(min_val, max_val)
        self._should_make_int = True

    # rounds a value to the nearest value
    def _normalize_inbound_values_vectorized(self, given_value: ndarray):
        return np_round(given_value, 0)

    def _get_inbound_lower_bound_vectorized(self, normalized_value: ndarray | float):
        return normalized_value

    def _get_inbound_upper_bound_vectorized(self, normalized_value: ndarray | float):
        return normalized_value


class DataNormalizer_RoundingBase(DataNormalizer_Bounded):
    def __init__(self, round_to, min_val: float = DEFAULT_MIN_VAL, max_val: float = DEFAULT_MAX_VAL):
        super().__init__(min_val, max_val)
        self.round_to = round_to
        self._should_make_int = can_be_int(round_to)


class DataNormalizer_Round(DataNormalizer_RoundingBase):
    # rounds a value to the nearest multiple
    def __init__(self, round_to, min_val: float = DEFAULT_MIN_VAL, max_val: float = DEFAULT_MAX_VAL):
        super().__init__(round_to, min_val, max_val)
        self.half_value = round_to / 2

    def _normalize_inbound_values_vectorized(self, given_value: ndarray):
        return np_round(given_value / self.round_to, 0) * self.round_to

    def _get_inbound_lower_bound_vectorized(self, normalized_value: ndarray | float):
        return normalized_value - self.half_value

    def _get_inbound_upper_bound_vectorized(self, normalized_value: ndarray | float):
        return normalized_value + self.half_value


class DataNormalizer_RoundDown(DataNormalizer_RoundingBase):
    # rounds a value to the nearest multiple
    def __init__(self, round_to, min_val: float = DEFAULT_MIN_VAL, max_val: float = DEFAULT_MAX_VAL):
        super().__init__(round_to, min_val, max_val)
        # figure out the decimals to round the result to
        # if rounding to 0.25, then the log is between 0 and -1. Floor to -1.
        # Take the negative, so are keeping 1 decimal place.
        # Add 2 decimal places, so don't cut off anything needed.
        self._decimals_to_round_to = int(-extract_exponent(round_to)) + 2

    def _normalize_inbound_values_vectorized(self, given_value: ndarray):
        return np_round(np_floor(given_value / self.round_to) * self.round_to, self._decimals_to_round_to)

    def _get_inbound_lower_bound_vectorized(self, normalized_value: ndarray | float):
        return np_round(normalized_value, self._decimals_to_round_to)

    def _get_inbound_upper_bound_vectorized(self, normalized_value: ndarray | float):
        return np_round(normalized_value + self.round_to, self._decimals_to_round_to)


class Incrementing_Steps_Rule:
    def __init__(self, min_inclusive, max_exclusive, step_size):
        self.min_inclusive = min_inclusive
        self.max_exclusive = max_exclusive
        self.step_size = step_size


class DataNormalizer_Incrementing_Steps(DataNormalizer_Bounded):
    # rounds a value based on set "increment" steps
    # for example, round to 5 if > 100, round to 2 if > 10, round to 1 if > 1
    # This assumes that all increment change values are equally spaced apart
    #   In example above, this means that 10 to 100 are equal steps of 2, and 1 to 10 are equal steps of 1.
    def __init__(self, incremental_steps: dict[float | int, float | int], min_val: float = DEFAULT_MIN_VAL,
                 max_val: float = DEFAULT_MAX_VAL):
        super().__init__(min_val, max_val)
        # require all increments to be greater than 0
        assert all([v > 0 for v in incremental_steps.values()]), "All increments must be greater than 0."
        # convert the keys to floats / ints if stored as strings (since may be stored in JSON as strings)
        # note that because is rounding, no user should input string data (unless it is numeric), so if it errors, its on them...
        str_keys = [k for k in incremental_steps.keys() if isinstance(k, str)]
        for k in str_keys:
            float_key = float(k)
            if can_be_int(float_key):
                float_key = int(float_key)
            incremental_steps[float_key] = incremental_steps[k]
            del incremental_steps[k]
        # require that the minimum be in the list
        assert min_val in incremental_steps, "Minimum value must be included within the mappings"
        # if all of the keys and all fo the values are integers, then can set to round to integers
        self._should_make_int = all([can_be_int(k) and can_be_int(v) for k, v in incremental_steps.items()])
        # set up the sorted list, so that can identify the increment less than the current value
        self.incremental_steps = SortedDict(incremental_steps)
        # set up the values for each index
        self._increment_values_by_index = [v for v in self.incremental_steps.values()]
        # also, set up the bounding rules for each of the increments
        self._incrementing_steps: list[Incrementing_Steps_Rule] = []
        # get the list of the increments from the min to the max - don't need to include any values above the maximum value
        ordered_increments = [(key, incremental_steps[key]) for key in sorted(incremental_steps.keys()) if key < max_val]
        # add the max value (step doesn't matter)
        ordered_increments.append((max_val, 0))
        # go through each of the increments (excluding the last item since isn't real increment), and add a step
        for i in range(len(ordered_increments) - 1):
            curr_increment = ordered_increments[i]
            next_increment = ordered_increments[i + 1]
            self._incrementing_steps.append(Incrementing_Steps_Rule(curr_increment[0], next_increment[0], curr_increment[1]))

    def _find_increment(self, given_value: float | ndarray):
        if isinstance(given_value, ndarray):
            # create an output array
            output_value = np.zeros(given_value.shape)
            # for each of the steps, find the indices of the input array within the bounds,
            # and set the given value to the given value
            for inc in self._incrementing_steps:
                output_value[(given_value >= inc.min_inclusive) & (given_value < inc.max_exclusive)] = inc.step_size
            return output_value
        else:
            # assume that the given value is greater than or equal to the minimum
            # find the index that is less than or equal to the given value
            # NOTE: since the minimum value is set & checked, there should ALWAYS be an item less than or equal to the given value
            # therefore, for any value, bisect_right will give the next index for the given value
            # for example, bisect_right will give 1 for all values >= min, but less than the next value
            # If you subtract 1 from this value, you should get the index to use for the floor function
            pos = self.incremental_steps.bisect_right(given_value) - 1
            return self._increment_values_by_index[pos]

    def _normalize_inbound_values_vectorized(self, given_value: float):
        # the values coming in here should be between the min (inclusive) and max (exclusive)
        # find the value lower than or equal to the current value to round to
        round_increment = self._find_increment(given_value)
        # round to 3 decimal places (which should be close enough for rounding errors)
        return np_round(np_floor(given_value / round_increment) * round_increment, 3)

    def _get_inbound_lower_bound_vectorized(self, normalized_value: float | ndarray):
        # the lowerbound should be the normalized value, since is essentially flooring the value
        return normalized_value

    def _get_inbound_upper_bound_vectorized(self, normalized_value: float | ndarray):
        # the upperbound should be the normalized value, plus the increment
        return normalized_value + self._find_increment(normalized_value)


class DataNormalizer_LargeNumber(DataNormalizer_Bounded):
    # rounds a value down to the nearest value of a given large number
    # The rounding values get exponentially higher
    # Given a list of mantissas (i.e., numeric non-exponential part of the scientific notation)
    # For example, if given [1, 2.5, 5], then will round to:
    #   10^0: 1, 2.5, 5,
    #   10^1: 10, 25, 50,
    #   10^2: 100, 250, 500

    def __init__(self, mantissas: List[float], min_val: float = 1, max_val: float = 1e9):
        # make sure the first mantissa option is 1, so that after the exponent removal, will always have a lower bound value
        assert any([m == 1 for m in mantissas]), "Must include at least one mantissa equal to 1"
        # make sure that all mantissas are between 1 and 10.
        # If less than 1 (like 0.9), then should use the next lowest exponent (i.e., 9 * 10^-1)
        # If greater than 10 (like 11), then should use the next highest exponent (i.e., 1.1 * 10^1)
        assert all([1 <= m < 10 for m in mantissas]), "All mantissas should be in greater than or equal to 1, and less than 10"
        # sort form smallest to largest
        self.mantissas = sorted(mantissas)
        # Ensure that proper bounds have been given
        # note that assumes that are large positive numbers, so don't allow the minimum to go too low
        assert min_val >= 1, "Must have a minimum value larger than 1"
        assert max_val <= 1e20, "Must have a valid upperbound given"
        super().__init__(min_val, max_val)
        # normalize the minimum and maximum values to provide an easier time reversing the normalizer
        self.min_val = self._normalize_regular_non_bounded(min_val)
        self.max_val = int(self._normalize_regular_non_bounded(max_val))
        self._lower_than_min_val = 0
        self._higher_than_max_val = self.max_val
        # figure out if can round
        min_value_exp = int(extract_exponent(self.min_val))
        max_value_exp = int(extract_exponent(self.max_val))
        possible_values = [m * 10 ** x
                           for m in self.mantissas
                           for x in range(min_value_exp, max_value_exp + 1)
                           if self.min_val <= m * 10 ** x <= self.max_val]
        self._should_make_int = all([can_be_int(p) for p in possible_values])
        if self._should_make_int:
            self.min_val = int(self.min_val)
            self.max_val = int(self.max_val)

    def _normalize_inbound_values_vectorized(self, given_value: ndarray):
        # break into the mantissa and exponent component
        mantissa, exponent_component = self._break_into_mantissa_and_exponent(given_value)
        rounded_down_mantissa = np_zeros(mantissa.shape)
        # for each of the mantissas, find all values greater than the value
        for m in self.mantissas:
            rounded_down_mantissa[mantissa >= m] = m
        # finally, build the values
        return rounded_down_mantissa * exponent_component

    def _normalize_regular_non_bounded(self, given_value: float):
        if given_value <= 0:
            return 0
        else:
            # find the exponent and the mantissa
            mantissa, exponent_component = self._break_into_mantissa_and_exponent(given_value)
            # find the mantissa lower than this value
            mantissa_lower_than = max([m for m in self.mantissas if m <= mantissa])
            # build the full value
            return mantissa_lower_than * exponent_component

    @staticmethod
    def _break_into_mantissa_and_exponent(given_value: float | ndarray | Series):
        exponent = extract_exponent(given_value)
        exponent_component = np_float_power(10, exponent)
        mantissa = given_value / exponent_component
        return mantissa, exponent_component

    def _get_inbound_lower_bound_vectorized(self, normalized_value: ndarray | float):
        # the value is rounded down, so should already be the lower bound (inclusive)
        # note that if is smaller than minimum, then should have been sent to 0
        return normalized_value

    def _get_inbound_upper_bound_vectorized(self, normalized_value: ndarray | float):
        # find the next largest value
        # first, break apart the given value
        mantissa, exponent_component = self._break_into_mantissa_and_exponent(normalized_value)
        # find the mantissa option index
        allOptions = np.array(self.mantissas)
        sortedOptionIndices = np.argsort(allOptions)
        searchResults = np.searchsorted(allOptions[sortedOptionIndices], mantissa)
        mantissa_index = sortedOptionIndices[searchResults]
        # find the next index
        next_index = (mantissa_index + 1) % len(self.mantissas)
        # for any items where the next index is zero, multiply the exponents by 10
        # means that wrapped around to the next exponent, so add one to the exponent
        exponent_component[next_index == 0] *= 10
        # calculate the resulting mantissa
        new_mantissas = allOptions[next_index]
        return new_mantissas * exponent_component

    def _get_inbound_upper_bound(self, normalized_value: float):
        # find the next largest value
        # first, break apart the given value
        mantissa, exponent_component = self._break_into_mantissa_and_exponent(normalized_value)
        # find the mantissa lower than this value
        mantissa_index = self.mantissas.index(mantissa)
        # find the next index
        next_index = (mantissa_index + 1) % len(self.mantissas)
        if next_index == 0:
            # means that wrapped around to the next exponent, so add one to the exponent
            exponent_component *= 10
        return self.mantissas[next_index] * exponent_component
