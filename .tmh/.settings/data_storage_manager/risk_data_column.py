from data_normalizers import *
from json_manager import JsonBase
from pandas import Series
from pandas.api.types import is_numeric_dtype
from render_helpers import get_short_text_for_large_number, remove_decimal_if_needed
from typing import Any


class RiskDataColumn(JsonBase):
    # relates to a column of a data set
    # contains a normalizing function, which can be called on the source data to normalize it
    # also contains an explainer function, which takes a normalized value and explains it
    # for example, the normalizer function may round a value down to the nearest 5,
    # For an input of 7, this would return 5.
    # Given the normalized value of 5, the explainer function would say that it has a value between 5 (inclusive) and 10 (exclusive)
    def __init__(self, name: str, normalizer: DataNormalizer, is_continuous: bool, min_value=None, max_value=None, can_threshold=True,
                 can_range=False, can_categorize=False, units: str | None = None):
        self.name = name
        self.normalizer = normalizer
        self.is_continuous = is_continuous
        # the minimum value and max value are the absolute bounds for the reporting ranges
        # note that the normalizer may also have its own min / max
        self.min_value = min_value
        self.max_value = max_value
        self.can_threshold = can_threshold
        self.can_range = can_range
        self.can_categorize = can_categorize
        self.units = units

    def __str__(self):
        return f"Column '{self.name}'"

    def normalize(self, given_value):
        return self.normalizer.normalize(given_value)

    def get_lower_bound(self, given_value):
        return self.normalizer.get_lower_bound(given_value)

    def get_upper_bound(self, given_value):
        return self.normalizer.get_upper_bound(given_value)

    @property
    def has_min_value(self):
        return self.min_value is not None

    @property
    def has_max_value(self):
        return self.max_value is not None

    def get_bounds_text(self, col_val: Series, show_values_as_range=False):
        # save off the normalized data
        is_number_column = is_numeric_dtype(col_val)
        can_be_normalized = self.is_continuous and is_number_column
        if can_be_normalized:
            # show the upper and lower value
            normalized_vals = self.normalize(col_val)
            lower_bound, lower_inclusive = self.get_lower_bound(normalized_vals)
            upper_bound, upper_inclusive = self.get_upper_bound(normalized_vals)
            # create the strings, removing any trailing decimal point if is present
            # .astype(str).str.replace('(\\.0)$', "", regex=True)
            lower_bound_text = Series(lower_bound).apply(get_short_text_for_large_number)
            upper_bound_text = Series(upper_bound).apply(get_short_text_for_large_number)
            if show_values_as_range:
                # display as range notation
                display_value: Series = ("[" if lower_inclusive else "(") \
                                        + lower_bound_text + ", " + upper_bound_text + \
                                        ("]" if upper_inclusive else ")")
            else:
                # put in simple terms
                display_value: Series = lower_bound_text + " to " + upper_bound_text
                # if the lower bound is -inf, then set to < upperbound
                lower_is_inf_mask = lower_bound == -np_inf
                display_value[lower_is_inf_mask] = "< " + upper_bound_text[lower_is_inf_mask]
                # if the upper bound is inf, then set to > lowerbound
                upper_is_inf_mask = upper_bound == np_inf
                display_value[upper_is_inf_mask] = "> " + lower_bound_text[upper_is_inf_mask]
            # add in the units
            display_value = self.get_text_with_units(display_value)
            # if different lower / upper bound value, then keep. Otherwise if same, replace with the lower bound.
            display_value = display_value.where(lower_bound_text != upper_bound_text, lower_bound_text)
            # also add the lower and upper bound value
            return display_value, lower_bound, upper_bound
        else:
            # just convert over to text and save
            display_value = col_val.astype(str)
            return self.get_text_with_units(display_value), None, None

    def get_value_as_text(self, value: Any, prettify=True):
        if prettify:
            if isinstance(value, int) or isinstance(value, float):
                if self.is_continuous:
                    return self.get_text_with_units(get_short_text_for_large_number(value))
                else:
                    return self.get_text_with_units(str(remove_decimal_if_needed(value)))
            else:
                return self.get_text_with_units(str(value))

        return str(value)

    def get_text_with_units(self, existing_text: str | Series):
        if self.units is not None:
            if self.units == "Dollars":
                return "$" + existing_text
            else:
                return existing_text + " " + self.units
        else:
            return existing_text


class RiskColumnContinuousTemplate:
    def __init__(self, normalizer: DataNormalizer,
                 min_value: float | int | None = None, max_value: float | int | None = None,
                 units: str | None = None):
        self.normalizer = normalizer
        self.min_value = min_value
        self.max_value = max_value
        self.units = units
