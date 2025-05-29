# provides linear color interpolation between data points
from __future__ import annotations
from typing import Tuple, List
from overall_helpers import *


class ColorDataPoint:
    # contains the color for a known data value
    def __init__(self, value: float, color: Tuple | List, round_decimals: int = 0):
        # the data value given
        self.value = value
        # the R, G, B values for the given color point
        self.color: Tuple | List = color
        self.round_decimals = round_decimals

    def interpolate(self, data, other_pt: ColorDataPoint):
        if other_pt is None:
            return self.color
        else:
            assert len(other_pt.color) == len(self.color), "Colors are not of the same length"
            percent = (data - self.value) / (other_pt.value - self.value)
            # interpolate for each of the points
            interpolated_color = [round(percent * (c2 - c1) + c1, self.round_decimals)
                                  for c1, c2 in zip(self.color, other_pt.color)]
            return interpolated_color


class ColorInterpolator:
    UNSET_NUM_COLORS = -1
    UNSET_DECIMALS = -1

    def __init__(self, num_colors: int = UNSET_NUM_COLORS, round_decimals: int = UNSET_DECIMALS):
        self.round_decimals = round_decimals
        self.num_colors = num_colors
        self.data_points: ModifySafeList[ColorDataPoint] = ModifySafeList()

    def register_data_point(self, value: float, color: Tuple | List):
        # run the checks
        if self.num_colors == ColorInterpolator.UNSET_NUM_COLORS:
            self.num_colors = len(color)
        else:
            assert len(color) == self.num_colors, f"Colors ({color}) is not of expected length ({self.num_colors})"

        # find the position that is closest to but lower than
        is_not_inserted = True
        # by default, slot to add to the end.
        # if finds a better spot, then will insert in there
        insert_position = len(self.data_points)
        built_color = ColorDataPoint(value, color)
        for i, d in enumerate(self.data_points):
            if value < d.value:
                # if less than the given point, then insert at position
                insert_position = i
                break
            elif value == d.value:
                # overwrite the value in the position
                self.data_points[i] = built_color
                is_not_inserted = False
                break

        # insert into the relevant position
        if is_not_inserted:
            self.data_points.insert(insert_position, built_color)

    def interpolate(self, value):
        # provides the color values that are between
        # note that if are less than the minimum value, or greater than the maximum value, will return those
        # first, find the bounds that are between
        # note that the lowerbound will be less than or equal to the value,
        # and the upperbound will be greater than the value
        assert len(self.data_points) > 0, "No Data points found..."
        lowerOrEqualBound: ColorDataPoint | None = None
        upperIndex = 0

        for i, d in enumerate(self.data_points):
            if d.value <= value:
                lowerOrEqualBound = d
                upperIndex = i + 1
            else:
                # since is more, than have found the associated index
                break

        # figure out what the value at the index is
        if upperIndex >= len(self.data_points):
            # means that doesn't exist, so return the lower bound if it exists
            return lowerOrEqualBound.color
        else:
            upperBound: ColorDataPoint = self.data_points[upperIndex]
            # if the lowerbound doesn't exist, then return the upperbound color
            if lowerOrEqualBound is None:
                return upperBound.color
            else:
                # interpolate between the lower bound and the upper bound value
                percent = (value - lowerOrEqualBound.value) / (upperBound.value - lowerOrEqualBound.value)
                # interpolate for each of the points
                interpolated_color = [self.round_color(percent * (c2 - c1) + c1)
                                      for c1, c2 in zip(lowerOrEqualBound.color, upperBound.color)]
                return interpolated_color

    def round_color(self, color_val):
        if self.round_decimals == ColorInterpolator.UNSET_DECIMALS:
            return color_val
        elif self.round_decimals == 0:
            return int(round(color_val, 0))
        else:
            return round(color_val, self.round_decimals)

    def interpolate_rgb_hex(self, input_val):
        r, g, b = self.interpolate(input_val)
        return "#" + convert_to_hex(r) + convert_to_hex(g) + convert_to_hex(b)
