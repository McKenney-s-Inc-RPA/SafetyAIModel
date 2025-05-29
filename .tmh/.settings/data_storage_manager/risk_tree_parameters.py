from __future__ import annotations
from typing import Tuple
from math import comb as calculate_num_combinations
from risk_data_column import *
from numpy import ndarray, zeros as np_zeros, arange as np_arange, sqrt as np_sqrt, unique as np_unique, \
    bincount as np_bin_count
from pandas import isna as pd_isna
from pandas.errors import PerformanceWarning
from numpy.random import default_rng
from warnings import warn, simplefilter
from overall_helpers import *
from itertools import combinations as itertools_combinations, chain as itertools_chain

# pandas gives off the following warning when grouping data based on the given results:
#   PerformanceWarning: DataFrame is highly fragmented.
#   This is usually the result of calling `frame.insert` many times, which has poor performance.
#   Consider joining all columns at once using pd.concat(axis=1) instead.
#   To get a de-fragmented frame, use `newframe = frame.copy()`
# However, these functions aren't implicitly called, and don't hurt execution of the code...
simplefilter(action="ignore", category=PerformanceWarning)


class RiskDecisionTreeParameters:
    def __init__(self, *, n_estimators=1, max_depth=5, min_hours_leaf=150_000, min_injuries_leaf=10,
                 terminating_num_injuries_leaf=10,
                 feature_ratio=0.7, data_ratio=1.0, random_state=None,
                 max_selectable_categories=3, max_category_combinations=10_000, max_categories=30,
                 min_partition_hours: float = 1000, threshold_of_max_partition_hours: float = 0.01,
                 chi_squared_confidence=0.1,
                 gradient_boost_learning_rate=0.1, gradient_boost_iterations=0,
                 selection_percent_of_max: float = 0.95, debug_retain_percent_of_max: float | None = None,
                 injury_column: str | None = None, hours_column: str | None = None):
        # the number of estimators to use within a random forest model
        self.n_estimators = n_estimators
        # the maximum depth for each of the trees
        self.max_depth = max_depth
        # these two parameters adjust the confidence of the leaf
        # note that a leaf must have one of these two minimums.
        # this allows that if lots of injuries are seen, then will be able to bypass the minimum hours
        self.min_hours_leaf = min_hours_leaf
        self.min_injuries_leaf = min_injuries_leaf
        # the number of injuries after which a split won't be attempted
        # this is to prevent against over-fitting the model
        self.terminating_num_injuries_leaf = terminating_num_injuries_leaf
        # the random state for selecting the features to be included within the decision tree
        self.random_state = random_state
        # the amount of features to be selected within the given tree
        self.feature_ratio = feature_ratio
        assert 0 < self.feature_ratio <= 1, "The feature ratio must be greater than 0, and less than or equal to 1."
        # the portion of the data to use for training
        self.data_ratio = data_ratio
        # the maximum number of categories to use within splits
        # note that this number can grow exponentially (or worse), so keep relatively low
        # for example, 5 categories can be chosen 1 at a time, 2 at a time, 3 at a time, or 4 at a time (5 / all would not create a split)
        # In this case, the number of combinations is 2^(5-1) = 16. With 10 categories chosen 1 to 9 ways, this would be 2^9 = 512
        self.max_selectable_categories = max_selectable_categories
        self.max_category_combinations = max_category_combinations
        # the maximum number of categories to even consider a feature for splitting
        self.max_categories = max_categories
        self._category_split_indices_lookup: Dict[int, ndarray] = {}
        self._range_bounds_indices_lookup: Dict[int, Tuple[ndarray, ndarray]] = {}
        assert self.max_selectable_categories <= 15, "Consider 10 or fewer categories as the maximum value."
        # a partition is the slice of data containing a unique value for a given column
        # for example, the data subset where Branch = ATL
        # some smaller partitions (like Position Type = "Designer") may not be interesting, and
        # therefore shouldn't be considered as a split option
        # this decreases the split sizes and allows for quicker computation
        # the minimum values for hours in a given partition
        self.min_partition_hours = min_partition_hours
        self.threshold_of_max_partition_hours = threshold_of_max_partition_hours
        self.chi_squared_confidence = chi_squared_confidence
        # the default injury and hour column to use
        self.injury_column = injury_column
        self.hours_column = hours_column
        # whether to use gradient boosting
        self.gradient_boost_learning_rate = gradient_boost_learning_rate
        self.gradient_boost_iterations = gradient_boost_iterations
        # how likely the best heuristic value will be chosen
        self.selection_percent_of_max = selection_percent_of_max
        self.debug_retain_percent_of_max = self.selection_percent_of_max if debug_retain_percent_of_max is None else debug_retain_percent_of_max
        assert 0 < selection_percent_of_max <= 1, "The Selection Percentage of Max should be 0 < p <= 1"

    def get_relevant_partition_indices(self, hours: ndarray):
        # gets the partitions (or rows of a data slice) which have enough data to be relevant
        min_threshold = self.min_partition_hours
        max_value = hours.max(initial=0)
        threshold_of_max = max_value * self.threshold_of_max_partition_hours
        # if the percentage of max provides a higher threshold,
        # OR if the max value doesn't meet the minimum threshold,
        # then go with the percentage of max value.
        if threshold_of_max > min_threshold or max_value < self.min_partition_hours:
            min_threshold = threshold_of_max
        # return the indices where the hours is greater than the minimum threshold needed
        return hours >= min_threshold

    def get_range_split_indices(self, num_unique_values):
        # gets the lowerbound and upperbound index for each possible range
        # the ranges include the given values (i.e., 2-3 include both 2 and 3)
        # note that for 4 items (0-indexed), then the pairs will be
        #   - Lowerbound of 0: 0, 1, 2 (note that can't include 3, as would include all values)
        #       - Note that all of these are similar to having a threshold at the given value, so may be able to ignore
        #   - Lowerbound of 1: 1, 2 (note that shouldn't include 3, because 0-0 inversion is 1-3)
        #   - Lowerbound of 2: 2 (not that shouldn't include 3, because 0-1 inversion is 2-3)
        #   - Lowerbound of 3: None (note that 3-3 is the inversion of 0-2, which is shown above
        # This means that 6 pairs will be generated: Lowerbounds: [0, 0, 0, 1, 1, 2] with upperbounds [0, 1, 2, 1, 2, 2]
        # the number of indices is (n-1) + (n-2) + ... + 1 = n * (n-1) / 2.
        # However, since can ignore a lowerbound of 0 (as would be the same as thresholding), then
        # there are 3 pairs: Lowerbounds: [1, 1, 2] with upperbounds [1, 2, 2]
        # the number of indices is n*(n-1)/2 - (n-1) = (n-1)*(n-2)/2
        if num_unique_values in self._range_bounds_indices_lookup:
            return self._range_bounds_indices_lookup[num_unique_values]
        else:
            num_ranges: int = (num_unique_values - 1) * (num_unique_values - 2) / 2
            # Ideally, want a function which maps the index [0, 1, 2...] to the lowerbound
            index_value: ndarray = np_arange(start=0, stop=num_ranges)
            # note that the lowerbounds [0, 0, 0, 1, 1, 2] can be written as 3 - [3, 3, 3, 2, 2, 1]
            # note that this has 3 3's, 2 2's, and 1 1, which follows a similar distribution to the square root function
            # The index can also be written as 6 - [6, 5, 4, 3, 2, 1]
            # The rounded sqrt value of (the reversed index * 2 - 1) will be equivalent to the reversed lowerbound value
            reversed_index: ndarray = num_ranges - index_value
            reversed_lower_bound: ndarray = np_round(np_sqrt(2 * reversed_index - 1))
            lower_bound: ndarray = num_unique_values - reversed_lower_bound - 1
            # next, get the offsets from the lowerbound to the upper bound
            # in the example above, this would be [0, 1, 2, 0, 1, 0]
            # approach: find the difference between the reversed index and the result of a function of the reversed lower bound
            # note that this uses the similar formula as before for finding the number of ranges
            reversed_base_for_offset: ndarray = reversed_lower_bound * (reversed_lower_bound + 1) / 2
            # the reversed base will always be larger than or equal to the index value
            # thus, the reversed offset will be 0 to a negative number
            reversed_offset: ndarray = reversed_index - reversed_base_for_offset
            # subtract the (negative) offset from the lower bound to create the higher upper bound
            upper_bound: ndarray = lower_bound - reversed_offset
            # convert the lower and upper bounds to integers, and return
            lower_indices = lower_bound.astype(int)
            upper_indices = upper_bound.astype(int)
            self._range_bounds_indices_lookup[num_unique_values] = (lower_indices, upper_indices)
            return lower_indices, upper_indices

    def get_category_split_indices(self, total_num_categories):
        # given N categories and M max categories, returns P options
        # returns in an array of NxP, where the rows correspond to a given category
        # and the columns correspond to the combinations the category is a part of
        # the array is 0's and 1's so that can be used with matrix multiplication for easy summing
        # Assume are looking at options [A, B, C, D]
        # all options with length 1 are guaranteed to be new & not complements
        # all options with length 2 may be new & not complements. For example, [A, B] and [C, D] are complements
        # all options with length 3 are guaranteed to be complements of the options of length 2
        # For an even number of options,
        #   - all options with length i < N / 2 will be new and unique
        #   - All options with length i > N / 2 will not be new and unique
        #   - All options with length i == N / 2 may or may not be unique
        # For an odd number of options,
        #   - All options with length i < N/2 will be new and unique (similar to above)
        #   - All options with length i > N/2 will not be new and unique (similar to above)
        #   - This is because any combination with length i > N/2 will be a complement of a set with length N - i, which is less than N/2
        # therefore, only need to go up to N/2, or the max categories, whichever occurs first
        # for i == N/2, for now, just calculate all options, and take the computation hit...
        if total_num_categories in self._category_split_indices_lookup:
            return self._category_split_indices_lookup[total_num_categories]
        else:
            max_selection = min(total_num_categories // 2, self.max_selectable_categories)
            option_indices = list(range(total_num_categories))
            # only add options if the total options doesn't go above a set limit
            total_options = 0
            max_categories_feasible = 0
            while max_categories_feasible < max_selection:
                # try the next category count (plus 1)
                new_options = calculate_num_combinations(total_num_categories, max_categories_feasible + 1)
                new_total_options = total_options + new_options
                if new_total_options < self.max_category_combinations:
                    total_options = new_total_options
                    max_categories_feasible += 1
                else:
                    break
            option_sets = np_zeros((total_num_categories, total_options))
            applicable_options = itertools_chain.from_iterable(itertools_combinations(option_indices, n)
                                                               for n in range_one_based(max_categories_feasible))
            for i, options_selected in enumerate(applicable_options):
                option_sets[options_selected, i] = 1
            self._category_split_indices_lookup[total_num_categories] = option_sets
            return option_sets


class RiskTreeParametersForDataSet:
    # contains the parameters for training a model from a given data set
    def __init__(self, model_params: RiskDecisionTreeParameters, new_injuries_column: str | None, new_hours_column: str | None,
                 features: List[RiskDataColumn], col_unique_handler: RiskColumnSetUniqueHandler):
        self.model_params = model_params
        # the column used for injuries
        # this includes all injury counts
        self.injury_col = new_injuries_column if new_injuries_column is not None else model_params.injury_column
        # the injury column to train on
        # note that this could be the same as the injury column, or it could be the injury error column
        self.training_injury_col = self.injury_col
        self.hours_col = new_hours_column if new_hours_column is not None else model_params.hours_column
        self.cols_to_sum = {self.injury_col, self.hours_col}
        self.features = features
        # figure out the number of features, using at least one feature
        num_cols = len(features)
        assert num_cols > 0, "No data columns given..."
        self.num_features_to_use = max(1, round(num_cols * self.model_params.feature_ratio))
        # determine what data to use
        # if are not bootstrapping, then warn if the number of features to use is the same
        # as the number of columns and are using multiple trees
        if self.model_params.data_ratio > 0.95 and self.model_params.n_estimators > 1 and not self.randomly_select_features:
            warn(f"All ({num_cols}) features are used for {self.model_params.n_estimators} trees using most of the data. "
                 + "This could result in trees which are very similar. Are these parameters correct?")

        # seed the random values
        self.random_generator = default_rng(self.model_params.random_state)

        # keep track of the unique values from the data set
        self.col_unique_handler = col_unique_handler

        # keep track of how many splits (or potential splits) the features are a part of
        self.weight_by_feature: EasyDict[str, float] = EasyDict(0)
        self.weight_by_feature.initialize([f.name for f in features])

    @property
    def randomly_select_features(self):
        return self.num_features_to_use < len(self.features)

    @property
    def randomly_select_data(self):
        return self.model_params.data_ratio < 1

    def change_training_injury_col(self, new_injury_col):
        # if the training column is already different from the current injury column, then remove it
        if self.training_injury_col != self.injury_col:
            self.cols_to_sum.remove(self.training_injury_col)
        # only set the training column, as the injury column should never be changed
        self.cols_to_sum.add(new_injury_col)
        self.training_injury_col = new_injury_col

    def select_params_for_training(self, data_for_training: DataFrame, dropna=True, consolidate=True):
        # select which features within the data will look at
        if self.randomly_select_features:
            selected_features = self.random_generator.choice(self.features, self.num_features_to_use, replace=False)
        else:
            selected_features = self.features
        # return a percentage of the rows, if needed
        if self.randomly_select_data:
            selected_data = data_for_training.sample(frac=self.model_params.data_ratio, random_state=self.random_generator)
        else:
            selected_data = data_for_training

        if consolidate:
            # consolidate the data, in case there are any duplicates within the features
            # the data can be consolidated here, as the data going to training has already been randomly selected
            feature_names = [c.name for c in selected_features]
            colsNotUsed = [c for c in selected_data.columns if c not in feature_names and c not in self.cols_to_sum]
            # remove any columns not summed or selected features
            # for more than one column, may need to turn sort off for deterministic behavior
            # don't drop na's, in case they are used later
            # use only observed items, so that if multiple grouping categorical columns are used, then doesn't create a cross-product of all options
            # can sum all remaining items (aka columns wanting to sum)
            # after aggregation, reset the index from hierarchy index to individual row index
            consolidated_data = selected_data.groupby(feature_names, dropna=dropna, sort=False, observed=True,
                                                      as_index=False)[list(self.cols_to_sum)].sum()
        else:
            consolidated_data = selected_data
        return selected_features, consolidated_data

    def group_by(self, data_subset: DataFrame, feature_name: str):
        # perform a group by, and then sort by the given column
        # drop any na's, as will compare the known values against all others (including na's)
        return group_data_frame(data_subset, feature_name, {c: "sum" for c in self.cols_to_sum}, sort=True, dropna=True)

    def group_by_optimized(self, data_subset: DataFrame, feature_name: str):
        return self.col_unique_handler.feature_unique_lookup[feature_name].group_by_index(data_subset, self.injury_col,
                                                                                          self.training_injury_col, self.hours_col)


class RiskDataColumnSet(ModifySafeList):
    # collection of features. If a feature doesn't have a good split, then will remove
    def __init__(self, features: List[RiskDataColumn]):
        super().__init__(features)
        self._features_by_name = {f.name: f for f in features}

    def __getitem__(self, item):
        # if is giving a name, then lookup within the dictionary
        if isinstance(item, str):
            return self._features_by_name[item]
        else:
            return super().__getitem__(item)

    def __str__(self):
        return f"Data Risk Column Set of size {self._num_items}"

    def remove(self, feature: RiskDataColumn):
        if feature.name in self._features_by_name:
            # remove from the lookup
            self._features_by_name.pop(feature.name)
            super().remove(feature)

    def _add_feature_to_lookup(self, feature: RiskDataColumn):
        self._features_by_name[feature.name] = feature

    def insert(self, index, feature: RiskDataColumn):
        super().insert(index, feature)
        self._add_feature_to_lookup(feature)

    def append(self, feature: RiskDataColumn):
        super().append(feature)
        self._add_feature_to_lookup(feature)

    def copy(self):
        return RiskDataColumnSet(self._items.copy())

    def remove_if_not_interesting(self, feature: RiskDataColumn, num_options: int):
        # if only one option, then no longer is a pertinent feature
        is_not_interesting = num_options <= 1
        if is_not_interesting:
            self.remove(feature)
        return is_not_interesting

    def copy_for_split(self, feature: RiskDataColumn, num_options: int):
        # create a copy, and log the feature option counts
        c = self.copy()
        c.remove_if_not_interesting(feature, num_options)
        return c


class RiskColumnSetUniqueHandler:
    def __init__(self, features: List[RiskDataColumn], data_set: DataFrame):
        # build out the unique values for each of the feature columns
        self.feature_unique_lookup = {f.name: RiskColumnUniqueHandler(f.name, data_set)
                                      for f in features
                                      if f.name in data_set.columns}

    def convert_to_index_values(self, data_subset: DataFrame):
        # converts the values from each column in the dataframe to the index value (sorted increasing)
        # this will standardize and greatly increase training efficiency
        data_index_values = data_subset.copy()
        for c, v in self.feature_unique_lookup.items():
            data_index_values[c] = v.convert_to_index(data_index_values[c])
        return data_index_values


class RiskColumnUniqueHandler:
    # handles the unique values for a column, to help speed up the group by process
    # written using Numpy to bypass slow pandas execution
    def __init__(self, name: str, data: DataFrame):
        # gets the ordered unique value of the given data frame
        # note that any subset of the given data will contain all unique values
        # precomputes the order, so when given a subset, can tell what order the values should go in
        # note that some of the indices may be skipped, which is OK
        self.name = name
        feature_values: ndarray = data[self.name].values
        self.unique_sorted_feature_values: ndarray | None = None
        self.index_arr: ndarray | None = None
        self.include_arr: ndarray | None = None
        self.index_lookup: dict[Any, int] = {}
        self.value_lookup: dict[int, Any] = {}
        try:
            self.unique_sorted_feature_values = np_unique(feature_values)
            self.unique_sorted_feature_values.sort()
            for i, k in enumerate(self.unique_sorted_feature_values):
                self.index_lookup[k] = i
                self.value_lookup[i] = k
            self.index_arr = np_arange(len(self.unique_sorted_feature_values), dtype=int)
            # ignore any nan values
            self.include_arr = ~pd_isna(self.unique_sorted_feature_values)
        except Exception as e:
            print(f"Problem with creating values for column '{self.name}'")
            raise e

    def convert_to_index(self, data_col: Series) -> Series:
        return data_col.map(self.index_lookup)

    def convert_from_index(self, data_index) -> Any:
        return self.value_lookup[int(data_index)]

    def group_by_index(self, data_subset: DataFrame, injury_col: str, predicting_injury_col: str, hours_col: str):
        # note that the injury column contains all injuries,
        # whereas the predicting injury column contains the values that the model is predicting
        # which could be the total injuries, or the injury error
        # assumes that the data has been converted over to the index value
        feature_index = data_subset[self.name].values
        # note that may not use all of the unique features, so take only up until the maximum feature used
        max_feature_index = feature_index.max(initial=0)
        max_length = max_feature_index + 1
        used_unique_sorted_features = self.index_arr[:max_length]
        # sum the values, placing each within the index of the given feature
        injuries_by_feature_value = np_bin_count(feature_index, weights=data_subset[injury_col].values, minlength=max_length)
        hours_by_feature_value = np_bin_count(feature_index, weights=data_subset[hours_col].values, minlength=max_length)
        # only keep values that have hours associated with them, and are not ignored (i.e., are nan)
        should_keep = (hours_by_feature_value > 0) & (self.include_arr[:max_length])
        # create the output data frame
        output_data = {self.name: used_unique_sorted_features[should_keep],
                       injury_col: injuries_by_feature_value[should_keep],
                       hours_col: hours_by_feature_value[should_keep]}
        if predicting_injury_col != injury_col:
            predicted_injuries_by_feature_value = np_bin_count(feature_index, weights=data_subset[predicting_injury_col].values, minlength=max_length)
            output_data[predicting_injury_col] = predicted_injuries_by_feature_value[should_keep]
        return DataFrame(output_data)
