from __future__ import annotations
from feature_criteria import FeatureCriteriaSet
from risk_tree_parameters import *
from abc import abstractmethod, ABC
from render_helpers import CodeStyle
from pandas.api.types import is_number as pd_is_number
from typing import Tuple, Callable, Iterable
from risk_data_column import *
from json_manager import JsonBase
from numpy import abs as np_abs, \
    sqrt as np_sqrt, matmul as np_matmul, logical_and as np_and, logical_not as np_not, \
    log10 as np_log10, maximum as np_maximum, nonzero as np_nonzero, any as np_any, vstack as np_vstack
from overall_helpers import *
from render_helpers import get_short_text_for_large_number, remove_decimal_if_needed
from scipy.stats import chisquare

RISK_ROUNDING = 2


def convert_to_storable(value: Any):
    # converts potential numpy types and other types to their simple equivalent
    if pd_is_number(value):
        return remove_decimal_if_needed(float(value))
    else:
        return value


def calculate_risk_weighted_standard_deviation(injuries: Series, hours: Series):
    # calculates the standard deviation of the risks, weighted by the hours
    # assumes that all hours are non-zero
    num_rows = len(injuries)
    # first, calculate the average risk (i.e., sum of injuries divided by sum of hours)
    total_injuries = injuries.sum()
    total_hours = hours.sum()
    avg_risk = total_injuries / total_hours
    # next, calculate the risk array
    risk_arr = injuries / hours
    # next, calculate the average weight (i.e., average number of hours)
    avg_hours = total_hours / num_rows
    # next, calculate the square residuals, weighted by the hours
    weighted_sq_residuals: Series = hours * (risk_arr - avg_risk) ** 2
    weighted_sq_residual_sum: float = weighted_sq_residuals.sum()
    # lastly, calculate the weighted standard deviation
    return (weighted_sq_residual_sum / avg_hours / (num_rows - 1)) ** 0.5


class RiskTreeBaseNode(JsonBase):
    lowest_depth = 0

    def __init__(self, num_injuries, num_hours, parent: RiskTreeSplitNode | None):
        self.parent = parent
        # round the hours and injuries, for storage
        self.num_hours = int(round(num_hours, 0))
        self.num_injuries = remove_decimal_if_needed(float(round(num_injuries, 3)))
        # normalize the risk over 100 full-time employees, similar to RIR or AICR
        self._risk_value = round(calculate_risk(self.num_injuries, self.num_hours), RISK_ROUNDING)
        if self.parent is None:
            self.depth = RiskTreeBaseNode.lowest_depth
            # default an empty node to being left, for sorting purposes
        else:
            self.depth = self.parent.depth + 1
        self._height = -1
        self._ordering_number = -1

    def __str__(self):
        return f"Node with risk {self.injury_risk} ({self.num_injuries} injuries over {self.num_hours_text} hours)"

    @property
    def is_left(self):
        if self.parent is None:
            # default an empty node to being left, for sorting purposes
            return True
        else:
            return self.parent.left == self

    @property
    def num_hours_text(self):
        return get_short_text_for_large_number(self.num_hours)

    @property
    def injury_risk(self):
        return self._risk_value

    @property
    def is_leaf(self) -> bool:
        return True

    def render(self, render_style: CodeStyle):
        return render_style.build_value(self.depth, self.injury_risk)

    @property
    def height(self):
        # the root should have the highest height, and the furthest leaf should have a height of 0.
        # height is based on the lowest node (which is assigned 0)
        # assign the highest height to the root of the tree
        if self._height < 0:
            if self.parent is None:
                self._height = self._get_largest_depth()
            else:
                self._height = self.parent.height - 1
        return self._height

    def _get_largest_depth(self):
        return self.depth

    def predict_values(self, dataset: DataFrame, predictions: ndarray, applies_to: ndarray):
        # for a leaf, apply the current risk to all values of the current leaf
        predictions[applies_to] = self.injury_risk

    @property
    def ordering_number(self):
        # figures out the ordering number
        # the ordering number will be the 2 ^ depth, plus (0 if on the parent's left,
        # otherwise the parent's ordering number)
        # for example, assume 1 parent split node, which has a left child and a right child
        # the parent will be 1 (to show first)
        # the left node will be 2
        # the child node will be 3
        # As another example, take the following tree
        # Base Node (1)
        #   Left Child (2)
        #       L Left (3)
        #       L Right (4)
        #   Right Child (5)
        #       R Left (6)
        #       R Right (7)
        # Each left child is 1 + parent order
        # Each right child is 1 + parent order + 2^(node height)
        # this will create an integer numbering for the nodes, to place them in display order

        # if not calculated yet, then do so.
        if self._ordering_number <= 0:
            # calculate the base order number
            # note that if doesn't have a parent, then should treat as a left node, so nothing is added
            self._ordering_number = 1 if self.parent is None else self.parent.ordering_number + 1
            # if is the right node, then add enough space for all left children
            if not self.is_left:
                self._ordering_number += 2 ** self.height
        return self._ordering_number

    def get_conditions(self) -> FeatureCriteriaSet:
        condition_info = FeatureCriteriaSet()
        ancestor = self.parent
        meets_criteria = self.is_left
        while ancestor is not None:
            ancestor.split_criteria.log_within_feature_bound_set(condition_info, meets_criteria)
            # move up a level
            # if the node is on the right of the split, then is less than the split threshold
            meets_criteria = ancestor.is_left
            ancestor = ancestor.parent
        return condition_info

    def get_readable_rule(self, prettify=True):
        condition_info = self.get_conditions()
        if condition_info.has_data:
            condition_text = condition_info.get_conditions_text(prettify)
            return f"When {condition_text} then Risk = {self.injury_risk} ({self.num_injuries} injuries over {self.num_hours_text} hours)"
        else:
            # means that no data was given
            return f"Risk = {self.injury_risk} ({self.num_injuries} injuries over {self.num_hours_text} hours)"


class RiskTreeSplitNode(RiskTreeBaseNode):
    def __init__(self, split_criteria: RiskTreeNodeSplitCriteriaBase, num_injuries, num_hours,
                 parent: RiskTreeSplitNode | None):
        super().__init__(num_injuries, num_hours, parent)
        self.left: RiskTreeBaseNode | None = None
        self.right: RiskTreeBaseNode | None = None
        self.split_criteria = split_criteria

    def __str__(self):
        return f"Split Node where {self.split_criteria.get_where_statement()} " \
               f"with risk {self.injury_risk} ({self.num_injuries} injuries over {self.num_hours_text} hours)"

    @property
    def feature_name(self):
        return self.split_criteria.feature.name

    @property
    def is_leaf(self) -> bool:
        return False

    def predict_values(self, dataset: DataFrame, predictions: ndarray, applies_to: ndarray):
        # for a split node, pass the applicable rows (in this case, applicable mask)
        # and the rows that meet the criteria to the left side,
        # and the rows that don't meet the criteria to the right side
        splits_true = self.split_criteria.get_true_mask(dataset).values
        true_values = np_and(applies_to, splits_true)
        false_values = np_and(applies_to, np_not(splits_true))
        self.left.predict_values(dataset, predictions, true_values)
        self.right.predict_values(dataset, predictions, false_values)

    def _get_largest_depth(self):
        return max(self.left._get_largest_depth(), self.right._get_largest_depth())

    def render(self, render_style: CodeStyle):
        left_rendering = self.left.render(render_style)
        right_rendering = self.right.render(render_style)
        return render_style.build_full_if_statement(self.depth, self.feature_name, self.threshold, left_rendering,
                                                    right_rendering)

    def explain_split(self, split_direction: bool, prettify=True):
        condition_info = FeatureCriteriaSet()
        self.split_criteria.log_within_feature_bound_set(condition_info, split_direction)
        return condition_info.get_conditions_text(prettify)


class RiskTreeNodeSplitCriteriaBase(JsonBase):
    def __init__(self, feature: RiskDataColumn):
        self.feature = feature

    @abstractmethod
    # gets the true mask for a given set of data
    def _get_true_mask_from_data(self, data: Series) -> Series: ...

    def get_true_mask(self, data: DataFrame):
        return self._get_true_mask_from_data(data[self.feature.name])

    def split_data(self, data: DataFrame):
        true_mask = self.get_true_mask(data)
        true_data = data.loc[true_mask]
        false_data = data.loc[~true_mask]
        return true_data, false_data

    @abstractmethod
    # returns a string explaining how the split occurs, like "Split where feature 'X' <= Y"
    def get_where_statement(self) -> str: ...

    def __str__(self):
        return self.get_where_statement()

    @abstractmethod
    def log_within_feature_bound_set(self, condition_info: FeatureCriteriaSet, is_left: bool): ...

    @abstractmethod
    def convert_criteria_from_index(self, col_value_handler: RiskColumnUniqueHandler): ...

    @staticmethod
    def convert_from_index(col_value_handler: RiskColumnUniqueHandler, index_value):
        return convert_to_storable(col_value_handler.convert_from_index(index_value))


class RiskTreeNodeThresholdSplit(RiskTreeNodeSplitCriteriaBase):
    # true includes all values less than or equal to the threshold
    def __init__(self, feature: RiskDataColumn, threshold: float):
        super(RiskTreeNodeThresholdSplit, self).__init__(feature)
        # convert over to a float in case is not already
        self.threshold = convert_to_storable(threshold)

    def _get_true_mask_from_data(self, data: Series) -> Series:
        return data <= self.threshold

    def log_within_feature_bound_set(self, condition_info: FeatureCriteriaSet, is_left: bool):
        condition_info.filter_with_threshold(self.feature, self.threshold, is_left)

    def get_where_statement(self) -> str:
        return f"'{self.feature.name}' <= {self.threshold}"

    def convert_criteria_from_index(self, col_value_handler: RiskColumnUniqueHandler):
        self.threshold = self.convert_from_index(col_value_handler, self.threshold)


class RiskTreeNodeRangeSplit(RiskTreeNodeSplitCriteriaBase):
    # true includes all values that between (inclusive) the min and max value
    def __init__(self, feature: RiskDataColumn, min_value: float, max_value: float):
        super(RiskTreeNodeRangeSplit, self).__init__(feature)
        self.min_value = convert_to_storable(min_value)
        self.max_value = convert_to_storable(max_value)

    def _get_true_mask_from_data(self, data: Series) -> Series:
        return (data >= self.min_value) & (data <= self.max_value)

    def log_within_feature_bound_set(self, condition_info: FeatureCriteriaSet, is_left: bool):
        condition_info.filter_with_range(self.feature, self.min_value, self.max_value, is_left)

    def get_where_statement(self) -> str:
        return f"'{self.feature.name}' BETWEEN {self.min_value} AND {self.max_value}"

    def convert_criteria_from_index(self, col_value_handler: RiskColumnUniqueHandler):
        self.min_value = self.convert_from_index(col_value_handler, self.min_value)
        self.max_value = self.convert_from_index(col_value_handler, self.max_value)


class RiskTreeNodeSetSplit(RiskTreeNodeSplitCriteriaBase):
    def __init__(self, feature: RiskDataColumn, values: List[Any]):
        super(RiskTreeNodeSetSplit, self).__init__(feature)
        self.values = values

    def _get_true_mask_from_data(self, data: Series) -> Series:
        return data.isin(self.values)

    def log_within_feature_bound_set(self, condition_info: FeatureCriteriaSet, is_left: bool):
        condition_info.filter_with_list(self.feature, self.values, is_left)

    def get_where_statement(self) -> str:
        return f"'{self.feature.name}' IN {self.values}"

    def convert_criteria_from_index(self, col_value_handler: RiskColumnUniqueHandler):
        for i in range(len(self.values)):
            self.values[i] = self.convert_from_index(col_value_handler, self.values[i])


class RiskTreeNodeTrainingData:
    # contains the injuries, training data, and node data for the node being trained
    def __init__(self, consolidated_data: DataFrame, model_specifics: RiskTreeParametersForDataSet, feature: RiskDataColumn):
        self.feature = feature
        # figure out which items should be able to look at
        hours_vals = self.get_array_from_column(consolidated_data, model_specifics.hours_col)
        relevant_partition_mask = model_specifics.model_params.get_relevant_partition_indices(hours_vals)
        # only store the values for the metric values that have enough hours (i.e., can be split on)
        self.metric_values = self.get_array_from_column(consolidated_data, feature.name)[relevant_partition_mask]
        self.num_relevant_options = len(self.metric_values)
        self.hours = self.get_values_from_data(consolidated_data, model_specifics.hours_col, relevant_partition_mask)
        self.injuries = self.get_values_from_data(consolidated_data, model_specifics.injury_col, relevant_partition_mask)
        if model_specifics.injury_col != model_specifics.training_injury_col:
            self.training_injuries = self.get_values_from_data(consolidated_data, model_specifics.training_injury_col, relevant_partition_mask)
        else:
            self.training_injuries = self.injuries

    @staticmethod
    def get_values_from_data(consolidated_data: DataFrame, col_name: str, relevant_partition_mask: ndarray):
        return RiskTreeNodeTrainingDataColumn(RiskTreeNodeTrainingData.get_array_from_column(consolidated_data, col_name), relevant_partition_mask)

    @staticmethod
    def get_array_from_column(consolidated_data: DataFrame, col_name: str):
        return consolidated_data[col_name].values


class RiskTreeNodeTrainingDataColumn:
    # contains the data for a given column (like hours, injuries, etc)
    def __init__(self, data_values: ndarray, relevant_partition_mask: ndarray):
        self.data_values = data_values
        self.relevant_partition_mask = relevant_partition_mask
        self._sum = None
        self._cum_sum = None
        self._sum_between_ranges = None
        self._category_combination_sums = None

    def sum(self):
        if self._sum is None:
            # get the sum of the entire array
            self._sum = self.data_values.sum()
        return self._sum

    def cumsum(self):
        # only get the sum at the given options
        if self._cum_sum is None:
            self._cum_sum = self.data_values.cumsum()[self.relevant_partition_mask]
        return self._cum_sum

    def get_sum_in_between_indices(self, lower_bound_indices: ndarray, upper_bound_indices: ndarray):
        # ranges include the lower bound, so add back in
        if self._sum_between_ranges is None:
            cs = self.cumsum()
            included_data = self.data_values[self.relevant_partition_mask]
            self._sum_between_ranges = cs[upper_bound_indices] - cs[lower_bound_indices] + included_data[lower_bound_indices]
        return self._sum_between_ranges

    def get_sum_of_categories(self, option_indices: ndarray):
        # pick the relevant hours and injuries for each of the masks, and sum together
        # note that can use matrix multiplication for this
        if self._category_combination_sums is None:
            included_data = self.data_values[self.relevant_partition_mask]
            self._category_combination_sums = np_matmul(included_data, option_indices)
        return self._category_combination_sums


class RiskTreeNodeBestFitFinder(ABC):
    # gets an array with all included values at each combination / slice
    @abstractmethod
    def get_included_values(self, given_data: RiskTreeNodeTrainingDataColumn) -> ndarray: ...

    # get the number of dimensions / metrics that are chosen by this index
    @abstractmethod
    def get_number_included_metrics(self, index): ...

    # builds the node criteria (i.e., split criteria) at the given index
    @abstractmethod
    def build_node_criteria(self, index, f: RiskDataColumn, metric_values: ndarray): ...


class RiskTreeNodeBestFitFinderThreshold(RiskTreeNodeBestFitFinder):
    # works with data that is less than or equal to a given threshold
    def get_included_values(self, given_data: RiskTreeNodeTrainingDataColumn) -> ndarray:
        return given_data.cumsum()

    def get_number_included_metrics(self, index):
        # since includes all values less than or equal to the threshold, then that many values remain
        return index

    def build_node_criteria(self, index, f: RiskDataColumn, metric_values: ndarray):
        return RiskTreeNodeThresholdSplit(f, metric_values[index])


class RiskTreeNodeBestFitFinderRange(RiskTreeNodeBestFitFinder):
    # works with data that is between two values
    def __init__(self, model_specifics: RiskTreeParametersForDataSet, num_metric_options: int):
        # find the best gini coefficient and corresponding lower and upper index values
        self.lower_bound_indices, self.upper_bound_indices = model_specifics.model_params.get_range_split_indices(num_metric_options)

    def get_included_values(self, given_data: RiskTreeNodeTrainingDataColumn) -> ndarray:
        return given_data.get_sum_in_between_indices(self.lower_bound_indices, self.upper_bound_indices)

    def get_number_included_metrics(self, index):
        # find the difference between the lower and upper bound values
        # the number of options included will include the upper and lower index, so add 1
        return self.upper_bound_indices[index] - self.lower_bound_indices[index] + 1

    def build_node_criteria(self, index, f: RiskDataColumn, metric_values: ndarray):
        lower_index = self.lower_bound_indices[index]
        upper_index = self.upper_bound_indices[index]
        lower_bound = metric_values[lower_index]
        upper_bound = metric_values[upper_index]
        if lower_index == upper_index and not f.is_continuous:
            # since are the same, then set as a contains
            # however, only perform if is non-continuous.
            # Otherwise, will be stuck with X = 400 instead of X between 400 and 500
            return RiskTreeNodeSetSplit(f, [lower_bound])
        else:
            # since is a range, then provide the range bounds
            return RiskTreeNodeRangeSplit(f, lower_bound, upper_bound)


class RiskTreeNodeBestFitFinderCategoryGroup(RiskTreeNodeBestFitFinder):
    # works with data that is between two values
    def __init__(self, model_specifics: RiskTreeParametersForDataSet, num_metric_options: int):
        # find the option indices associated with each of the values
        self.option_indices = model_specifics.model_params.get_category_split_indices(num_metric_options)

    def get_included_values(self, given_data: RiskTreeNodeTrainingDataColumn) -> ndarray:
        # pick the relevant hours and injuries for each of the masks, and sum together
        # note that can use matrix multiplication for this
        return given_data.get_sum_of_categories(self.option_indices)

    def get_number_included_metrics(self, index):
        # get the number of options at the given index that are 1.
        # Note that can sum these values, and will give the count
        return self._get_options_at_index(index).sum()

    def _get_options_at_index(self, index) -> ndarray:
        return self.option_indices[:, index]

    def build_node_criteria(self, index, f: RiskDataColumn, metric_values: ndarray):
        categories_included = list(metric_values[self._get_options_at_index(index) == 1])
        return RiskTreeNodeSetSplit(f, categories_included)


class RiskTreeNodeRequest:
    # both of the following criteria heuristics are negative, as all heuristics should be greater than 0.

    # this is used if the conditions didn't meet minimum requirements for splitting, like minimum hours or injury requirements.
    # since splitting these criteria up further won't distill this down, then remove any values that have this as the best option
    # this is also used as the default value, so that any items better than this will make it so that the feature isn't removed
    SPLIT_NOT_MEET_CRITERIA = -2
    # this is if the heuristic doesn't meet the threshold value for the heuristic, like 95% of the maximum value seen
    # however, splitting rows up may cause a better heuristic to be found, so keep the feature until can no longer split
    HEURISTIC_NOT_MEET_CRITERIA = -1

    # contains a set of data to split apart
    def __init__(self, data: DataFrame, model_specifics: RiskTreeParametersForDataSet,
                 total_injuries: float | None = None,
                 total_training_injuries: float | None = None,
                 total_hours: float | None = None,
                 parent_node: RiskTreeSplitNode | None = None):
        # the subset of data that will be training on
        # reset the training data index, to help with indexing for ranges / category splits
        self.data = data.reset_index(drop=True)
        # the output nodes
        self.parent_node = parent_node
        self.created_node: RiskTreeBaseNode | None = None
        # calculate the total injuries that are within this partition
        self.total_injuries: float = total_injuries if total_injuries is not None else data[model_specifics.injury_col].sum()
        self.total_hours: float = total_hours if total_hours is not None else data[model_specifics.hours_col].sum()
        self.total_training_injuries = total_training_injuries if total_training_injuries is not None else data[
            model_specifics.training_injury_col].sum()

    @property
    def injuries_for_node_risk(self):
        # use the total training injuries, which is what the model is being trained to predict
        # note that this may be the number of injuries, or it could be the error from prior predictions
        return self.total_training_injuries

    def create_node(self, model_specifics: RiskTreeParametersForDataSet, applicable_features: RiskDataColumnSet, data_are_indices=True):
        # check that hasn't gone too deep, and that has injuries to apply towards the safety risk
        curr_depth = RiskTreeBaseNode.lowest_depth if self.parent_node is None else self.parent_node.depth + 1
        if abs(self.total_injuries) <= model_specifics.model_params.terminating_num_injuries_leaf \
                or self.total_hours <= model_specifics.model_params.min_hours_leaf \
                or curr_depth >= model_specifics.model_params.max_depth:
            # since have relatively low injuries, then no point in further splitting, so send to a leaf
            self._create_leaf_node()
        else:
            # find the best split
            chosen_split = self.find_best_split(model_specifics, applicable_features, data_are_indices)

            # now that have chosen the best option, then split the data on the given option, and add to the stack
            if chosen_split is not None:
                # create the split node
                # base off of the total number of injuries to be predicted
                self.created_node = RiskTreeSplitNode(chosen_split.split_criteria, self.injuries_for_node_risk,
                                                      self.total_hours, self.parent_node)

                # queue up the left and right side
                true_data, false_data = chosen_split.split_criteria.split_data(self.data)
                true_node_request = RiskTreeNodeRequest(true_data,
                                                        model_specifics,
                                                        chosen_split.true_split_injuries,
                                                        chosen_split.true_split_training_injuries,
                                                        chosen_split.true_split_hours,
                                                        self.created_node)
                applicable_features_for_true = applicable_features.copy_for_split(chosen_split.split_criteria.feature,
                                                                                  chosen_split.true_options_remaining)
                self.created_node.left = true_node_request.create_node(model_specifics, applicable_features_for_true, data_are_indices)
                false_node_request = RiskTreeNodeRequest(false_data,
                                                         model_specifics,
                                                         self.total_injuries - chosen_split.true_split_injuries,
                                                         self.total_training_injuries - chosen_split.true_split_training_injuries,
                                                         self.total_hours - chosen_split.true_split_hours,
                                                         self.created_node)
                applicable_features_for_false = applicable_features.copy_for_split(chosen_split.split_criteria.feature,
                                                                                   chosen_split.false_options_remaining)
                self.created_node.right = false_node_request.create_node(model_specifics, applicable_features_for_false, data_are_indices)
            else:
                # since a good split wasn't found, then create as a leaf node
                self._create_leaf_node()

        return self.created_node

    def _create_leaf_node(self):
        self.created_node = RiskTreeBaseNode(self.injuries_for_node_risk, self.total_hours, self.parent_node)

    def find_best_split(self, model_specifics: RiskTreeParametersForDataSet, applicable_features: RiskDataColumnSet, data_are_indices=True):
        # go through the variations of the data
        split_options = RiskTreeNodePotentialSplitOptions(model_specifics, first_value=0)
        for f in applicable_features:
            self.add_split_options_for_feature(f, model_specifics, applicable_features, split_options, data_are_indices)
        # log the options within the model specifics
        model_specifics.weight_by_feature += split_options.get_metrics()
        return split_options.choose()

    def add_split_options_for_feature(self, f: RiskDataColumn, model_specifics: RiskTreeParametersForDataSet,
                                      applicable_features: RiskDataColumnSet,
                                      split_options: RiskTreeNodePotentialSplitOptions, data_are_indices=True):
        # keep track of the feature's best gini
        # if the best gini correlates with not being able to meet any criteria,
        # then any future splits wont help any more, so remove from the feature set
        feature_best_score = ScoreValue(RiskTreeNodeRequest.SPLIT_NOT_MEET_CRITERIA, minimum_viable_score=RiskTreeNodeRequest.SPLIT_NOT_MEET_CRITERIA)
        # flag on whether a feature can be removed
        # by default, allow removal if it doesn't provide a gini value
        feature_may_not_be_interesting = True
        # note that have the following splits:
        # - Threshold split (usual split for decision trees): Specified by a given number
        # - Range split: Check if value is between range
        #   - Good for modular numbers, like months
        # - In / Out Set Split: check if the value is in a given set
        # Note that all can be specified in a true (i.e., left) and false (i.e., right)
        # Will decide the split based on how the risk varies between the "true" and "false" values
        # Compare how much the risk varies by a variant of the Gini Coefficient (similar to income inequality)
        # which will be defined as the difference of the ratio of group injuries to total injuries
        # from the ratio of group hours to total hours, or defined as
        #   G = (injuries_group / injuries_total) - (hours_group / hours_total)
        # Note that the other group will necessarily have a value equal to the negative value of the first group, since
        #   G_other = (injuries_other / injuries_total) - (hours_other / hours_total)
        #   = ((injuries_tot - injuries_group) / injuries_total) - ((hours_total - hours_group) / hours_total)
        #   = 1 - (injuries_group / injuries_total) - (1 - (hours_group / hours_total))
        #   = (hours_group / hours_total) - (injuries_group / injuries_total)
        #   = -G
        # Take whichever value is furthest from 0 (i.e., the max of the absolute value)
        # For a threshold split, order by the value, take the cumulative sum of injuries and hours,
        #   then, can calculate the gini coefficient
        # For the range, iterate through all feature values, and use as a lower bound.
        #   Then iterate through all feature values that are greater than that that split, and use as the upper bound
        #   The hours / injuries between the bounds will be the difference between the cumulative hours / injuries
        #   From there, can calculate the gini coefficient
        # For the set, take all unique combinations of the feature values
        #   then calculate the gini coefficient for in / out of the set
        #   Potential optimization: If split into sets [A, B, C, D] and [E, F], note that these sets will have opposite gini coefficients
        #   This will be the case for where any set contains the first element, as any set that does not contain the first element
        #   will be the complement to a set that does contain the first element.
        #   Therefore, only need to test only combinations that have the first element, which will halve the combination checks.
        #   In addition, don't need to check the empty set or the full set, as both will have a gini coefficient of 0.
        # For continuous values, rounding the value won't affect data much, but will significantly improve the algorithm
        # for all split types, will want to group by the feature values, and then order by feature value ascending
        # next, calculate the cumulative sum values
        if data_are_indices:
            # since the data have already been converted over to indices, then can use an efficient method for calculating the data
            consolidated_data = model_specifics.group_by_optimized(self.data, f.name)
        else:
            consolidated_data = model_specifics.group_by(self.data, f.name)

        # if only have one value, then can't split, so just return the best split
        num_metric_options = len(consolidated_data)
        if applicable_features.remove_if_not_interesting(f, num_metric_options):
            # since this feature is no longer interesting (as any splits will only have one value), then remove
            return
        node_training_data = RiskTreeNodeTrainingData(consolidated_data, model_specifics, f)
        # if only has one relevant value, then quit out
        # note that this could be monopolized by one large value, so don't get rid of this feature yet,
        # as future splitting may make the value better
        if node_training_data.num_relevant_options <= 1:
            return

        if f.can_threshold:
            # find the best gini coefficient and corresponding index
            fit_finder = RiskTreeNodeBestFitFinderThreshold()
            self._add_splits_for_feature_using_finder(feature_best_score, split_options,
                                                      num_metric_options, model_specifics,
                                                      node_training_data, fit_finder)
        if f.can_range and node_training_data.num_relevant_options > 2:
            # only allow a split if more than 2 options. Otherwise, would be the equivalent to a threshold
            fit_finder = RiskTreeNodeBestFitFinderRange(model_specifics, node_training_data.num_relevant_options)
            self._add_splits_for_feature_using_finder(feature_best_score, split_options,
                                                      num_metric_options, model_specifics,
                                                      node_training_data, fit_finder)
        if f.can_categorize:
            if node_training_data.num_relevant_options > model_specifics.model_params.max_categories:
                # since there are too many options, then don't run
                # however, keep the column, in case becomes interesting later
                feature_may_not_be_interesting = False
            else:
                fit_finder = RiskTreeNodeBestFitFinderCategoryGroup(model_specifics, node_training_data.num_relevant_options)
                self._add_splits_for_feature_using_finder(feature_best_score, split_options,
                                                          num_metric_options, model_specifics,
                                                          node_training_data, fit_finder)
        # if the best score for the feature doesn't meet the baseline score, then remove from the features
        # this is because if it does not meet a criteria now, then when it is split further, it definitely wont meet the criteria
        if feature_best_score.score == RiskTreeNodeRequest.SPLIT_NOT_MEET_CRITERIA and feature_may_not_be_interesting:
            applicable_features.remove(f)

    @staticmethod
    def _add_splits_for_feature_using_finder(feature_best_score: ScoreValue,
                                             split_options: RiskTreeNodePotentialSplitOptions,
                                             num_metric_options: int,
                                             model_specifics: RiskTreeParametersForDataSet,
                                             node_training_data: RiskTreeNodeTrainingData,
                                             fit_finder: RiskTreeNodeBestFitFinder):
        # calculate the included items
        included_injuries: ndarray = fit_finder.get_included_values(node_training_data.injuries)
        included_training_injuries: ndarray = fit_finder.get_included_values(node_training_data.training_injuries)
        included_hours: ndarray = fit_finder.get_included_values(node_training_data.hours)
        # find the number of injuries not within each bucket
        leftover_injuries = node_training_data.injuries.sum() - included_injuries
        # find the number of hours not within each bucket
        leftover_hours = node_training_data.hours.sum() - included_hours
        # find the number of injuries not within each bucket that want to predict
        leftover_training_injuries = node_training_data.training_injuries.sum() - included_training_injuries

        # calculate the heuristic array for the included injuries
        # note that all hours should always be greater than 0
        # if all of the injuries are greater than (or equal to) 0, then can use binomial split
        # otherwise, use z score difference
        use_z_score_when_negative = False
        if use_z_score_when_negative and (np_any(included_training_injuries < 0) or np_any(leftover_training_injuries < 0)):
            heuristic_vals = RiskTreeNodeRequest._calculate_z_score_difference(included_training_injuries, included_hours,
                                                                               leftover_training_injuries, leftover_hours)
        else:
            heuristic_vals = RiskTreeNodeRequest._calculate_chi_squared_heuristic((included_training_injuries, leftover_training_injuries),
                                                                                  (included_hours, leftover_hours),
                                                                                  model_specifics.model_params.chi_squared_confidence)
            # heuristic_vals = RiskTreeNodeRequest._calculate_binomial_split(included_training_injuries, included_hours,
            #                                                                leftover_training_injuries, leftover_hours,
            #                                                                total_injuries, total_hours)

        # find the split that maximizes the heuristic, that still meets the model specific criteria
        # sets the output to a low number where a split isn't feasible, because it would go against the parameter criteria
        # each side MUST have either the minimum hours or the minimum injuries

        # return any values that don't meet the minimum injury and minimum hours threshold
        min_injuries_leaf = model_specifics.model_params.min_injuries_leaf
        min_hours_leaf = model_specifics.model_params.min_hours_leaf
        cant_split = ((included_hours < min_hours_leaf) & (included_injuries < min_injuries_leaf)) | \
                     ((leftover_hours < min_hours_leaf) & (leftover_injuries < min_injuries_leaf))
        heuristic_vals[cant_split] = RiskTreeNodeRequest.SPLIT_NOT_MEET_CRITERIA
        # picks an index of the heuristic array that is a percentage of the max
        # note if the heuristic is 1, this is equivalent to picking one of the indices with the maximum value
        # first, find the threshold that must be greater than
        # use the initial value as the maximum value, so that finds the maximum value of this set or the overall max value
        max_value = heuristic_vals.max(initial=feature_best_score.score)
        split_options.update_max_value(max_value)
        # also update the score with the given values
        feature_best_score.update_score(max_value)
        # find the indices that are better than the threshold
        # np_nonzero returns a tuple of length 1 (containing the values) when an input 1D array is given,
        # so select the indices (i.e., first item) from the tuple
        index_options = np_nonzero(split_options.can_add(heuristic_vals))[0]
        # for each of the indices, add as an option
        for idx in index_options:
            heuristic_at_split: float = heuristic_vals[idx]
            criteria = fit_finder.build_node_criteria(idx, node_training_data.feature, node_training_data.metric_values)
            split_option = RiskTreeNodePotentialSplit(criteria,
                                                      included_injuries[idx], included_hours[idx],
                                                      included_training_injuries[idx],
                                                      fit_finder.get_number_included_metrics(idx), num_metric_options, heuristic_at_split)
            split_options.add(split_option)

    @staticmethod
    def _calculate_gini_index(included_injuries: ndarray, included_hours: ndarray, total_injuries: float, total_hours: float) -> ndarray:
        # calculate the gini coefficient
        return np_abs(included_injuries / total_injuries - included_hours / total_hours)

    @staticmethod
    def _calculate_chi_squared_heuristic(injury_lists: Iterable[ndarray], hours_lists: Iterable[ndarray], probability_threshold=0.02):
        # injury lists is a list (length m) of n-length ndarrays
        # hours lists is a list (length m) of n-length ndarrays
        # runs a chi-squared test on the actual injuries vs the expected injuries (if had the average injury rate)
        # first, create the injury and hour matrices
        # each column corresponds to one way that the data can be split, so summing injuries and hours makes sense
        #   The sum of injuries / hours represents the total for the data partition being analyzed
        # each row represents data within a given cluster
        # for example, if thresholding data from column A on A <1, A < 2, and A < 3,
        #   all of the "true" criteria would go in one row,
        #   whereas all of the "false" would go in another row
        padding_hours = 1000
        padding_injuries = 1
        injury_mat = np_maximum(np_vstack(injury_lists), padding_injuries)
        hours_mat = np_maximum(np_vstack(hours_lists), padding_hours)
        # next, calculate the average risk across all groups for each row
        # keep the dimensions so that can multiply later with hours to get the total dimensions
        avg_risk_mat = (injury_mat.sum(axis=0, keepdims=True) / hours_mat.sum(axis=0, keepdims=True))
        expected_injuries_mat = hours_mat * avg_risk_mat
        # finally, submit to the chisquare test, and get the p-values (of the groups being the same)
        # note that 1 means that there is no difference between the groups,
        # and 0 means that there is definitely a difference between the groups
        chi2result = chisquare(injury_mat, expected_injuries_mat, axis=0)
        pvalues = chi2result.pvalue
        # lastly, calculate the error of the average risk from the group risk
        # this will be equal to the absolute value of the error between the actual injuries of the group
        # and the expected injuries.
        # Note that assumes that all groups have the same number of injuries, which should be the case
        # since will be analyzing from the perspective of a single node on a tree, which during training
        # should always have the same values
        # Since there should be two groups (included and excluded), any "errors" omitted are from transferring from one group to the other
        # This means that the number of averted injuries should be the average (or divided by 2) of the two groups
        # for example, if the expected values for included group is 10 and excluded group is 10,
        #   and the new predicted values for included group is 8 and excluded group is 12,
        # Then the injury error averted will be abs(8 - 10) + abs(12 - 10), or equal to 4.
        # This is because two injuries were removed from the included group, and added to the excluded group, for a total of 4.
        # Therefore, divide the sum of the error by 2 for each column
        # Note that each column represents a unique value for the feature.
        injury_errors_averted: ndarray = np_abs(injury_mat - expected_injuries_mat)
        group_error_averted = injury_errors_averted.sum(axis=0) / 2.
        # set the places where the pvalues are greater than the probability threshold to 0 (i.e., are likely to be the same group)
        group_error_averted[pvalues > probability_threshold] = RiskTreeNodeRequest.HEURISTIC_NOT_MEET_CRITERIA
        # return the splits with the greatest likelihood of being probable, with the greatest difference of injuries between the groups
        return group_error_averted

    @staticmethod
    def _calculate_binomial_split(included_injuries: ndarray, included_hours: ndarray,
                                  excluded_injuries: ndarray, excluded_hours: ndarray, total_injuries: float, total_hours: float) -> ndarray:
        # Act as though the two samples are identified by a binomial distribution, namely that the likelihood of injury per hour
        # (which should always be way less than one, so can use a binomial distribution for it).
        # Want to test that the two samples come from the same underlying distribution (i.e., have the same risk)
        # with the reported samples.
        # Calculate the Z-score, noting that a Z-score further from zero is less likely, and therefore
        # a stronger reason to split
        # make sure that the number of hours is greater than 0, and that the baseline risk is greater than zero
        # note that both sets of hours should be greater than zero,
        # and the total injuries (and thus the total rate) should also be greater than zero
        padding_hours = 1000
        included_padded_hours = np_maximum(included_hours, padding_hours)
        excluded_padded_hours = np_maximum(excluded_hours, padding_hours)
        # calculate the sample rate for both included and excluded samples
        baseline_risk = max(total_injuries, 1) / max(total_hours, padding_hours)
        included_risk = included_injuries / included_padded_hours
        excluded_risk = excluded_injuries / excluded_padded_hours
        return np_abs(included_risk - excluded_risk) / \
               np_sqrt(baseline_risk * (1 - baseline_risk) * (1 / included_padded_hours + 1 / excluded_padded_hours))

    @staticmethod
    def _calculate_z_score_difference(included_injuries: ndarray, included_hours: ndarray,
                                      excluded_injuries: ndarray, excluded_hours: ndarray) -> ndarray:
        # Compare the z scores for the included groups vs the excluded groups, and return the greatest difference
        # note that this is beneficial for negative values, which can occur within Gradient Boosting Algorithm
        # Treat the data as if it comes from a population with a known mean M and standard deviation S with hours H
        # Treat sub-samples of the data as having a mean m over hour H,
        # with a standard deviation equal to the population standard deviation S multiplied by a correction factor (i.e., sqrt(H / h))
        # Note that this is similar to using a different N within the standard deviation calculation
        # This penalizes small samples that are further away from the mean.
        # Then, for each group, calculate the Z-score using the following formula:
        #   Z = (m - M) / (S * sqrt(H / h))
        # Optimization: note that since the population standard deviation is used by all Z-scores,
        # and since are comparing Z-Score differences to once another to find the maximum,
        # then can factor out the population standard deviation to not calculate.
        # This makes the new Z-score = (m - M) * sqrt(h / H)
        # In addition, add padding anywhere that is dividing by a potentially zero value.
        # Also, want Z-scores that are furthest from 0, so take an absolute value of the difference (so larger number is better)
        padding_hours = 1000
        included_padded_hours = np_maximum(included_hours, padding_hours)
        excluded_padded_hours = np_maximum(excluded_hours, padding_hours)
        # calculate the sample rate for both included and excluded samples
        included_risk = included_injuries / included_padded_hours
        excluded_risk = excluded_injuries / excluded_padded_hours
        # formula adapted from the formula for two sample T test
        return np_abs(included_risk - excluded_risk) / np_sqrt(1 / included_padded_hours + 1 / excluded_padded_hours)


class RiskTreeNodePotentialSplitOptions:
    # keeps track of all potential options for a given split
    # only keeps values that are a percentage of the maximum
    def __init__(self, model_specifics: RiskTreeParametersForDataSet, first_value=0):
        self.model_specifics = model_specifics
        self.max_val: float = first_value
        self.threshold_val: float = first_value
        self.potential_splits: SortedList[RiskTreeNodePotentialSplit] = SortedList(lambda s: s.heuristic_at_split, is_ascending=False)

    def can_add(self, heuristic: ndarray | float):
        return heuristic >= self.threshold_val

    def update_max_value(self, potential_max):
        if potential_max > self.max_val:
            # update the threshold
            new_threshold = potential_max * self.model_specifics.model_params.debug_retain_percent_of_max
            self.potential_splits.remove_below(new_threshold)
            # set the values
            self.max_val = potential_max
            self.threshold_val = new_threshold

    def add(self, new_split: RiskTreeNodePotentialSplit):
        if self.can_add(new_split.heuristic_at_split):
            self.update_max_value(new_split.heuristic_at_split)
            # add to the split list
            self.potential_splits.add(new_split)

    def get_relevant_selections(self):
        # picks the selections that are at or above the threshold
        threshold = self.max_val * self.model_specifics.model_params.selection_percent_of_max
        itemsAboveThreshold = self.potential_splits.get_above(threshold)
        # get the total weight
        total_weight = 0
        for s in itemsAboveThreshold:
            total_weight += s.heuristic_at_split
        return itemsAboveThreshold, total_weight

    def get_metrics(self):
        itemsAboveThreshold, total_weight = self.get_relevant_selections()
        if len(itemsAboveThreshold) == 0:
            return {}
        weight_by_feature: EasyDict[str, float] = EasyDict(0)
        for s in itemsAboveThreshold:
            weight_by_feature[s.split_criteria.feature.name] += s.heuristic_at_split
        return {f: v / total_weight for f, v in weight_by_feature.items()}

    def choose(self) -> RiskTreeNodePotentialSplit | None:
        # choose any of the options, weighted by the potential split values
        # if there are no values, then return none
        # if there is only one value, then return the given value
        itemsAboveThreshold, total_weight = self.get_relevant_selections()
        num_splits = len(itemsAboveThreshold)
        if num_splits == 0:
            return None
        elif num_splits == 1:
            return itemsAboveThreshold[0]

        # choose a number up to the total weight
        chosen_weight = total_weight * self.model_specifics.random_generator.random()
        # find the split that contains that weight
        total_weight_after_addition = 0
        for s in itemsAboveThreshold:
            total_weight_after_addition += s.heuristic_at_split
            if chosen_weight <= total_weight_after_addition:
                return s

    def group_by_feature(self):
        # creates a dictionary of each of the features, with the value being a list of all items included for that feature
        # note that since the full list is already sorted, then each feature list will also be sorted
        feature_lookup: dict[str, list[RiskTreeNodePotentialSplit]] = {}
        for s in self.potential_splits.all_items():
            fName = s.split_criteria.feature.name
            if fName in feature_lookup:
                feature_list = feature_lookup[fName]
            else:
                feature_list = []
                feature_lookup[fName] = feature_list
            feature_list.append(s)
        return feature_lookup

    def print_feature_selections(self, desiredName):
        for s in self.potential_splits.all_items():
            fName = s.split_criteria.feature.name
            if fName == desiredName:
                print(s)


class RiskTreeNodePotentialSplit:
    # logs the information from a potential split within a column
    def __init__(self, split_criteria: RiskTreeNodeSplitCriteriaBase,
                 true_split_injuries: float, true_split_hours: float, true_split_training_injuries: float,
                 true_options_remaining: int, all_option_count: int, heuristic_at_split: float):
        self.split_criteria = split_criteria
        self.true_split_hours = convert_to_storable(true_split_hours)
        self.true_split_injuries = true_split_injuries
        self.true_split_training_injuries = convert_to_storable(true_split_training_injuries)
        self.true_options_remaining = int(true_options_remaining)
        self.false_options_remaining = int(all_option_count - true_options_remaining)
        self.heuristic_at_split = heuristic_at_split

    def __str__(self):
        return f"{self.split_criteria.get_where_statement()} ({round(self.heuristic_at_split, 3)})"
