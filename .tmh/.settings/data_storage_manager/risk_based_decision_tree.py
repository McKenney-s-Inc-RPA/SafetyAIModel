from __future__ import annotations
from os.path import exists as path_exists, join as pathJoin
from os import mkdir
from pandas import NA as pd_NA, ExcelWriter, concat as pd_concat
from pandas.api.types import is_numeric_dtype
from risk_tree_nodes import *
from risk_tree_parameters import *
from data_collection import DataCollection, DataColumnAggregation
from json_manager import JsonBase
from numpy import zeros as np_zeros, ones as np_ones, \
    int32 as np_int_type, float64 as np_double_type, inf as np_inf, prod as np_prod
from overall_helpers import *
from graphviz import Digraph
from color_interpolator import ColorInterpolator
from textwrap import TextWrapper
from sklearn.model_selection import train_test_split
from risk_data_column import RiskColumnContinuousTemplate
from progress_bar import ProgressBar
from actionable_steps_recommender import ActionableStepRow, build_actions_and_criteria
from file_creator import getNWeekTrainingData, generated_folder
from copy import deepcopy

# decision tree that aggregates data by hours and injuries
# this allows decision trees to predict a meta-indicator (like injury risk) on
# an aggregate level, instead of a row-by-row level.
# note that this also allows for flexibility with testing data, as the injury risk can be multiplied to new data
GRAPH_VIZ_ADDED_TO_PATH = SingletonLatch()


def register_graphviz():
    if GRAPH_VIZ_ADDED_TO_PATH.should_run():
        import os
        os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"


class TreeVisualizerHelper:
    def __init__(self, baseline_risk: float, color_split_nodes: bool = False, criteria_on_edges: bool = False):
        # whether to show the splitting criteria in the node (and label edges as True / false)
        # or to put splitting criteria on the edge labels
        self.criteria_on_edges = criteria_on_edges
        # allow to be a little wider if criteria is in the nodes, since are less than them
        # otherwise, require to be more narrow
        text_width = 20 if self.criteria_on_edges else 30
        self.text_wrapper = TextWrapper(width=text_width)
        self.color_handler = ColorInterpolator(3, 0)
        self.color_split_nodes = color_split_nodes
        self.baseline_risk = baseline_risk

        # use RGB values
        self.color_handler = ColorInterpolator(3, 0)
        # note that 0 risk should be green,
        self.color_handler.register_data_point(0, (0, 125, 0))
        # the baseline risk should be white,
        self.color_handler.register_data_point(baseline_risk, (255, 255, 255))
        # 2x the baseline risk should be yellow
        self.color_handler.register_data_point(2 * baseline_risk, (255, 255, 0))
        # 3x the baseline risk should be orange
        self.color_handler.register_data_point(3 * baseline_risk, (255, 180, 0))
        # 4x and above the baseline risk should be red
        self.color_handler.register_data_point(4 * baseline_risk, (255, 0, 0))
        self.not_included_color = "#b5d4f5"  # corresponds to a light sky blue

    def get_risk_color(self, risk: float, is_leaf: bool, is_incremental_risk=False):
        if is_leaf or self.color_split_nodes:
            # if risk is absolute, then return the risk color
            # otherwise, is relative to the baseline risk, so add the baseline risk
            if is_incremental_risk:
                return self.color_handler.interpolate_rgb_hex(risk + self.baseline_risk)
            else:
                return self.color_handler.interpolate_rgb_hex(risk)
        else:
            # if not colored, then set to a default value (like light blue)
            return self.not_included_color

    def wrap_text(self, given_text):
        return self.join_text(self.split_text_for_wrapping(given_text))

    def split_text_for_wrapping(self, given_text):
        return self.text_wrapper.wrap(given_text)

    def join_text(self, text_parts: List[str]):
        return "\\n".join(text_parts)


class RiskDecisionTree(JsonBase):
    NO_CACHE = -1

    def __init__(self, weight: float = 1, is_incremental=False):
        self.root: RiskTreeBaseNode | None = None
        self.all_nodes: List[RiskTreeBaseNode] = []
        # the weight applied to the prediction value
        # allows the outputs of all trees to be added together
        # weight of the tree when predicting risk
        self.weight = weight
        # whether the risk is absolute or relative to the average baseline
        self.is_incremental = is_incremental
        self._num_leaves = RiskDecisionTree.NO_CACHE
        self._num_nodes = RiskDecisionTree.NO_CACHE
        self._num_splits = RiskDecisionTree.NO_CACHE

    def _reset_cached_values(self):
        self._num_leaves = RiskDecisionTree.NO_CACHE
        self._num_nodes = RiskDecisionTree.NO_CACHE
        self._num_splits = RiskDecisionTree.NO_CACHE

    @classmethod
    def prioritize_json_names(cls, attributes_to_prioritize: List[str]):
        # prioritize the all nodes list first, as this will grab all nodes first, and then use placeholders elsewhere
        attributes_to_prioritize.append("all_nodes")

    @property
    def baseline_risk(self):
        # the baseline risk of this tree, weighted for easy sum usage
        return self.root.injury_risk * self.weight

    @property
    def num_nodes(self):
        if self._num_nodes == RiskDecisionTree.NO_CACHE:
            self._num_nodes = len(self.all_nodes)
        return self._num_nodes

    @property
    def num_leaf_nodes(self):
        if self._num_leaves == RiskDecisionTree.NO_CACHE:
            self._num_leaves = len(self.leaf_nodes)
        return self._num_leaves

    @property
    def leaf_nodes(self) -> List[RiskTreeBaseNode]:
        return [n for n in self.all_nodes if n.is_leaf]

    @property
    def num_split_nodes(self):
        if self._num_splits == RiskDecisionTree.NO_CACHE:
            self._num_splits = self.num_nodes - self.num_leaf_nodes
        return self._num_splits

    def train(self, model_specifics: RiskTreeParametersForDataSet, features: List[RiskDataColumn], data_for_training, data_are_indices=False):
        # trains on a standardized data given
        assert self.num_nodes == 0, "Model has already been trained"
        # create the first request
        root_request = RiskTreeNodeRequest(data_for_training, model_specifics)
        feature_set = RiskDataColumnSet(features)
        root_node = root_request.create_node(model_specifics, feature_set, data_are_indices)
        self.set_root(root_node)

        # if trained using index data, then convert back to their original value
        if data_are_indices:
            for n in self.all_nodes:
                if isinstance(n, RiskTreeSplitNode):
                    col_value_handler = model_specifics.col_unique_handler.feature_unique_lookup[n.feature_name]
                    n.split_criteria.convert_criteria_from_index(col_value_handler)

    def set_root(self, root_node: RiskTreeBaseNode):
        self._reset_cached_values()
        self.root = root_node
        self._log_node(root_node)
        # now that all heights have been calculated, then order from top to bottom, left to right
        # this ordering assures that when saved to JSON, nodes are saved in an order from least to most references
        self.all_nodes.sort(key=lambda x: x.ordering_number)

    def _log_node(self, curr_node: RiskTreeBaseNode):
        # reads the node information into this tree
        # reads from "left" of tree to "right"
        # first, check to see if is a split node
        self._reset_cached_values()
        if isinstance(curr_node, RiskTreeSplitNode):
            # read left node first
            self._log_node(curr_node.left)
            # add this item to the all nodes
            self.all_nodes.append(curr_node)
            # after that, read the right node
            self._log_node(curr_node.right)
        else:
            # means that is a leaf node
            self.all_nodes.append(curr_node)

    def get_as_readable_rules(self):
        # iterate through the leaves from right to left, and print as human readable rules
        # cluster the thresholds together by columns, to reduce data constraints
        return "\n".join([n.get_readable_rule() for n in self.leaf_nodes])

    def predict_risk(self, dataset: DataFrame):
        # predicts the risk for each row within the given dataset
        risk_predicted = np_zeros(len(dataset), dtype=np_double_type)
        # for root node, applies to all rows
        incoming_mask = np_ones(len(dataset), dtype=bool)
        self.root.predict_values(dataset, risk_predicted, incoming_mask)
        return risk_predicted * self.weight

    def explain_risk_contribution(self, dataset: DataFrame, col_lookup: Dict[str, int]):
        # calculates the SHAP values for this tree for a given data set
        # import the needed functions (note that this is done here to reduce upfront cost
        from shap_python import NO_CHILD as SHAP_NO_CHILD, calculate_shap_for_tree_meets_criteria
        # Convert the tree over to a few arrays, with the following structure
        #   - The first item is ALWAYS the root node
        #   - All split nodes come first, and then leaf nodes
        #       - This is so that the data comparison to the split can be a contiguous array
        # The arrays to provide are the following:
        #   - Children left: The index of the child node to the left
        #       - If is a leaf node (i.e., has no child), then will be -1
        #       - Length: all nodes
        #   - Children right: the index of the child node to the right
        #       - If is a leaf node (i.e., has no child), then will be -1
        #       - Length: all nodes
        #   - Features: index of the feature / column used
        #       - Length: Only corresponds to feature split nodes
        #   - Values: risk associated with a given node, from training
        #       - Length: all nodes
        #   - Node sample weights: the hours associated with each node, from training
        #       - Length: all nodes
        #   - Meets Criteria: whether the data point meets the criteria for a given split
        #       - Takes the place of the data and the thresholds within the shap algorithm
        #       - For N data points, will be N rows by (# feature split nodes)
        # First, assign IDs to all of the nodes
        node_index_lookup: Dict[RiskTreeBaseNode, int] = {}
        # first comes the split nodes, ordered by depth
        # after that comes all leaf nodes, ordered by depth
        # create the placeholder arrays
        # left and right children of each node. If no children (i.e., is leaf), then set as no child value
        child_left = np_ones(self.num_nodes, dtype=np_int_type) * SHAP_NO_CHILD
        child_right = np_ones(self.num_nodes, dtype=np_int_type) * SHAP_NO_CHILD
        # the feature indices that are splitting on, only for split node data
        features = np_zeros(self.num_split_nodes, dtype=np_int_type)
        # risk (aka values) of each of the nodes, based on the training data
        values = np_zeros(self.num_nodes, dtype=np_double_type)
        # hours (aka weights) of each of the nodes, based on the training data
        node_sample_weights = np_zeros(self.num_nodes, dtype=np_double_type)
        meets_criteria = np_zeros((len(dataset), self.num_split_nodes), dtype=bool)
        # the output risk contributions
        risk_contributions = np_zeros((len(dataset), len(col_lookup)), dtype=np_double_type)
        tree_depth = 0

        # the first split id will start with the root, and be 0-based
        next_split_id = 0
        # the first leaf ID will be after the last split node, so assign as the number of split nodes
        next_leaf_id = self.num_split_nodes
        # go through the list (starting with the root node), and add to the sets
        # order the split nodes first, so that the features array can be shorter
        node_stack: List[RiskTreeBaseNode] = [self.root]
        split_nodes: List[RiskTreeSplitNode] = []
        while len(node_stack) > 0:
            curr_node = node_stack.pop(0)
            if isinstance(curr_node, RiskTreeSplitNode):
                # since is a split, then add as the next split index, and then append the left and right child for analysis
                assigned_id = next_split_id
                next_split_id += 1
                # add to the features array
                features[assigned_id] = col_lookup[curr_node.split_criteria.feature.name]
                # add whether the data set meets the split criteria
                meets_criteria[:, assigned_id] = curr_node.split_criteria.get_true_mask(dataset)
                # add the left and right children to the stack
                node_stack.append(curr_node.left)
                node_stack.append(curr_node.right)
                # add as a node to check the left and right values for
                split_nodes.append(curr_node)
            else:
                # since is a leaf, then add as the next node
                assigned_id = next_leaf_id
                next_leaf_id += 1
            # add to the lookup
            node_index_lookup[curr_node] = assigned_id
            # add the values and sample weights
            values[assigned_id] = curr_node.injury_risk
            node_sample_weights[assigned_id] = curr_node.num_hours
            # update the tree depth
            if curr_node.depth > tree_depth:
                tree_depth = curr_node.depth

        # next, assign the left and right child ID
        # in addition, check if the data meets the split threshold
        for n in split_nodes:
            node_index = node_index_lookup[n]
            child_left[node_index] = node_index_lookup[n.left]
            child_right[node_index] = node_index_lookup[n.right]
        # iterate through each row of the sample node weights, and perform the shap calculation
        # multiply by the weight, for easy summation
        return calculate_shap_for_tree_meets_criteria(child_left, child_right, features, meets_criteria, values, node_sample_weights,
                                                      risk_contributions, tree_depth) * self.weight

    def save_visual(self, filename, metric_name, injury_name, hours_name, viz_helper: TreeVisualizerHelper):
        g = Digraph()
        # create all of the nodes
        id_lookup = {}
        next_id = 1
        edges: List[Tuple[RiskTreeSplitNode, RiskTreeBaseNode, str]] = []
        for n in self.all_nodes:
            # get the node id, and store within the lookup
            node_id = "n_" + str(next_id)
            next_id += 1
            id_lookup[n] = node_id
            # for each node, show the risk, as well as the number of hours
            # for example Risk: 8.76 (12 injuries in 274K Hours)
            num_injuries_text = round(n.num_injuries, 2)
            risk_text = str(round(n.injury_risk, 1))
            metric_name_text = metric_name
            injury_name_text = injury_name
            if self.is_incremental:
                metric_name_text += " Adj."
                injury_name_text += " Adj."
            node_label_parts = [f"{metric_name_text}: {risk_text}",
                                f"{injury_name_text}: {num_injuries_text}",
                                f"{hours_name}: {n.num_hours_text}"]
            if isinstance(n, RiskTreeSplitNode):
                # get the criteria for splitting
                true_split_explanation = n.explain_split(True)
                if viz_helper.criteria_on_edges:
                    # place the criteria on the edges
                    true_edge_label = viz_helper.wrap_text(true_split_explanation)
                    false_split_explanation = n.explain_split(False)
                    false_edge_label = viz_helper.wrap_text(false_split_explanation)
                else:
                    # place the true criteria on the bottom of the node
                    # create a new line to space out a little
                    true_edge_label = "True"
                    false_edge_label = "False"
                    node_label_parts.append("")
                    node_label_parts += viz_helper.split_text_for_wrapping(true_split_explanation)
                # add the edges to the children
                edges.append((n, n.left, true_edge_label))
                edges.append((n, n.right, false_edge_label))

            # create the node
            g.node(node_id,
                   label=viz_helper.join_text(node_label_parts),
                   shape="box",
                   fillcolor=viz_helper.get_risk_color(n.injury_risk, n.is_leaf, self.is_incremental),
                   style="filled",
                   tooltip=n.get_readable_rule())

        # now, create all of the edges
        for parent, child, label in edges:
            g.edge(id_lookup[parent], id_lookup[child], label=label)

        # lastly, save to the file
        # perform cleanup to remove the temporary graphviz file
        g.render(outfile=filename, cleanup=True)


class RiskColumnListUser(JsonBase):
    def __init__(self, columns: List[RiskDataColumn]):
        self.columns = columns

    def _normalize_data(self, data: DataFrame) -> DataFrame:
        normalized_data = data.copy()
        # next, perform the rounding on each of the columns
        for c in self.columns:
            if c.name in normalized_data.columns:
                data_before_normalize = normalized_data[c.name]
                data_after_normalize: Series = c.normalize(data_before_normalize)
                normalized_data[c.name] = data_after_normalize
                # check the number of unique values, to ensure that doesn't have too many
                unique_values = data_after_normalize.unique()
                if len(unique_values) > 50:
                    print(f"Column '{c.name}' has {len(unique_values)} unique values")
                # replace any string NAs with Other
                if not is_numeric_dtype(data_after_normalize):
                    normalized_data[c.name] = data_after_normalize.fillna("Other")
        return normalized_data


class RiskModelPart(RiskColumnListUser):
    # uses 1 to many decision trees to perform regression on a single data set
    def __init__(self, name: str, injury_column: str, hours_column: str, risk_description: str | None = None,
                 injury_description: str | None = None, columns: List[RiskDataColumn] | None = None):
        super().__init__([] if columns is None else columns)
        self.name = name
        self.risk_description = risk_description if risk_description is not None else "Risk"
        self.injury_description = injury_description if injury_description is not None else "Injuries"
        self.injury_column = injury_column
        self.hours_column = hours_column
        self.trees: List[RiskDecisionTree] = []

    def duplicate_model_without_data(self, new_name):
        dup_model = RiskModelPart(new_name, self.injury_column, self.hours_column, self.risk_description, self.injury_description)
        dup_model.columns += self.columns
        return dup_model

    def reset(self):
        # removes all trained data
        # note that this should just remove the trees
        self.trees.clear()

    @classmethod
    def prioritize_json_names(cls, attributes_to_prioritize: List[str]):
        # prioritize the columns first, so that will parse columns, and then use where needed within the trees
        attributes_to_prioritize.append("columns")

    @property
    def num_trees(self):
        return len(self.trees)

    @property
    def risk_col(self):
        return self.injury_column + "_Risk"

    @property
    def predicted_risk_col(self):
        return self.injury_column + "_Risk_Predicted"

    @property
    def predicted_injury_col(self):
        return self.injury_column + "_Predicted"

    @property
    def shap_predicted_risk_col(self):
        return "Predicted " + self.risk_description

    @property
    def shap_predicted_injury_col(self):
        return "Predicted " + self.injury_description

    def INTERNAL_normalize_data_for_training(self, data: DataFrame, accelerate=True, data_is_normalized=False):
        diagnostic_log_message(f"Preparing to train '{self.name}'")
        # normalize all column data before sending off to be trained (if not already normalized)
        data_for_training: DataFrame = data if data_is_normalized else self._normalize_data(data)
        unique_val_handler = RiskColumnSetUniqueHandler(self.columns, data_for_training)

        # in addition, convert all of the column data to the sorted integer values
        # this will dramatically increase efficiency, as can use better optimizations
        if accelerate:
            prepped_data_for_training: DataFrame = unique_val_handler.convert_to_index_values(data_for_training)
        else:
            prepped_data_for_training = data_for_training

        return data_for_training, prepped_data_for_training, unique_val_handler

    def INTERNAL_create_dataset_model_parameters(self, model_params: RiskDecisionTreeParameters, unique_val_handler: RiskColumnSetUniqueHandler):
        # create parameters for the model and dataset
        return RiskTreeParametersForDataSet(model_params, self.injury_column, self.hours_column, self.columns, unique_val_handler)

    def INTERNAL_train_with_parameters(self, data_for_training: DataFrame, prepped_data_for_training: DataFrame,
                                       unique_val_handler: RiskColumnSetUniqueHandler,
                                       model_params: RiskDecisionTreeParameters, accelerate=True,
                                       gb_boost_callback: Callable[[int, RiskDecisionTreeParameters], None] | None = None,
                                       after_tree_addition: Callable[[int, RiskDecisionTreeParameters], None] | None = None):
        # reset the model, in case there are any straggling trees
        self.reset()
        # create parameters for the model and dataset
        applicable_params = self.INTERNAL_create_dataset_model_parameters(model_params, unique_val_handler)

        # split out the data, and pass to a tree to train
        diagnostic_log_message(f"Training '{self.name}'")
        if model_params.gradient_boost_iterations > 0:
            # if wanting to use gradient boosting, then do so
            # first, create a tree with the baseline risk of the training set
            # this tree is the baseline tree, using the average risk as the springboard for all other trees
            # because of this, use a weight of 1
            base_tree = RiskDecisionTree(1)
            self.trees.append(base_tree)
            actualInjuryColAttr = applicable_params.injury_col
            total_injuries = prepped_data_for_training[actualInjuryColAttr].sum()
            total_hours = prepped_data_for_training[applicable_params.hours_col].sum()
            base_tree.set_root(RiskTreeBaseNode(total_injuries, total_hours, None))
            # change the applicable parameters to train on the error column
            error_injury_column = actualInjuryColAttr + "_Error"
            applicable_params.change_training_injury_col(error_injury_column)
            # now, perform the gradient boosting algorithm until get the desired result
            tree_weight = model_params.gradient_boost_learning_rate / model_params.n_estimators
            for i in range_one_based(model_params.gradient_boost_iterations):
                diagnostic_log_message(f"Training Iteration {i} of gradient boosting on '{self.name}'")
                # predict the risk and injuries of the training set, based on the prior iterations
                # use the normalized data for predicting,
                # then apply to the prepared data set (which will be used for training)
                predicted_risk = self.INTERNAL_predict_from_normalized_data(data_for_training)
                predicted_injuries = calculate_injuries_from_risk(predicted_risk, prepped_data_for_training[applicable_params.hours_col])
                # calculate the injury error from the calculation, and use this to train the next set of trees
                prepped_data_for_training[error_injury_column] = prepped_data_for_training[actualInjuryColAttr] - predicted_injuries
                # train the next set of the random forest
                for n in range_one_based(model_params.n_estimators):
                    new_tree = self._create_and_train_tree(tree_weight, applicable_params, prepped_data_for_training, accelerate, True)
                    diagnostic_log_message(f"Finished training Tree {n} of {model_params.n_estimators} "
                                           f"for Iteration {i} of gradient boosting on '{self.name}' "
                                           f"({new_tree.num_nodes} nodes; {new_tree.num_leaf_nodes} leaves)")
                    # perform the callback to train the tree
                    if after_tree_addition is not None and n < model_params.n_estimators:
                        after_tree_addition(n, model_params)
                # perform the callback if needed
                if gb_boost_callback is not None and i < model_params.gradient_boost_iterations:
                    gb_boost_callback(i, model_params)
        else:
            # if wanting to build a single-layer random forest, then do so
            tree_weight = 1 / model_params.n_estimators
            for n in range_one_based(model_params.n_estimators):
                new_tree = self._create_and_train_tree(tree_weight, applicable_params, prepped_data_for_training, accelerate)
                diagnostic_log_message(f"Finished training Tree {n} of {model_params.n_estimators} for '{self.name}' "
                                       f"({new_tree.num_nodes} nodes; {new_tree.num_leaf_nodes} leaves)")
                # perform the callback after training the tree
                if after_tree_addition is not None and n < model_params.n_estimators:
                    after_tree_addition(n, model_params)
        # diagnostic_log_message(f"Finished training '{self.name}'")
        # applicable_params.weight_by_feature.print_by_value_desc()

    def train(self, data: DataFrame, model_params: RiskDecisionTreeParameters, accelerate=True,
              gbBoostCallback: Callable[[int, RiskDecisionTreeParameters], None] | None = None, data_is_normalized=False):
        data_for_training, prepped_data_for_training, unique_val_handler = self.INTERNAL_normalize_data_for_training(data, accelerate,
                                                                                                                     data_is_normalized)
        self.INTERNAL_train_with_parameters(data_for_training, prepped_data_for_training, unique_val_handler,
                                            model_params, accelerate, gbBoostCallback)

    def _create_and_train_tree(self, tree_weight: float, applicable_params: RiskTreeParametersForDataSet,
                               prepped_data_for_training: DataFrame, data_are_indices: bool, is_incremental=False):
        # create the tree
        new_tree = RiskDecisionTree(tree_weight, is_incremental)
        self.trees.append(new_tree)
        # pick training data
        selected_features, selected_data = applicable_params.select_params_for_training(prepped_data_for_training, dropna=False)
        # train the model
        new_tree.train(applicable_params, selected_features, selected_data, data_are_indices=data_are_indices)
        # return the new tree
        return new_tree

    def profile_training(self, data: DataFrame, model_params: RiskDecisionTreeParameters, file_name="training_runtime.csv", show_dirs=False):
        profile_func_runtime(lambda: self.train(data, model_params), file_name, show_dirs)

    def get_feature_usage(self):
        # iterate through the trees and get the number of times each feature is used within a decision
        feature_usage = {f.name: 0 for f in self.columns}
        self.increment_feature_usage_counts(feature_usage)
        return feature_usage

    def increment_feature_usage_counts(self, feature_usage: Dict[str, int]):
        for t in self.trees:
            for n in t.all_nodes:
                if isinstance(n, RiskTreeSplitNode):
                    prior_usages = feature_usage.setdefault(n.feature_name, 0)
                    feature_usage[n.feature_name] = prior_usages + 1

    def print_feature_usage(self):
        feature_usage = self.get_feature_usage()
        for k, v in sorted(feature_usage.items(), key=lambda x: x[1]):
            print(f"{k}: {v}")

    @property
    def baseline_risk(self):
        return sum([t.baseline_risk for t in self.trees])

    def explain_risk_contribution_as_matrix(self, dataset: DataFrame):
        # calculates the SHAP values for this set of trees for a given data set
        # normalize the data set, but don't remove any rows
        normalized_dataset = self._normalize_data(dataset)
        # first, convert the columns to indices. This will stand in for the column names.
        col_lookup: Dict[str, int] = {c: normalized_dataset.columns.get_loc(c) for c in normalized_dataset.columns}
        risk_contributions = np_zeros((len(normalized_dataset), len(col_lookup)), dtype=np_double_type)
        for i, t in enumerate(self.trees):
            diagnostic_log_message(f"Explaining Tree {i + 1} of {self.num_trees} for '{self.name}'")
            risk_contributions += t.explain_risk_contribution(normalized_dataset, col_lookup)
        diagnostic_log_message(f"Finished explaining '{self.name}'")
        return risk_contributions, col_lookup

    def explain_risk_contribution(self, dataset: DataFrame):
        risk_contributions, col_lookup = self.explain_risk_contribution_as_matrix(dataset)
        # build the output data frame
        return DataFrame({c: risk_contributions[:, i] for c, i in col_lookup.items()})

    def add_predicted_risk_and_injuries(self, dataset: DataFrame):
        # predicts the risk based on the model, and adds a new column
        predicted_risk = self.predict(dataset)
        dataset[self.predicted_risk_col] = predicted_risk
        # predict the number of injuries, based on risk
        dataset[self.predicted_injury_col] = calculate_injuries_from_risk(dataset[self.predicted_risk_col], dataset[self.hours_column])
        return dataset

    def predict(self, dataset: DataFrame):
        # predicts the risk based on the model, and returns the predicted values
        # first, normalize the data, but don't remove any rows with 0 hours
        normalized_data = self._normalize_data(dataset)
        return self.INTERNAL_predict_from_normalized_data(normalized_data)

    def INTERNAL_predict_from_normalized_data(self, normalized_data: DataFrame):
        # set up the prediction
        predicted_risk = np_zeros(len(normalized_data), dtype=np_double_type)
        # next, pass through to each of the trees
        for t in self.trees:
            tree_predicted_risk = t.predict_risk(normalized_data)
            predicted_risk += tree_predicted_risk
        return predicted_risk

    def save_visuals(self, directory, hour_name="Hours"):
        register_graphviz()
        # create the folder, if doesn't exist
        parent_folder = directory
        if not path_exists(parent_folder):
            mkdir(parent_folder)
        # create the visualizer helper, but don't color the split nodes
        viz_helper = TreeVisualizerHelper(self.baseline_risk, color_split_nodes=True, criteria_on_edges=False)

        if self.num_trees == 1:
            # just save as its own file
            file_name = pathJoin(parent_folder, f"{self.name}.svg")
            self.trees[0].save_visual(file_name, self.risk_description, self.injury_description, hour_name, viz_helper)
        else:
            # create the model folder
            model_folder = pathJoin(directory, self.name)
            if not path_exists(model_folder):
                mkdir(model_folder)
            # save within the directory
            for i, t in enumerate(self.trees):
                file_name = pathJoin(model_folder, f"{self.name}_Tree_{i + 1}.svg")
                t.save_visual(file_name, self.risk_description, self.injury_description, hour_name, viz_helper)


class RiskModelAnalysisSlice:
    def __init__(self, relevant_dimensions: List[str] | str, results_file_suffix: str | None = None):
        self.relevant_dimensions = relevant_dimensions
        self.results_file_suffix = results_file_suffix

    @property
    def display_name(self):
        return self.results_file_suffix


class RiskModelAnalyzer:
    # analyzes model results for a given injury / hours column
    # across multiple slices (i.e., dimension or set of dimensions)
    # for multiple data sources (i.e., training and testing data)
    def __init__(self, injury_col: str, hours_col: str, slices: list[RiskModelAnalysisSlice]):
        self.injury_col = injury_col
        self.hours_col = hours_col
        self.slices = slices
        self.normalized_data_sets: dict[str, DataFrame] = {}
        self.analyzers: dict[str, list[RiskModelAnalysisSliceForData]] = {}

    def add_data(self, normalized_data: DataFrame, key_name: str):
        # add the data set
        self.normalized_data_sets[key_name] = normalized_data
        # calculate the baseline risk
        injuries = normalized_data[self.injury_col].values
        hours = normalized_data[self.hours_col].values
        baseline_risk = calculate_risk(injuries.sum(), hours.sum())
        baseline_injuries = calculate_injuries_from_risk(baseline_risk, hours)
        # for each of the slices, precompute the analysis data
        self.analyzers[key_name] = [RiskModelAnalysisSliceForData(s, normalized_data, injuries, baseline_injuries) for s in self.slices]

    def run_tests(self, model: RiskModelPart):
        # create the output
        output = [{"slice": s.display_name} for s in self.slices]
        # for each data set, run the model, and then get the R2 value for each of the slices
        for k, normalized_data in self.normalized_data_sets.items():
            predicted_risk = model.INTERNAL_predict_from_normalized_data(normalized_data)
            hours = normalized_data[self.hours_col]
            predicted_injuries_series: Series = calculate_injuries_from_risk(predicted_risk, hours)
            predicted_injuries = predicted_injuries_series.values
            analyzers = self.analyzers[k]
            for i, v in enumerate(analyzers):
                output[i][k] = round(v.calculate_root_mean_square_error_accuracy(predicted_injuries), 4)
        return output

    def save_results_xlsx(self, xlsx_filename: str, model: RiskModelPart):
        # figure out any of the dimensions that could be analyzed by
        # if only a few dimensions are used, then this could significantly reduce the analysis set
        relevant_unique_dimensions = set()
        for s in self.slices:
            relevant_unique_dimensions.update(s.relevant_dimensions)
        relevant_dimensions = list(relevant_unique_dimensions)

        # next, build out the excel writer
        with ExcelWriter(xlsx_filename) as writer:
            for data_type, analyzers in self.analyzers.items():
                # the normalized data should have all numeric values rounded, but
                # all dimensions (like project ID, employee ID, etc) should still be intact
                normalized_data = self.normalized_data_sets[data_type]
                # create the data collection (create a copy so that doesn't affect the stored list)
                data_result = DataCollection(normalized_data.copy())
                # add all of the dimensions that will want to analyze by
                data_result.add_dimensions(relevant_dimensions)
                # add the hours, and actual injuries
                data_result.add_measures([self.hours_col, self.injury_col])

                # add in the predicted risk and injuries before consolidation (since needs all of the columns to run prediction)
                predicted_risk = model.INTERNAL_predict_from_normalized_data(normalized_data)
                predicted_injury_col = "Predicted " + self.injury_col
                predicted_risk_col = "Predicted Risk"
                predicted_injuries = self.add_injuries_and_risk(data_result, predicted_risk, predicted_injury_col, predicted_risk_col)

                # consolidate the data - keep the dimensions and the measures used
                relevant_columns = relevant_dimensions + [self.hours_col, self.injury_col, predicted_injury_col, predicted_risk_col]
                data_result.keep_data_columns(relevant_columns, True)

                # add the actual risk
                data_result.add_calculated_field("Actual Risk", [self.injury_col, self.hours_col], calculate_risk)

                # calculate the baseline injuries and risk - can do after, since only uses the injury and hours columns
                baseline_risk = calculate_risk(normalized_data[self.injury_col].sum(), normalized_data[self.hours_col].sum())
                baseline_injury_col = "Baseline Injuries"
                self.add_injuries_and_risk(data_result, baseline_risk, baseline_injury_col, "Baseline Risk")

                # for each of the slices, keep only the relevant dimensions
                for sliceAnalysis in analyzers:
                    currSlice = sliceAnalysis.parent_slice
                    slice_data = data_result.copy()
                    slice_data.keep_dimensions(currSlice.relevant_dimensions)
                    # lastly, calculate the R2 contributions for the predicted and baseline columns
                    # pass in the raw injuries (instead of the aggregated ones from the slice),
                    # as the orders could be different between the two aggregated data frames
                    # sliceAnalysis.set_up_r2_contributions(slice_data.df, predicted_injuries)
                    # don't need to reorder the columns since won't be used externally.
                    slice_data.save_data(xlsx_filename, currSlice.results_file_suffix + " " + data_type, excel_writer=writer, order_columns=False)

    def add_injuries_and_risk(self, data_result: DataCollection, risk_values: ndarray | Series | float, injury_name: str, risk_name: str):
        # first, calculate the number of injuries for each of the risk values
        # do this by multiplying by the number of hours
        hours = data_result.df[self.hours_col]
        injuries_series: Series = calculate_injuries_from_risk(risk_values, hours)
        data_result.set_measure_data_column(injury_name, injuries_series)

        # next, add the risk (as a calculated field)
        # add as a calculated field so that if is consolidated, then will update automatically
        data_result.add_calculated_field(risk_name, [injury_name, self.hours_col], calculate_risk)

        return injuries_series


class RiskModelAnalysisSliceForData:
    # Helps analyze a specific slice for a given test for data
    # Particularly helpful when doing parameter sweeps
    # the data, the test, and the slice should always remain the same
    def __init__(self, parent_slice: RiskModelAnalysisSlice,
                 data: DataFrame,
                 injuries: ndarray,
                 baseline_injuries: ndarray):
        self.parent_slice = parent_slice
        # create a copy of the data, and reset the index (so that will be from 0 to N)
        data_copy = data.reset_index(inplace=False, drop=True)
        # the unique values for the data
        # sort to maintain the same order within train and test
        data_by_slice = data_copy.groupby(by=parent_slice.relevant_dimensions, sort=True, dropna=False)
        # the number of slices that are used
        self.num_slice_options = len(data_by_slice.groups)
        # the category that each row from the data falls into
        self.data_slice_indices: ndarray = np_zeros(len(data), dtype=int)
        # go through each group, and set the index of the group to the indices used
        slice_headers = []
        for i, (k, v) in enumerate(data_by_slice.groups.items()):
            if not isinstance(k, tuple):
                k = (k,)
            try:
                slice_headers.append({h: j for h, j in zip(parent_slice.relevant_dimensions, k)})
                self.data_slice_indices[v] = i
            except Exception as e:
                print("Problem with zipping")
        # keep the relevant groups, in the order that they occur
        self.slices = DataFrame(slice_headers)
        # precompute the actual and baseline values, to compute the R2
        self.actual_values = self._get_sum_by_slice(injuries)
        baseline_values = self._get_sum_by_slice(baseline_injuries)
        baseline_r2_values = self._get_squared_errors(baseline_values)
        # calculate the total sum of squares, i.e., sum of squares of actual vs baseline
        self.totalSumOfSquares = sum(baseline_r2_values)
        # lastly, calculate the baseline r2 contributions (for usage later
        self._normalized_baseline_r2_contributions = baseline_r2_values / self.totalSumOfSquares

    def _get_sum_by_slice(self, values: ndarray):
        return np_bin_count(self.data_slice_indices, values, minlength=self.num_slice_options)

    def _get_squared_errors(self, given_values: ndarray):
        return (self.actual_values - given_values) ** 2

    def calculate_R2(self, predicted_values):
        # get the values array
        slice_predicted_values = self._get_sum_by_slice(predicted_values)
        # consolidate by the slice
        # first, calculate the sum of squares of residuals, i.e., sum of squares of actual vs predicted
        predicted_r2_values = self._get_squared_errors(slice_predicted_values)
        sumOfSquaresOfResiduals = sum(predicted_r2_values)
        return 1 - sumOfSquaresOfResiduals / self.totalSumOfSquares

    def calculate_root_mean_square_error_accuracy(self, predicted_values):
        # note that the RMSE is equivalent to:
        #   RMSE_predicted = sqrt(sum(predicted_error_by_group ^ 2) / num_groups)
        #   RMSE_baseline = sqrt(sum(baseline_error_by_group ^ 2) / num_groups)
        #   RMSE_accuracy = 1 - RMSE_predicted / RMSE_baseline
        # Simplifying, this becomes 1 - sqrt(sum(predicted_error_by_group^2) / sum(baseline_error_by_group^2))
        # This is very similar to the R2 calculation above, just with a square root
        # get the values array
        slice_predicted_values = self._get_sum_by_slice(predicted_values)
        # consolidate by the slice
        # first, calculate the sum of squares of residuals, i.e., sum of squares of actual vs predicted
        predicted_r2_values = self._get_squared_errors(slice_predicted_values)
        sumOfSquaresOfResiduals = sum(predicted_r2_values)
        return 1 - (sumOfSquaresOfResiduals / self.totalSumOfSquares) ** 0.5

    def set_up_r2_contributions(self, data: DataFrame, predicted_values, aggregate_injuries=True):
        # first, calculate the predicted values by group / slice
        slice_predicted_values = self._get_sum_by_slice(predicted_values) if aggregate_injuries else predicted_values
        # next, calculate the r2 value contributions for the predicted values, normalizing by the total sum of squares for the baseline
        normalized_predicted_r2_contributions = self._get_squared_errors(slice_predicted_values) / self.totalSumOfSquares
        # lastly, write the columns. The normalized contributions represent the "deviance" for each row, as a percentage of the baseline
        # higher deviance => further from actual => contributes more to a lower R2 value.
        data["Predicted R2 %"] = normalized_predicted_r2_contributions
        data["Baseline R2 %"] = self._normalized_baseline_r2_contributions
        # lastly, subtract the prediction from the baseline percentage
        # since R2 is calculated as 1 - (normalized r2), then the sum of these values will be equivalent to the R2 value
        # This allows users to see exactly what rows / slices contribute the most to the R2 reading.
        data["R2 Contribution %"] = self._normalized_baseline_r2_contributions - normalized_predicted_r2_contributions


class RiskModel(JsonBase):
    # uses multiple risk models (each which perform on their own data set) to build a full regression model
    def __init__(self, name: str):
        self.name = name
        self.parts: List[RiskModelPart] = []

    def create_part(self, data_source: ModelDataSource, name: str, injury_column: str,
                    hours_column: str | None = None, risk_description: str | None = None,
                    injury_description: str | None = None):
        # creates a new part that will work off of a single data set
        used_hours_col = hours_column if hours_column is not None else data_source.hours_column
        new_part = RiskModelPart(name, injury_column, used_hours_col, risk_description, injury_description, data_source.columns)
        self.parts.append(new_part)
        return new_part


class ShapIncrementalExplainer:
    def __init__(self, model: RiskModelPart, data: DataFrame):
        self.model = model
        self.explanations = model.explain_risk_contribution(data)
        self.prior_explanations = np_zeros(len(data))
        self.injury_name = self.model.shap_predicted_injury_col
        self.risk_name = self.model.shap_predicted_risk_col


class ModelDataSource(RiskColumnListUser):
    # houses data from a given data table,
    # carries helpers (like providing training / testing data),
    # and contains metadata (like column information) about the data set
    def __init__(self, data_set: DataCollection, hours_column: str):
        # only retain data where the hours are greater than 0
        super().__init__([])
        hours_values = data_set.df[hours_column]
        self.df: DataFrame = data_set.df.loc[hours_values > 0]
        # dimensions used for testing / training data
        self.relevant_dimensions: List[str] = []
        # ways to slice the data for testing
        self.test_slices: List[RiskModelAnalysisSlice] = []
        self.hours_column: str = hours_column
        # columns that should be removed for training
        self.columns_to_ignore_for_training: set[str] = set()

    def order_by_dimensions(self, given_dims: List[str] | str | None = None, use_common_dimensions=True):
        # orders by the dimensions of the set to give a consistent and predictable result order

        # create the set of the data dimensions
        # default to the common dimensions
        data_dimensions: set[str] = set()
        if use_common_dimensions:
            data_dimensions.update(commonDimensions)
        data_dimensions.update(convert_to_iterable(given_dims, str))
        # get the columns that will be sorting
        sorting_cols = [c for c in self.df.columns if c in data_dimensions]
        # ensure that the columns are sorted, so that the sorting order is predictable
        sorting_cols.sort()
        # sort the values by the column, and then reset the index (so that is always the index & order)
        self.df.sort_values(sorting_cols, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def register_column(self, col_name: str, normalizer: DataNormalizer, is_continuous: bool, min_value=None, max_value=None,
                        can_threshold=True, can_range=True, can_categorize=True, units: str | None = None):
        if col_name not in self.columns_to_ignore_for_training:
            new_col = RiskDataColumn(col_name, normalizer, is_continuous, min_value, max_value, can_threshold, can_range, can_categorize, units)
            self.columns.append(new_col)

    def register_text_column(self, col_names: list[str] | str):
        for c in convert_to_iterable(col_names, str):
            self.register_column(c, DataNormalizer(), False, can_threshold=False, can_range=False, can_categorize=True)

    def register_numeric_discrete_column(self, col_names: list[str] | str, min_value: float | int | None = None,
                                         max_value: float | int | None = None, units: str | None = None, can_categorize=True):
        for c in convert_to_iterable(col_names, str):
            self.register_column(c, DataNormalizer(), False, min_value, max_value,
                                 can_threshold=True, can_range=True, can_categorize=can_categorize, units=units)

    def register_numeric_continuous_column(self, col_names: list[str] | str, normalizer: DataNormalizer,
                                           min_value: float | int | None = None, max_value: float | int | None = None,
                                           units: str | None = None):
        for c in convert_to_iterable(col_names, str):
            self.register_column(c, normalizer, True, min_value, max_value,
                                 can_threshold=True, can_range=True, can_categorize=False, units=units)

    def register_continuous_col_from_template(self, col_names: list[str] | str, col_template: RiskColumnContinuousTemplate):
        self.register_numeric_continuous_column(col_names, col_template.normalizer, col_template.min_value,
                                                col_template.max_value, col_template.units)

    def register_ignore_columns_during_training(self, col_names: list[str] | str):
        for c in convert_to_iterable(col_names, str):
            self.columns_to_ignore_for_training.add(c)
        # remove any columns that are to be ignored
        for i in range(len(self.columns) - 1, -1, -1):
            if self.columns[i].name in self.columns_to_ignore_for_training:
                del self.columns[i]

    def _normalize_data_for_training(self, data: DataFrame, consolidate_data=False) -> Tuple[DataFrame, DataFrame]:
        normalized_data = self._normalize_data(data)
        # some of the data may map to the same value
        # if so, then consolidate (to reduce the number of future rows)
        if consolidate_data:
            # assume that if are consolidating, then any columns without a normalizer set up will be summed together
            grouping_columns = []
            aggregations = {}
            cols_with_normalizers = {c.name for c in self.columns}
            # iterate through each column.
            #   - If has a normalizer, then group by the value.
            #   - Otherwise, if numeric, then sum.
            for c in normalized_data.columns:
                if c in cols_with_normalizers:
                    grouping_columns.append(c)
                elif is_numeric_dtype(normalized_data[c]):
                    aggregations[c] = "sum"
            return normalized_data, group_data_frame(normalized_data, grouping_columns, aggregations)
        return normalized_data, normalized_data

    def split_by_time(self, year: int, month: int):
        year_values = self.df["Year"]
        month_values = self.df["Month"]
        train_data = self.df[(year_values < year) | ((year_values == year) & (month_values < month))]
        test_data = self.df[(year_values > year) | ((year_values == year) & (month_values >= month))]
        return train_data, test_data

    def split_random(self, test_size: float = 0.2, random_state: int | None = None):
        dfTrain, dfTest = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return dfTrain, dfTest

    def split_random_projectwise(self, test_size: float = 0.2, random_state: int | None = None) -> Tuple[DataFrame, DataFrame]:
        # splits up the data in the dataset by projects
        # this allows for data to definitely be shared when training, and may provide a better view into model efficiency than random alone
        # first, get the unique projects
        projectData = self.df[projectCol]
        uniqueProjects = projectData.unique()
        trainProjects, testProjects = train_test_split(uniqueProjects, test_size=test_size, random_state=random_state)
        isTrainProject = projectData.isin(trainProjects)
        dfTrain = self.df.loc[isTrainProject]
        dfTest = self.df.loc[~isTrainProject]
        return dfTrain, dfTest

    def register_test_slice(self, slice_dimensions: List[str] | str, results_file_prefix: str | None = None):
        slice_dim_list = convert_to_iterable(slice_dimensions, str)
        for d in slice_dim_list:
            if d not in self.relevant_dimensions:
                self.relevant_dimensions.append(d)
        self.test_slices.append(RiskModelAnalysisSlice(slice_dim_list, results_file_prefix))

    def train_and_analyze_random(self, model: RiskModelPart, model_params: RiskDecisionTreeParameters,
                                 test_size: float = 0.2, random_state: int | None = None,
                                 xlsx_filename: str | None = None, perform_save=True):
        train_data, test_data = self.split_random(test_size, random_state)
        return self._train_and_analyze(model, model_params, train_data, test_data, xlsx_filename, perform_save)

    def train_and_analyze_random_projectwise(self, model: RiskModelPart, model_params: RiskDecisionTreeParameters,
                                             test_size: float = 0.2, random_state: int | None = None,
                                             xlsx_filename: str | None = None, perform_save=True):
        train_data, test_data = self.split_random_projectwise(test_size, random_state)
        return self._train_and_analyze(model, model_params, train_data, test_data, xlsx_filename, perform_save)

    def train_and_analyze_timewise(self, model: RiskModelPart, model_params: RiskDecisionTreeParameters,
                                   year: int, month: int, xlsx_filename: str | None = None, perform_save=True):
        train_data, test_data = self.split_by_time(year, month)
        return self._train_and_analyze(model, model_params, train_data, test_data, xlsx_filename, perform_save)

    def _train_and_analyze(self, model: RiskModelPart, model_params: RiskDecisionTreeParameters,
                           train_data: DataFrame, test_data: DataFrame,
                           xlsx_filename: str | None = None, perform_save=True):
        analyzer = RiskModelAnalyzer(model.injury_column, model.hours_column, self.test_slices)
        normalized_train_data, consolidated_train_data = self._normalize_data_for_training(train_data)
        analyzer.add_data(normalized_train_data, "train")
        normalized_test_data = self._normalize_data(test_data)
        analyzer.add_data(normalized_test_data, "test")

        # first, train the model
        # provide a callback function that can be used to report back incremental training
        def incremental_training_callback(iter_num: int, mp: RiskDecisionTreeParameters):
            incremental_results = analyzer.run_tests(model)
            print(f"Results for '{model.name}' after {iter_num} training rounds: {incremental_results}")

        # train the model on the data
        model.train(consolidated_train_data, model_params, gbBoostCallback=incremental_training_callback, data_is_normalized=True)

        # save off the results of the test data
        if perform_save:
            if xlsx_filename is None:
                xlsx_filename = pathJoin("Output", model.name + " Results.xlsx")
            analyzer.save_results_xlsx(xlsx_filename, model)

        # run the tests to tell the accuracy of the predictions
        # returns the r2 value of the tests
        r2Results = analyzer.run_tests(model)
        return r2Results

    def perform_parameter_sweep(self, model: RiskModelPart, param_options: dict[str, list[float | int]],
                                train_data: DataFrame = None, test_data: DataFrame = None,
                                data_split_creator: Callable[[], Tuple[DataFrame, DataFrame]] | None = None,
                                iters=10, output_file="Hyperparameter Sweep Results.csv", consolidate_output=False):
        # the data set iterations are the iterations that occur when creating a new data source
        # the param iterations are the iterations that occur for each parameter set
        # so that aren't fine-tuning for a random data set, multiple data set splits should be used
        # each data split should be run with each parameter set, so that know the full values
        # if a data set splitter function is given, then run the iterations using different data sets
        # however, if not, then run with each parameter set for the number of given iterations
        create_new_data_split_each_iter = data_split_creator is not None
        if create_new_data_split_each_iter:
            data_set_iters = iters
            params_iters = 1
        else:
            params_iters = iters
            data_set_iters = 1
        # set up the status bar
        total_tests = np_prod([len(v) for v in param_options.values()]) * iters
        sb = ProgressBar(total_tests)
        # while the status bar is running, temporarily avoid messages
        priorVerboseMode = VERBOSE_MODE
        update_verbose_mode(False)
        # iterate through the parameter combinations, to see what the best option is
        iter_results = []

        for _ in range(data_set_iters):
            param_options_copy = deepcopy(param_options)
            # if creating a data set split, then do so here
            if create_new_data_split_each_iter:
                train_data, test_data = data_split_creator()

            # run the initialization, which will normalize the data (for faster tree execution)
            accelerate = True
            diagnostic_log_message("Preparing for parameter sweep")
            normalized_train_data, consolidated_train_data = self._normalize_data_for_training(train_data)
            data_for_training, prepped_data_for_training, unique_val_handler \
                = model.INTERNAL_normalize_data_for_training(consolidated_train_data, accelerate, data_is_normalized=True)
            analyzer = RiskModelAnalyzer(model.injury_column, model.hours_column, self.test_slices)
            analyzer.add_data(data_for_training, "train")
            analyzer.add_data(self._normalize_data(test_data), "test")
            # calculate the minimum and maximum values for the Gradient Boost algorithm and number of trees
            # this reduces the training cycles by checking values within the training process
            # for example, if training 4 trees, then can get data for training 4, 3, 2, and 1 trees.
            gb_iter_options: set[int] = set()
            gb_iter_name = "gradient_boost_iterations"
            gb_loop_options = [0]  # for looping through in here, as 0 and any other value have different meanings
            if gb_iter_name in param_options_copy:
                gb_iter_options.update(param_options_copy[gb_iter_name])
                # since 0 and N have two separate functionalities, then check if 0 is in the set
                if 0 in gb_iter_options and len(gb_iter_options) > 1:
                    # put both 0 and the max into the options
                    gb_loop_options = [0, max(gb_iter_options)]
                else:
                    # only put the maximum in. Note that if is only 0, then this will put 0 in as the max
                    gb_loop_options = [max(gb_iter_options)]
            gb_learning_rate_options = None
            gb_learning_rate_option_name = "gradient_boost_learning_rate"
            if gb_learning_rate_option_name in param_options_copy:
                gb_learning_rate_options = param_options_copy[gb_learning_rate_option_name]

            num_trees_options: set[int] = set()
            original_tree_options = [1]
            num_trees_name = "n_estimators"
            if num_trees_name in param_options_copy:
                original_tree_options = param_options_copy[num_trees_name]
                num_trees_options.update(param_options_copy[num_trees_name])
                param_options_copy[num_trees_name] = [max(num_trees_options)]

            # iterate through the given gb loop options
            for g in gb_loop_options:
                # set the value
                param_options_copy[gb_iter_name] = [g]
                if g == 0:
                    # means that should run as a regular decision tree
                    # this means that can run the analysis after creating each tree, so set the number of trees to the maximum value
                    param_options_copy[num_trees_name] = [max(original_tree_options)]
                    if gb_learning_rate_options is not None and gb_learning_rate_option_name in param_options_copy:
                        param_options_copy.pop(gb_learning_rate_option_name)

                    def after_tree_creation(iter_num: int, model_params: RiskDecisionTreeParameters):
                        if iter_num in num_trees_options:
                            # change the weighting of each of the trees
                            original_estimators = model_params.n_estimators

                            # change over to the new weighting
                            model_params.n_estimators = iter_num
                            new_weighting = 1 / iter_num
                            for t in model.trees:
                                t.weight = new_weighting

                            # run the analysis
                            self._run_analysis_and_return_report(model, model_params, analyzer, iter_results, sb)

                            # change back to the old weighting
                            model_params.n_estimators = original_estimators
                            original_weighting = 1 / original_estimators
                            for t in model.trees:
                                t.weight = original_weighting

                    # run the training on all options
                    for model_params, instance_params in self._build_out_parameter_options(param_options_copy):
                        for _ in range(params_iters):
                            model.INTERNAL_train_with_parameters(data_for_training, prepped_data_for_training, unique_val_handler,
                                                                 model_params, accelerate, after_tree_addition=after_tree_creation)
                            # run the analysis
                            self._run_analysis_and_return_report(model, model_params, analyzer, iter_results, sb)
                else:
                    # means that are running the gradient boost algorithm, so revert back the number of trees to the original options
                    param_options_copy[num_trees_name] = original_tree_options
                    if gb_learning_rate_options is not None and gb_learning_rate_option_name not in param_options_copy:
                        param_options_copy[gb_learning_rate_option_name] = gb_learning_rate_options

                    # set up the gb results callback
                    def after_gb_iteration(iter_num: int, model_params: RiskDecisionTreeParameters):
                        if iter_num in gb_iter_options:
                            original_num_iterations = model_params.gradient_boost_iterations
                            # change over to the new iteration number
                            model_params.gradient_boost_iterations = iter_num

                            # run the analysis
                            self._run_analysis_and_return_report(model, model_params, analyzer, iter_results, sb)

                            # revert to the original iteration number
                            model_params.gradient_boost_iterations = original_num_iterations

                    # run the training on all options
                    for model_params, instance_params in self._build_out_parameter_options(param_options_copy):
                        for _ in range(params_iters):
                            model.INTERNAL_train_with_parameters(data_for_training, prepped_data_for_training, unique_val_handler,
                                                                 model_params, accelerate, gb_boost_callback=after_gb_iteration)
                            # run the analysis
                            self._run_analysis_and_return_report(model, model_params, analyzer, iter_results, sb)
        # now that is done with progress bar, reinstate verbose mode.
        update_verbose_mode(priorVerboseMode)
        # now that are done, save to a CSV file
        iter_collection = DataCollection(DataFrame(iter_results))
        param_names = list(param_options.keys())
        dim_names = param_names.copy()
        dim_names.append("slice")
        measures = ["test", "train"]
        iter_collection.add_dimensions(dim_names)
        iter_collection.add_measures(measures, aggregation_method=DataColumnAggregation.SUM)
        cols_to_keep = dim_names.copy()
        cols_to_keep += measures
        # don't perform consolidation, as will want to see all values that occur
        iter_collection.keep_data_columns(cols_to_keep, consolidate_output)
        iter_collection.save_data(output_file)
        return iter_results

    def _run_analysis_and_return_report(self, model: RiskModelPart, model_params: RiskDecisionTreeParameters,
                                        analyzer: RiskModelAnalyzer, iter_results: List[dict],
                                        pb: ProgressBar | None = None):
        # run the analysis
        param_values = model_params.__dict__
        incremental_results = analyzer.run_tests(model)
        for res in incremental_results:
            for k, v in param_values.items():
                res[k] = v
            iter_results.append(res)
        if pb is not None:
            pb.update()

    @staticmethod
    def _build_out_parameter_options(param_options: dict[str, list[float | int]]):
        keys = list(param_options.keys())
        num_params = len(keys)
        option_counts = [len(param_options[k]) for k in keys]
        curr_options = [0] * num_params
        more_options_available = True
        while more_options_available:
            # return the applicable parameters
            instance_params = {keys[i]: param_options[keys[i]][curr_options[i]] for i in range(num_params)}
            # build the parameters with the given options
            yield RiskDecisionTreeParameters(**instance_params), instance_params

            # now, shift start adding at the first index
            addition_index = 0
            shift_addition_spot = True
            while shift_addition_spot:
                curr_options[addition_index] += 1
                # if goes above the number of counts, then roll over, and move to the next addition index
                if curr_options[addition_index] == option_counts[addition_index]:
                    curr_options[addition_index] = 0
                    addition_index += 1
                    # if rolling over past the last index, then quit out
                    if addition_index == num_params:
                        more_options_available = False
                        return
                else:
                    shift_addition_spot = False

    def create_shap_insights(self, models: List[RiskModelPart] | RiskModelPart, filename: List[str] | str | None = None,
                             data: DataFrame | None = None, reporting_dimensions: List[str] | None = None, write_risk=False, write_injuries=True,
                             show_values_as_range=False, recommendations_filename: List[str] | str | None = None, save_path="Shap Value Output"):
        # creates the shap insights for each of the columns used within this data source, for each of the given models
        # start by predicting the shap insights for each value
        used_data = data if data is not None else self.df
        # reset the index since the reported data uses the provided data index,
        # and the shap explainer uses a 0-N index
        used_data = used_data.copy()
        used_data.reset_index(drop=True, inplace=True)
        m: RiskModelPart
        # the list of metrics that are being explained.
        # For example, AICR and RIR
        shap_explanations: List[ShapIncrementalExplainer] = [ShapIncrementalExplainer(m, used_data)
                                                             for m in convert_to_iterable(models, RiskModelPart)]
        hours_values = used_data[self.hours_column].values
        used_dims = reporting_dimensions if reporting_dimensions is not None else self.relevant_dimensions
        if self.hours_column not in used_dims:
            used_dims.append(self.hours_column)
        shap_handler = ShapInsightsHandler(shap_explanations, used_dims, self.hours_column, show_values_as_range)
        # first, save off the baseline value
        # reset the index since the reported data uses the provided data index,
        # and the shap explainer uses a 0-N index
        reported_data = shap_handler.build_reported_data(used_data, "Baseline")
        for explainer in shap_explanations:
            if write_injuries:
                # calculate the number of injuries, based only on the baseline risk
                reported_data[explainer.injury_name] = calculate_injuries_from_risk(explainer.model.baseline_risk, hours_values)
            if write_risk:
                reported_data[explainer.risk_name] = explainer.model.baseline_risk

        # figure out the columns that are used by the models, as these will affect the trees
        col_usage_counts = {}
        for m in convert_to_iterable(models, RiskModelPart):
            m.increment_feature_usage_counts(col_usage_counts)
        col_lookup = {c.name: c for c in self.columns if c.name in col_usage_counts}

        # for each column, write the column value, predicted injury contribution for each model, and bounds
        for c in col_lookup.values():
            shap_handler.add_column(used_data, c, write_risk=write_risk, write_injuries=write_injuries)

        if recommendations_filename is not None:
            # now, work on the recommended actions
            # include all dimensions and metric / metric value (and lowerbound / upperbound), so are adequately tracked
            # include the hours, injuries, and recommended items, and include a flag to omit any duplicates
            # (so that hours & injuries are not double counted)
            # next, download the recommended actions
            recommendation_shap_handler = ShapInsightsHandler(shap_explanations, used_dims, self.hours_column, show_values_as_range)
            recommended_steps: list[ActionableStepRow] = build_actions_and_criteria()
            for recommendation in recommended_steps:
                applicable_col_names = recommendation.criteria.get_columns()
                applicable_row_mask = recommendation.criteria.get_dataset_mask(used_data)
                for n in applicable_col_names:
                    if c := TryGetValue(col_lookup, n):
                        reported_data = recommendation_shap_handler.add_column(used_data, c, applicable_row_mask,
                                                                               write_risk=write_risk, write_injuries=write_injuries)
                        reported_data["Recommended Action"] = recommendation.recommended_action.action
                        reported_data["Responsible Role"] = recommendation.recommended_action.role

            # concatenate all data, and save to file
            recommendation_shap_insight_handler = recommendation_shap_handler.generate_output(recommendations_filename, save_path)

        # concatenate all data, and save to file
        shap_insight_handler = shap_handler.generate_output(filename, save_path)

        return shap_insight_handler


class ShapInsightsHandler:
    def __init__(self, shap_explanations: List[ShapIncrementalExplainer],
                 used_dims: list[str], hours_column: str, show_values_as_range=False):
        self.used_dims = used_dims
        self.output_data: List[DataFrame] = []
        self.hours_column = hours_column
        self.show_values_as_range = show_values_as_range
        self.shap_explanations = shap_explanations

    def build_reported_data(self, data_set: DataFrame, metric_name: str) -> DataFrame:
        reported_data = data_set[self.used_dims].copy()
        reported_data["Metric"] = metric_name
        self.output_data.append(reported_data)
        return reported_data

    def get_hours(self, data_set: DataFrame):
        return data_set[self.hours_column].values

    def add_column(self, data_set: DataFrame, given_column: RiskDataColumn, data_mask: Series | None = None,
                   write_risk=False, write_injuries=True):
        if data_mask is not None:
            data_set = data_set.loc[data_mask].copy()
            data_set.reset_index(drop=True, inplace=True)

        # start with the dimensions to report
        reported_data = self.build_reported_data(data_set, given_column.name)
        # save off the column value
        col_val = data_set[given_column.name]
        # save off the normalized data
        display_value, lower_bound, upper_bound = given_column.get_bounds_text(col_val, self.show_values_as_range)
        reported_data["Metric Value"] = display_value
        # also add the lower and upper bound value, if they are relevant
        reported_data["Metric Lowerbound"] = lower_bound
        reported_data["Metric Upperbound"] = upper_bound
        # next, predict the values for each of the models
        hours_values = self.get_hours(data_set)
        for explainer in self.shap_explanations:
            # save off the number of injuries / risk predicted
            predicted_risk = explainer.explanations[given_column.name].values
            if data_mask is not None:
                true_indices = data_mask[data_mask].index
                predicted_risk = predicted_risk[true_indices]
            if write_injuries:
                predicted_injuries = calculate_injuries_from_risk(predicted_risk, hours_values)
                reported_data[explainer.injury_name] = predicted_injuries
            if write_risk:
                reported_data[explainer.risk_name] = predicted_risk
        return reported_data

    def generate_output(self, filename: List[str] | str | None = None, save_path="Shap Value Output"):
        # concatenate all data
        shap_insights: DataFrame = pd_concat(self.output_data, ignore_index=True, axis=0)
        shap_insight_handler = DataCollection(shap_insights, "Shap Insights")
        # at some point, should the data be consolidated, so that only relevant rows are shown?
        # for now, assume that the important dimensions are provided, so data consolidation wouldn't change anything
        for f in convert_to_iterable(filename, str):
            save_file_name = pathJoin(save_path, filename)
            shap_insight_handler.save_data(save_file_name)
        return shap_insight_handler


def build_employee_model_source():
    # pull in the data to use
    from file_creator import employee_folder

    employee_metrics = employee_folder.get_data_source("Tenure vs Injuries").get_data()
    dataSrc = ModelDataSource(employee_metrics, hoursCol)
    hoursNormalizer = RiskColumnContinuousTemplate(DataNormalizer_RoundDown(40), 0, units="Hours")
    weeksNormalizer = RiskColumnContinuousTemplate(DataNormalizer_RoundDown(2), 0, units="Weeks")
    yearsNormalizer = RiskColumnContinuousTemplate(DataNormalizer_RoundDown(0.25), 0, units="Years")
    # set up the months. Don't set up year, since training year wont be applicable to future predictions
    dataSrc.register_numeric_discrete_column(monthCol, 1, 12)
    # include the branch and profit center column. However, don't include project ID, employee ID, or foreman ID
    dataSrc.register_text_column([branchCol, profitCenterTypeCol])
    # set up the employee tenure
    # set up position, position type, branch, and profit center type something that can categorize
    dataSrc.register_text_column(["Position", "Position Type"])
    # round time to 40 hour increments
    dataSrc.register_continuous_col_from_template(["Total Time on Project", "Continuous Time on Project"],
                                                  hoursNormalizer)
    # round weeks to nearest 2 weeks
    dataSrc.register_continuous_col_from_template(["Total Weeks on Project", "Continuous Weeks on Project"],
                                                  weeksNormalizer)
    # round years of service to nearest 3 months
    dataSrc.register_continuous_col_from_template(["Time at McK", "Time in Position Type Since Last Hired", "Time in Position Type",
                                                   "Time Since Last Hired", "Time in Position Since Last Hired", "Time in Position"],
                                                  yearsNormalizer)
    return dataSrc


# To train on relevant data, only use 2019 to present (meaning that can use observations)
default_use_observations = True


def set_up_data_collection_as_model_source(ds: DataCollection, use_observations=default_use_observations):
    # if using observations, then only include 2019 to present
    if use_observations:
        ds.filter_remove_before_year(2019)
    # only include construction data (which will be for ATL and CLT)
    ds.filter(branchCol).is_in(["ATL", "CLT"])
    dataSrc = ModelDataSource(ds, hoursCol)
    # set up the normalizers
    hoursNormalizer = RiskColumnContinuousTemplate(DataNormalizer_Round(4), 0, units="Hours")
    # round total project time to 40 hour increments
    # hoursTenureNormalizer = RiskColumnContinuousTemplate(DataNormalizer_RoundDown(40), 0, units="Hours")
    hoursNormalizerProvider = DataNormalizer_Incrementing_Steps({0: 4, 40: 8, 96: 16, 160: 40, 480: 80, 960: 160}, 0, 2080)
    hoursTenureNormalizer = RiskColumnContinuousTemplate(hoursNormalizerProvider, 0, units="Hours")
    # round total project weeks to 2 weeks
    # weeksNormalizer = RiskColumnContinuousTemplate(DataNormalizer_RoundDown(2), 0, units="Weeks")
    weeksNormalizerProvider = DataNormalizer_Incrementing_Steps({0: 1, 4: 2, 24: 4}, 0, 100)
    weeksNormalizer = RiskColumnContinuousTemplate(weeksNormalizerProvider, 0, units="Weeks")
    # round years of service to nearest 3 months
    yearsNormalizerProvider = DataNormalizer_Incrementing_Steps({0: 0.25, 2: 0.5, 5: 1, 10: 5}, 0, 100)
    yearsNormalizer = RiskColumnContinuousTemplate(yearsNormalizerProvider, 0, units="Years")
    # round prior injuries
    injuryNormalizer = RiskColumnContinuousTemplate(DataNormalizer_RoundDown(0.05), 0, units="Injuries")
    # set up the months. Don't set up year, since training year wont be applicable to future predictions
    # only allow ranges of months, and not individually selected months
    dataSrc.register_numeric_discrete_column(monthCol, 1, 12, can_categorize=False)
    # include the branch and profit center column. However, don't include project ID, employee ID, or foreman ID
    dataSrc.register_text_column([branchCol, profitCenterTypeCol, profitCenterIdCol, craftCol])
    # set up the closed value
    dataSrc.register_numeric_discrete_column("Closed", 0, 1)

    # set up the employee tenure
    # set up position, position type, branch, and profit center type something that can categorize
    # don't include position because of spurious positions, instead relying on position type and / or time in position type
    dataSrc.register_text_column([positionCol, positionTypeCol])
    dataSrc.register_continuous_col_from_template(["FMNHours"], hoursNormalizer)
    dataSrc.register_continuous_col_from_template(["FMNJTDHours", "JTDHours"], hoursTenureNormalizer)
    dataSrc.register_continuous_col_from_template(["JTDWeeksOnProject", "FMNJTDWeeksOnProject"],
                                                  weeksNormalizer)
    dataSrc.register_continuous_col_from_template(["YearsAtMcK", "YearsSinceLastHired", "YearsInPosition", "YearsInPositionSinceLastHired",
                                                   "YearsInPositionType", "YearsInPositionTypeSinceLastHired",
                                                   "FMNYearsAtMcK", "FMNYearsSinceLastHired", "FMNYearsInPosition",
                                                   "FMNYearsInPositionSinceLastHired",
                                                   "FMNYearsInPositionType", "FMNYearsInPositionTypeSinceLastHired"],
                                                  yearsNormalizer)
    # set up the foreman tenure
    # set up position, position type, branch, and profit center type something that can categorize
    dataSrc.register_text_column(["FMNPosition", "FMNPositionType"])

    # set up the data based on observations
    # don't include near miss or traning observations, since injuries are sometimes labeled as these
    if use_observations:
        # since observations started in 2019, will need only from then to present
        smallObsNormalizer = RiskColumnContinuousTemplate(DataNormalizer_RoundInteger(0, 30), units="Obs")
        dataSrc.register_continuous_col_from_template(["# Safety Obs", "TotalNumForemanObs", "PTPs", "# At Risk Obs", "# Training Obs"],
                                                      smallObsNormalizer)

    # set up the # injuries in the past period
    dataSrc.register_continuous_col_from_template(["Recent # Recordables", "Recent # Injuries",
                                                   "Craft Recent # Recordables", "Craft Recent # Injuries",
                                                   "Project Recent # Recordables", "Project Recent # Injuries"], injuryNormalizer)

    positionTypes = ['Apprentice', 'ClassifiedWorker', 'Foreman', 'Helper', 'JourneyMan', 'Technician', 'Tradesman', 'Office', 'Other']

    # set up the craft and project crew mix percentages
    percentNormalizer = RiskColumnContinuousTemplate(DataNormalizer_RoundDown(5), 0, 100, units="%")
    dataSrc.register_continuous_col_from_template([f"Craft {r} %" for r in positionTypes], percentNormalizer)
    dataSrc.register_continuous_col_from_template([f"Project {r} %" for r in positionTypes], percentNormalizer)
    projectHoursNormalizer = RiskColumnContinuousTemplate(DataNormalizer_LargeNumber([1, 2.5, 5], 10, 10000), 0, units="Hours")
    dataSrc.register_continuous_col_from_template([f"Craft{r}Hrs" for r in positionTypes], projectHoursNormalizer)
    dataSrc.register_continuous_col_from_template([f"Project{r}Hrs" for r in positionTypes], projectHoursNormalizer)

    # set up the project actuals
    smallDollarRounding = RiskColumnContinuousTemplate(DataNormalizer_LargeNumber([1, 2.5, 5], 100, 1e6), 0, units="Dollars")
    largeRangeHoursRounding = RiskColumnContinuousTemplate(DataNormalizer_LargeNumber([1, 2.5, 5], 10, 1e6), 0, units="Hours")
    largeDollarRounding = RiskColumnContinuousTemplate(DataNormalizer_LargeNumber([1, 2.5, 5], 100_000, 1e9), 0, units="Dollars")
    dataSrc.register_continuous_col_from_template(["CraftReworkCost", "JTDCraftReworkCost", "ProjectReworkCost", "JTDProjectReworkCost"],
                                                  smallDollarRounding)
    dataSrc.register_continuous_col_from_template(["TotalCraftHours", "JTDCraftActualHours", "TotalProjectHours", "ProjectReworkHours",
                                                   "JTDProjectActualHours", "JTDProjectReworkHours", "JTDCraftReworkHours", "CraftReworkHours"],
                                                  largeRangeHoursRounding)
    # set up the project / craft budgets
    dataSrc.register_continuous_col_from_template(["RevisedCraftHours", "RevisedProjectHours"],
                                                  largeRangeHoursRounding)
    dataSrc.register_continuous_col_from_template(["RevisedCraftCost", "RevisedProjectCost"],
                                                  largeDollarRounding)

    dataSrc.register_text_column(["MasterCustomerName", "BuildingType", "CustomerID", "CustomerName"])
    # remove customer, customer name, and position (can comment out if like to retain)
    dataSrc.register_ignore_columns_during_training(["MasterCustomerName", 'CustomerID', 'CustomerName', positionCol, "Closed"])
    # set up the columns to ignore
    dataSrc.register_ignore_columns_during_training(['Project ID', 'Foreman ID', 'Employee ID', '# Entries', 'Year', "Week"])

    # set up adherence metrics
    overblownPercentNormalizer = RiskColumnContinuousTemplate(DataNormalizer_Round(10, 0, 200), units="%")
    dataSrc.register_continuous_col_from_template(["% Budget TotalCraftHours", "% Budget JTDCraftActualHours",
                                                   "% Budget TotalProjectHours", "% Budget JTDProjectActualHours",
                                                   "% Budget Craft Cost Budget", "% Budget Craft Hours Budget"],
                                                  overblownPercentNormalizer)
    # rework metrics will need to be a lower percentage, so that is not lost
    reworkOverblownPercentNormalizer = RiskColumnContinuousTemplate(DataNormalizer_Round(1, 0, 5), units="%")
    dataSrc.register_continuous_col_from_template(["% Budget ProjectReworkHours", "% Budget JTDProjectReworkHours",
                                                   "% Budget CraftReworkHours", "% Budget JTDCraftReworkHours",
                                                   "% Budget CraftReworkCost", "% Budget JTDCraftReworkCost",
                                                   "% Budget ProjectReworkCost", "% Budget JTDProjectReworkCost"],
                                                  reworkOverblownPercentNormalizer)
    # sort the data frame by dimensions so that will be in a deterministic order
    dataSrc.order_by_dimensions(["MasterCustomerName", "CustomerID", "CustomerName",
                                 positionCol, positionTypeCol, "FMNPosition", "FMNPositionType"])
    return dataSrc


def build_all_data_model_source(use_observations=default_use_observations):
    # read in the data
    ds = getNWeekTrainingData(2014, 8, 6)
    return set_up_data_collection_as_model_source(ds, use_observations)


if __name__ == "__main__":
    tbrt = RiskModelPart.from_json_file("Models/random_forest_time_based.json")
    # tbrt_rir = RiskModelPart.from_json_file("Models/random_forest_rir_time_based.json")

    # modelDataSource = build_employee_model_source()
    modelDataSource = build_all_data_model_source(use_observations=True)
    startTestYear = 2023
    startTestMonth = 6
    # train_data, test_data = modelDataSource.split_by_time(startTestYear, startTestMonth)
    train_data, test_data = modelDataSource.split_random_projectwise(0.25, randomSeed)

    # add the ways to analyze the results
    modelDataSource.register_test_slice([branchCol, profitCenterTypeCol], "Branch PC")
    modelDataSource.register_test_slice([branchCol, profitCenterTypeCol, projectCol], "Project")
    modelDataSource.register_test_slice([projectCol, positionTypeCol], "Project Position")
    modelDataSource.register_test_slice([monthCol, yearCol], "Month")
    modelDataSource.register_test_slice([monthCol, yearCol, positionTypeCol], "Month Position")
    modelDataSource.register_test_slice([positionTypeCol], "Position")
    modelDataSource.register_test_slice([employeeIdCol], "Employee")
    # modelDataSource.register_test_slice([employeeIdCol, monthCol, yearCol], "Employee Month")
    # add any other columns that want to report on
    modelDataSource.relevant_dimensions += [craftCol, foremanIdCol, weekCol]

    # modelDataSource.create_shap_insights([tbrt, tbrt_rir],
    #                                      filename="RF Shap Insights.hyper",
    #                                      data=test_data,
    #                                      recommendations_filename="RF Shap Insights Recommendations.hyper")

    # build the shap insights
    # tree_shap_output = modelDataSource.create_shap_insights([tbrt, tbrt_rir],
    #                                                         "RF Shap Insights.csv", test_data)
    # tree_shap_output.save_data("RF Shap Insights.hyper")

    # create the parameters for the decision tree
    dtParams = RiskDecisionTreeParameters(feature_ratio=1.0, random_state=randomSeed, max_depth=5,
                                          debug_retain_percent_of_max=0)  # , max_depth=6, max_category_combinations=50_000)
    timeBasedEmployeeModel = RiskModelPart("Employee Model", numInjuriesCol, hoursCol, "AICR", "Injuries", modelDataSource.columns)
    rirTimeBasedEmployeeDt = RiskModelPart("Employee RIR", numRecordablesCol, hoursCol, "RIR", "Recordables", modelDataSource.columns)

    # modelDataSource.perform_parameter_sweep(timeBasedEmployeeModel, {
    #     "selection_percent_of_max": [0.85, 0.9, 0.95, 1],
    #     "max_depth": [2, 3, 4, 5, 6]},
    #                                         train_data, test_data, output_file="Output/Hyperparameter Tuning_Chi Heuristic.csv", iters=7)

    # # train the model, and test on certain dimensions
    # single_tree_results = modelDataSource.train_and_analyze_timewise(timeBasedEmployeeModel, dtParams, startTestYear, startTestMonth,
    #                                                                  xlsx_filename="Output/Single Tree Results.xlsx", perform_save=True)
    # diagnostic_log_message(f"Single Tree AICR Results: {single_tree_results}")
    # timeBasedEmployeeModel.save_visuals("Visuals")
    #
    randomProjectBasedModel = timeBasedEmployeeModel.duplicate_model_without_data("Single Tree - Random Projects")
    single_tree_results_rp = modelDataSource.train_and_analyze_random_projectwise(randomProjectBasedModel, dtParams,
                                                                                  test_size=0.25, random_state=randomSeed,
                                                                                  xlsx_filename="Output/Single Tree Results - Random Projects.xlsx",
                                                                                  perform_save=True)
    diagnostic_log_message(f"Single Tree AICR Results (Random Projects): {single_tree_results_rp}")
    randomProjectBasedModel.save_visuals("Visuals")
    randomProjectBasedModel.to_json_file("Models/single_tree_project_based.json")

    # # create the RIR model
    # rir_single_tree_results = modelDataSource.train_and_analyze_timewise(rirTimeBasedEmployeeDt, dtParams, 2022, 6)
    # print(f"Single Tree RIR Results: {rir_single_tree_results}")
    # timeBasedEmployeeModel.save_visuals("Visuals")
    #
    # # save off both shap values
    # tree_shap_output = modelDataSource.create_shap_insights([timeBasedEmployeeModel, rirTimeBasedEmployeeDt], "Shap Insights.csv", test_data)
    # tree_shap_output.save_data("Shap Insights.hyper")

    # build a random forest model
    num_trees = 11
    feature_ratio = 1.0  # 0.9
    data_ratio = 0.9
    max_depth = 4
    selection_threshold = 0.95
    min_injuries_leaf = 20
    rtParams = RiskDecisionTreeParameters(feature_ratio=feature_ratio, n_estimators=num_trees,
                                          max_depth=max_depth, random_state=randomSeed,
                                          data_ratio=data_ratio,
                                          selection_percent_of_max=selection_threshold)  # ,
    # min_injuries_leaf=min_injuries_leaf,
    # terminating_num_injuries_leaf=30)
    rtParamsGB = RiskDecisionTreeParameters(feature_ratio=feature_ratio, n_estimators=3,
                                            max_depth=3, random_state=randomSeed,
                                            gradient_boost_learning_rate=0.15,
                                            data_ratio=data_ratio, gradient_boost_iterations=60,
                                            selection_percent_of_max=selection_threshold)
    rtParamsRIR = RiskDecisionTreeParameters(feature_ratio=feature_ratio, n_estimators=num_trees,
                                             max_depth=max_depth, random_state=randomSeed,
                                             terminating_num_injuries_leaf=5, min_injuries_leaf=4,
                                             selection_percent_of_max=selection_threshold)
    timeBasedEmployeeRt = timeBasedEmployeeModel.duplicate_model_without_data("Random Forest - Timewise")
    randomBasedEmployeeRt = timeBasedEmployeeModel.duplicate_model_without_data("Random Forest - Random")
    randomProjectBasedEmployeeRt = timeBasedEmployeeModel.duplicate_model_without_data("Random Forest - Random Projects")
    # perform a sweep of parameters
    # modelDataSource.perform_parameter_sweep(timeBasedEmployeeRt, {
    #     "feature_ratio": [0.3, 0.5, 0.7, 0.9],
    #     "n_estimators": [3, 5, 9, 13],
    #     "selection_percent_of_max": [0.85, 0.9, 0.95, 1],
    #     "max_depth": [2, 3, 4]},
    #                                         # "gradient_boost_learning_rate": [0.1, 0.2, 0.3],
    #                                         # "gradient_boost_iterations": [1, 3, 5, 7, 9]},
    #                                         train_data, test_data, output_file="Output/RF Hyperparameter Tuning_Chi Heuristic.csv", iters=15)
    # Analysis shows the following:
    # For projects:
    #   - In General, lower feature ratio is better (0.3 > 0.9)
    #   - In General, more depth is better (4 > 2)
    #   - In General, more estimators is better (13 > 3), and appears to level off after about 11 estimators
    #   - Having less than 1 selection percent of max is best, with 0.95 looking pretty good.
    # For departments:
    #   - In General, lower feature ratio is better (0.3 > 0.9)
    #   - In General, less depth is better (2 > 4)
    #   - In General, more estimators is better (13 > 3), and appears to level off after about 9 estimators
    #   - Having max provides the same results, which is better than 0.95. However, 0.95 is still better than 0.85 and 0.9
    # For month / year:
    #   - In General, more depth is better (4 > 2)
    #   - In General, more estimators is better (13 > 3), and appears to level off after about 11 estimators
    #   - In general, middle of the road feature ratio is better (0.5 or 0.7), with 0.7 being preferrable
    #   - Having less than 1 selection percent of max is best, with 0.95 looking pretty good.
    # For Employees:
    #   - In General, less depth is better (2 > 4), with 2 and 3 being equal
    #   - In General, more estimators is better (13 > 3), and >13 may be even better
    #   - Higher feature ratio is good for less depth, and mid-range feature ratio is good for more depth, with 0.7 a good option
    #   - Having less than 1 selection percent of max is best, with 0.95 looking pretty good.
    # For RF:
    # For Projects:
    #   - High N Estimators (11), selection percent of max doesn't change much (0.85), low feature ratio (0.3)
    #   - Max Depth of 2, Data Ratio of 0.5 seems to perform well
    #   - Highest R2 of 0.08
    # For Position type by Month and Year:
    #   - High N Estimators (9 or 11), high feature ratio (0.7 or 0.9), data ratio 0.5-0.9, high depth (4),
    #       selection percent are similar (0.95 may have slight edge)
    #   - Best R2: 0.19, training r2 of 0.35
    # For Employees:
    #   - High N Estimators (11), Feature ratio (0.5 or 0.7), high data ratio (0.9), selection percent similar (0.95 has best)
    #   - Depth of 3 or 4
    #   - Best R2: 0.01, training r2 of 0.08
    # For Branches:
    #   - High N Estimators (11), Selection percent of 0.9 is tightest, low feature ratio (0.1), mid depth (3), high data ratio (0.7)
    #   - Highest R2 of -0.15, training r2 of 0.33
    # For RF (7/19 training):
    #   - For Branches:

    # modelDataSource.perform_parameter_sweep(timeBasedEmployeeRt, {"feature_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    #                                                               "n_estimators": [3, 5, 7, 9, 11],
    #                                                               "max_depth": [2, 3, 4],
    #                                                               "selection_percent_of_max": [0.85, 0.9, 0.95],
    #                                                               "data_ratio": [0.3, 0.5, 0.7, 0.9]},
    #                                         data_split_creator=lambda: modelDataSource.split_random_projectwise(0.25, randomSeed),
    #                                         output_file="Output/RF Hyperparameter Tuning.csv", iters=11)
    # modelDataSource.perform_parameter_sweep(timeBasedEmployeeRt, {"feature_ratio": [0.3, 0.5, 0.7],
    #                                                               "n_estimators": [3, 5, 7, 9],
    #                                                               "max_depth": [2, 3, 4],
    #                                                               "selection_percent_of_max": [0.85, 0.95],
    #                                                               "data_ratio": [0.5, 0.7, 0.9],
    #                                                               "gradient_boost_learning_rate": [0.1, 0.2, 0.3],
    #                                                               "gradient_boost_iterations": [1, 3, 5, 7, 9]},
    #                                         train_data, test_data, output_file="Output/RF-GB Hyperparameter Tuning.csv", iters=11)

    # # set up the model parameters
    # normalized_data, consolidated_data = modelDataSource._normalize_data_for_training(modelDataSource.df, consolidate_data=False)
    # data_for_training, prepped_data_for_training, unique_val_handler \
    #     = timeBasedEmployeeRt.INTERNAL_normalize_data_for_training(consolidated_data, True, data_is_normalized=True)
    # applicable_params = timeBasedEmployeeRt.INTERNAL_create_dataset_model_parameters(rtParams, unique_val_handler)
    # # feature_selection, data_selection = applicable_params.select_params_for_training(prepped_data_for_training)
    # selected_features, selected_data = profile_func_runtime(
    #     lambda: applicable_params.select_params_for_training(prepped_data_for_training, consolidate=True),
    #     "Data selection.csv")
    # new_tree = RiskDecisionTree(1, False)
    # profile_func_runtime(lambda: new_tree.train(applicable_params, selected_features, selected_data, data_are_indices=True), "Tree Training.csv")
    # diagnostic_log_message("Completed tests!")

    timeBasedEmployeeRt_GB = timeBasedEmployeeModel.duplicate_model_without_data("Employee Random Forest GB")
    rt_results_GB = modelDataSource.train_and_analyze_random_projectwise(timeBasedEmployeeRt_GB, rtParamsGB, 0.25, randomSeed, perform_save=False)
    diagnostic_log_message(f"Random Projects GB AICR Results: {rt_results_GB}")

    rt_results = modelDataSource.train_and_analyze_random_projectwise(randomProjectBasedEmployeeRt, rtParams, 0.25, randomSeed, perform_save=True)
    diagnostic_log_message(f"Random Projects RF AICR Results: {rt_results}")
    randomProjectBasedEmployeeRt.save_visuals("Visuals")
    randomProjectBasedEmployeeRt.to_json_file("Models/random_forest_project_based.json")
    #
    # rt_results = modelDataSource.train_and_analyze_timewise(timeBasedEmployeeRt, rtParams, startTestYear, startTestMonth, perform_save=True)
    # diagnostic_log_message(f"Timewise RF AICR Results: {rt_results}")
    # timeBasedEmployeeRt.save_visuals("Visuals")
    #
    # rt_results = modelDataSource.train_and_analyze_random(randomBasedEmployeeRt, rtParams, 0.25, randomSeed, perform_save=True)
    # print(f"Random RF AICR Results: {rt_results}")
    # randomBasedEmployeeRt.save_visuals("Visuals")

    # use a low number for terminating number injuries, since will be doing gradient boosting (i.e., incremental changes to injuries)
    # otherwise, may start with 0 injuries net error, and won't create better estimators.
    # gb_rtParams = RiskDecisionTreeParameters(feature_ratio=0.3, n_estimators=5,
    #                                          max_depth=2, random_state=randomSeed,
    #                                          data_ratio=0.3,
    #                                          selection_percent_of_max=selection_threshold,
    #                                          gradient_boost_learning_rate=0.25,
    #                                          gradient_boost_iterations=15,
    #                                          terminating_num_injuries_leaf=-1)
    # randomProjectBasedEmployeeRtGB = randomProjectBasedEmployeeRt.duplicate_model_without_data("GB Random Forest - Random Projects")
    # gb_rt_results = modelDataSource.train_and_analyze_random_projectwise(randomProjectBasedEmployeeRtGB, gb_rtParams, 0.25,
    #                                                                      randomSeed, perform_save=True)
    # print(f"GB Random Projects RF AICR Results: {gb_rt_results}")
    # randomBasedEmployeeRt.save_visuals("Visuals")

    # timeBasedEmployeeRt.to_json_file("Models/random_forest_time_based.json")
    # # # timeBasedEmployeeRt_GB.profile_training(train_data, rtParamsGB)
    # # rt_results_GB = modelDataSource.train_and_analyze_timewise(timeBasedEmployeeRt_GB, rtParamsGB, 2022, 6, skip_file_save=True)

    # # print(f"Random Forest AICR Results with Gradient Boosting: {rt_results_GB}")
    # # timeBasedEmployeeRt_GB.save_visuals("Visuals")
    # # timeBasedEmployeeRt_GB.to_json_file("Models/random_forest_time_based_gb.json")
    #
    # # tbrt = RiskModelPart.from_json_file("Models/random_forest_time_based.json")
    #
    # # rirTimeBasedEmployeeRt = rirTimeBasedEmployeeDt.duplicate_model_without_data("Employee RIR Random Forest")
    # # rt_rir_results = modelDataSource.train_and_analyze_timewise(rirTimeBasedEmployeeRt, rtParamsRIR, 2023, 1, perform_save=False)
    # # print(f"Random Forest RIR Results: {rt_rir_results}")
    # # rirTimeBasedEmployeeRt.save_visuals("Visuals")
    # # rirTimeBasedEmployeeRt.to_json_file("Models/random_forest_rir_time_based.json")
    # # tbrt_rir = RiskModelPart.from_json_file("Models/random_forest_rir_time_based.json")
    # tree_shap_output = modelDataSource.create_shap_insights([timeBasedEmployeeRt],  # , rirTimeBasedEmployeeRt],
    #                                                         "RF Shap Insights.csv", test_data)
    # tree_shap_output.save_data("Shap Value Output/RF Shap Insights.hyper")

    # need to check FMNJTDHours once normalized - has basically all 9999 and NaN.
