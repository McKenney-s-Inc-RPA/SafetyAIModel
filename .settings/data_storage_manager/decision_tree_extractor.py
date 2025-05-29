from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from json_manager import JsonBase, json_converter
from render_helpers import *
import shap

# import information about the models into the json converter
dt_json_converter = json_converter.register(DecisionTreeRegressor)
rf_json_converter = json_converter.register(RandomForestRegressor)


# extracts data from a decision tree input multiple formats
# adapted from https://mljar.com/blog/extract-rules-decision-tree/


class FeatureBounds:
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.lower_bound = None
        self.upper_bound = None
        self.value_contribution = 0

    def log_lower_bound(self, value):
        # moves the bound upwards
        if self.lower_bound is None or value > self.lower_bound:
            self.lower_bound = value

    def log_upper_bound(self, value):
        # moves the bound downwards
        if self.upper_bound is None or value < self.upper_bound:
            self.upper_bound = value

    def adjust_value(self, value_contribution):
        self.value_contribution += value_contribution

    def get_text(self):
        if self.lower_bound is not None and self.upper_bound is not None:
            return f"{self.feature_name} between {str_with_sig_figs(self.lower_bound)} and " \
                   f"{str_with_sig_figs(self.upper_bound)}"
        elif self.lower_bound is not None:
            return f"{self.feature_name} > {str_with_sig_figs(self.lower_bound)}"
        elif self.upper_bound is not None:
            return f"{self.feature_name} <= {str_with_sig_figs(self.upper_bound)}"
        else:
            raise ValueError("No bounds given")


class FeatureBoundsSet:
    def __init__(self):
        self.feature_bounds: Dict[str, FeatureBounds] = {}
        self.base_value_contribution = 0

    def get_feature(self, feature_name: str) -> FeatureBounds:
        if feature_name in self.feature_bounds:
            return self.feature_bounds[feature_name]
        else:
            new_feature = FeatureBounds(feature_name)
            self.feature_bounds[feature_name] = new_feature
            return new_feature

    def get_conditions_text(self):
        return " and ".join([self.feature_bounds[k].get_text() for k in sorted(self.feature_bounds.keys())])


class SampleValue:
    def __init__(self, value, num_samples):
        self.value = value
        self.num_samples = num_samples

    def __add__(self, other: SampleValue):
        # assume that the other is also a sample value
        total_samples = self.num_samples + other.num_samples
        weighted_value = (self.value * self.num_samples + other.value * other.num_samples) / total_samples
        return SampleValue(total_samples, weighted_value)

    def __str__(self):
        return f"{str_with_sig_figs(self.value)} (based on {self.num_samples} samples)"


class TreeBaseNode(JsonBase):
    lowest_depth = 0

    def __init__(self, parent: TreeSplitNode | None):
        self.parent = parent
        if self.parent is None:
            self.depth = TreeBaseNode.lowest_depth
            # default an empty node to being left, for sorting purposes
        else:
            self.depth = self.parent.depth + 1
        self.height = 0
        self._ordering_number = -1
        self._sample_value: SampleValue | None = None

    @property
    def is_left(self):
        if self.parent is None:
            # default an empty node to being left, for sorting purposes
            return True
        else:
            return self.parent.left == self

    @property
    def node_value(self):
        return self.node_value_and_number_samples.value

    @property
    @abstractmethod
    def is_leaf(self) -> bool:
        ...

    @abstractmethod
    def render(self, render_style: CodeStyle):
        ...

    @property
    def node_value_and_number_samples(self):
        if self._sample_value is None:
            self._sample_value = self._calculate_node_value_and_number_samples()
        return self._sample_value

    @abstractmethod
    def _calculate_node_value_and_number_samples(self) -> SampleValue:
        ...

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


class TreeNodeRequest:
    # used when building the representation from a sci-kit learn decision tree
    def __init__(self, node_id, parent: TreeSplitNode | None, is_left: bool):
        self.node_id = node_id
        self.parent = parent
        self.is_left = is_left


class TreeSplitNode(TreeBaseNode):
    def __init__(self, feature, threshold, parent: TreeSplitNode | None):
        super().__init__(parent)
        self.left: TreeBaseNode | None = None
        self.right: TreeBaseNode | None = None
        self.feature = feature
        self.threshold = threshold

    def __str__(self):
        return f"Split Node on '{self.feature}' with value {self.threshold} " \
               f"(based on {self.node_value_and_number_samples.num_samples})"

    @property
    def is_leaf(self) -> bool:
        return False

    def _calculate_node_value_and_number_samples(self) -> SampleValue:
        return self.left.node_value_and_number_samples + self.right.node_value_and_number_samples

    def render(self, render_style: CodeStyle):
        left_rendering = self.left.render(render_style)
        right_rendering = self.right.render(render_style)
        return render_style.build_full_if_statement(self.depth, self.feature, self.threshold, left_rendering,
                                                    right_rendering)


class TreeLeaf(TreeBaseNode):
    def __init__(self, value: float, num_samples: int, parent: TreeSplitNode | None):
        super().__init__(parent)
        self.value = value
        self.num_samples = num_samples
        self._sample_value = SampleValue(value, num_samples)

    def __str__(self):
        return f"Leaf Node with value {self.value} (based on {self.num_samples})"

    @property
    def is_leaf(self) -> bool:
        return True

    def _calculate_node_value_and_number_samples(self) -> SampleValue:
        return self._sample_value

    def render(self, render_style: CodeStyle):
        return render_style.build_value(self.depth, self._sample_value.value)

    def get_conditions(self) -> FeatureBoundsSet:
        condition_info = FeatureBoundsSet()
        ancestor = self.parent
        is_less_than_threshold = self.is_left
        while ancestor is not None:
            feature_bounds = condition_info.get_feature(ancestor.feature)
            if is_less_than_threshold:
                feature_bounds.log_upper_bound(ancestor.threshold)
            else:
                feature_bounds.log_lower_bound(ancestor.threshold)

            # move up a level
            # if the node is on the right of the split, then is less than the split threshold
            is_less_than_threshold = ancestor.is_left
            ancestor = ancestor.parent
        return condition_info


class DecisionTreeHandler(JsonBase):
    def __init__(self):
        super().__init__()
        # adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        self.all_nodes: List[TreeBaseNode] = []
        self.leaf_nodes: List[TreeLeaf] = []
        self.root_node: TreeBaseNode | None = None
        self.max_depth = 0
        self.max_features = 0
        self.num_features = 0
        self.num_outputs = 0

    @classmethod
    def prioritize_json_names(cls, attributes_to_prioritize: List[str]):
        # prioritize the all nodes list first, as this will grab all nodes first, and then use placeholders elsewhere
        attributes_to_prioritize.append("all_nodes")

    @classmethod
    def create_from_tree(cls, tree: DecisionTreeRegressor, feature_names):
        # adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        dt = DecisionTreeHandler()
        tree_metadata = tree.tree_
        dt.max_features = tree.max_features
        dt.num_features = tree.n_features_in_
        dt.num_outputs = tree.n_outputs_
        # start with the root node id (0)
        stack: List[TreeNodeRequest] = [TreeNodeRequest(0, None, False)]
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            curr_request = stack.pop()

            # If the left and right child of a node is not the same we have a split node
            left_child_id = tree_metadata.children_left[curr_request.node_id]
            right_child_id = tree_metadata.children_right[curr_request.node_id]
            is_split_node = left_child_id != right_child_id
            # If a split node, append left and right children and depth to `stack` so we can loop through them
            curr_node: TreeBaseNode
            if is_split_node:
                split_node = TreeSplitNode(feature_names[tree_metadata.feature[curr_request.node_id]],
                                           tree_metadata.threshold[curr_request.node_id],
                                           curr_request.parent)
                stack.append(TreeNodeRequest(left_child_id, split_node, True))
                stack.append(TreeNodeRequest(right_child_id, split_node, False))
                curr_node = split_node
            else:
                leaf_node = TreeLeaf(tree_metadata.value[curr_request.node_id][0][0],
                                     int(tree_metadata.n_node_samples[curr_request.node_id]),
                                     curr_request.parent)
                dt.leaf_nodes.append(leaf_node)
                curr_node = leaf_node
            # add to the all nodes list
            dt.all_nodes.append(curr_node)

            # attach to the parent
            if curr_request.parent is not None:
                if curr_request.is_left:
                    curr_request.parent.left = curr_node
                else:
                    curr_request.parent.right = curr_node
            else:
                dt.root_node = curr_node

        # calculate the maximum depth
        dt.max_depth = max([n.depth for n in dt.leaf_nodes])

        # now that all nodes have been added and traversed, then calculate the height
        # each node's height will be equivalent to the max depth minus the node's depth
        # explained differently, it is the height above the smallest node in the tree
        # note that a leaf node could have a height greater than 0, if it doesn't have the maximum depth
        for n in dt.all_nodes:
            n.height = dt.max_depth - n.depth

        # now that all heights have been calculated, then order from top to bottom, left to right
        dt.all_nodes.sort(key=lambda x: x.ordering_number)
        dt.leaf_nodes.sort(key=lambda x: x.ordering_number)
        return dt

    def get_as_text(self):
        return self.root_node.render(PRINT_RENDER_STYLE)

    def get_as_python_rules(self):
        # write each of the rules, in order
        return self.root_node.render(PYTHON_RENDER_STYLE)

    def get_as_readable_rules(self):
        # iterate through the leaves from right to left, and print as human readable rules
        # cluster the thresholds together by columns, to reduce data constraints
        readable_rules = []
        for n in self.leaf_nodes:
            condition_info = n.get_conditions()
            condition_text = condition_info.get_conditions_text()
            readable_rules.append(f"When {condition_text} then value = {n.node_value_and_number_samples}")
        return "\n".join(readable_rules)

    def create_decision_tree_model(self):
        return self.__dict__


class DecisionTreeEnsembleHandler(JsonBase):
    # could contain information for 1 to many decision trees
    json_trees_attr = "trees"

    def __init__(self):
        super().__init__()
        self.trees: List[DecisionTreeHandler] = []
        self._explainer = None
        self.model_data = {}

    @classmethod
    def create_from_model(cls, model: DecisionTreeRegressor | RandomForestRegressor, feature_names):
        mh = DecisionTreeEnsembleHandler()
        if hasattr(model, 'estimators_'):
            # Import all trees from the random forest
            for i, estimator in enumerate(model.estimators_):
                mh.trees.append(DecisionTreeHandler.create_from_tree(estimator, feature_names))
        else:
            # Create the decision tree from the model
            mh.trees.append(DecisionTreeHandler.create_from_tree(model, feature_names))

        # set up the explainer
        mh._explainer = shap.Explainer(model)
        return mh

    def get_as_text(self):
        return self._print_using_function(DecisionTreeHandler.get_as_text)

    def get_as_python_rules(self):
        return self._print_using_function(DecisionTreeHandler.get_as_python_rules)

    def get_as_readable_rules(self):
        return self._print_using_function(DecisionTreeHandler.get_as_readable_rules)

    def _print_using_function(self, str_func: Callable[[DecisionTreeHandler], str]):
        curr_tree: DecisionTreeHandler
        # call the function on each of the trees, annotate with the tree number, and then return
        return "\n".join([f"Tree {i}:\n{str_func(curr_tree)}" for i, curr_tree in enumerate(self.trees)])

    def create_model(self) -> DecisionTreeRegressor | RandomForestRegressor:
        pass
