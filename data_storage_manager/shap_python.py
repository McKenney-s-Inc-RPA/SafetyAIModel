from __future__ import annotations
import numpy as np
import numba

# modified from https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Python%20Version%20of%20Tree%20SHAP.html
# C code here: https://github.com/shap/shap/blob/master/shap/cext/tree_shap.h#L503
NO_CHILD = -1
# Note that the Random Forest Regressor / Shap use 32-bit float precision,
# which sometimes causes discrepancies for data near the thresholds.
# If comparing to SHAP, set to true. Otherwise, set to false.
FALLBACK_TO_32_BIT_FLOAT = False
# since these functions are only used here, can cache to help with calling
CACHE_NUMBA = False


def calculate_shap_for_tree_meets_criteria(children_left, children_right, features, meets_criteria, values, node_sample_weight, phi, tree_depth):
    max_depth = tree_depth + 2
    s = (max_depth * (max_depth + 1)) // 2
    feature_indexes = np.zeros(s, dtype=np.int32)
    zero_fractions = np.zeros(s, dtype=np.float64)
    one_fractions = np.zeros(s, dtype=np.float64)
    pweights = np.zeros(s, dtype=np.float64)
    for i in range(meets_criteria.shape[0]):
        tree_shap_recursive_meets_criteria(
            children_left, children_right, features,
            meets_criteria[i, :], values, node_sample_weight,
            phi[i, :], 0, 0, feature_indexes, zero_fractions, one_fractions, pweights,
            1, 1, -1, 1
        )
    return phi


class Tree:
    def __init__(self, tree):
        if str(type(tree)).endswith("'sklearn.tree._tree.Tree'>"):
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.features = tree.feature.astype(np.int32)
            self.thresholds = tree.threshold.astype(np.float64)
            self.values = tree.value[:, 0, 0].copy()  # assume only a single output for now
            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

            # we recompute the expectations to make sure they follow the SHAP logic
            self.max_depth = self.compute_expectations(0)

    def compute_expectations(self, i, depth=0):
        if self.children_right[i] == NO_CHILD:
            # since is a leaf node, then don't need to change the value
            return 0
        else:
            li = self.children_left[i]
            ri = self.children_right[i]
            depth_left = self.compute_expectations(li, depth + 1)
            depth_right = self.compute_expectations(ri, depth + 1)
            left_weight = self.node_sample_weight[li]
            right_weight = self.node_sample_weight[ri]
            v = (left_weight * self.values[li] + right_weight * self.values[ri]) / (left_weight + right_weight)
            self.values[i] = v
            return max(depth_left, depth_right) + 1


# extend our decision path with a fraction of one and zero extensions
@numba.jit(
    numba.types.void(
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.int32,
        numba.types.float64,
        numba.types.float64,
        numba.types.int32
    ), nopython=True, nogil=True, cache=CACHE_NUMBA
)
def extend_path(feature_indexes, zero_fractions, one_fractions, pweights,
                unique_depth, zero_fraction, one_fraction, feature_index):
    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    pweights[unique_depth] = 1 if unique_depth == 0 else 0

    for i in range(unique_depth - 1, -1, -1):
        pweights[i + 1] += one_fraction * pweights[i] * (i + 1) / (unique_depth + 1)
        pweights[i] = zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1)


# undo a previous extension of the decision path
@numba.jit(
    numba.types.void(
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.int32,
        numba.types.int32
    ), nopython=True, nogil=True, cache=CACHE_NUMBA
)
def unwind_path(feature_indexes, zero_fractions, one_fractions, pweights,
                unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0:
            tmp = pweights[i]
            pweights[i] = next_one_portion * (unique_depth + 1) / ((i + 1) * one_fraction)
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i) / (unique_depth + 1)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i + 1]
        zero_fractions[i] = zero_fractions[i + 1]
        one_fractions[i] = one_fractions[i + 1]


# determine what the total permutation weight would be if
# we unwound a previous extension in the decision path
@numba.jit(
    numba.types.float64(
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.int32,
        numba.types.int32
    ),
    nopython=True, nogil=True, cache=CACHE_NUMBA
)
def unwound_path_sum(zero_fractions, one_fractions, pweights, unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]
    total = 0

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0:
            tmp = next_one_portion / ((i + 1) * one_fraction)
            total += tmp
            next_one_portion = pweights[i] - tmp * zero_fraction * (unique_depth - i)
        else:
            total += pweights[i] / (zero_fraction * (unique_depth - i))

    return total * (unique_depth + 1)


# recursive computation of SHAP values for a decision tree
@numba.jit(
    numba.types.void(
        numba.types.int32[:],
        numba.types.int32[:],
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.int64,
        numba.types.int64,
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64,
        numba.types.float64,
        numba.types.int64,
        numba.types.float64,
    ),
    nopython=True, nogil=True, cache=CACHE_NUMBA
)
def tree_shap_recursive(children_left, children_right, features, thresholds, values, node_sample_weight,
                        x, phi, node_index, unique_depth, parent_feature_indexes,
                        parent_zero_fractions, parent_one_fractions, parent_pweights, parent_zero_fraction,
                        parent_one_fraction, parent_feature_index, condition_fraction):
    # stop if we have no weight coming down to us
    if condition_fraction == 0:
        return

    # extend the unique path
    feature_indexes = parent_feature_indexes[unique_depth + 1:]
    feature_indexes[:unique_depth + 1] = parent_feature_indexes[:unique_depth + 1]
    zero_fractions = parent_zero_fractions[unique_depth + 1:]
    zero_fractions[:unique_depth + 1] = parent_zero_fractions[:unique_depth + 1]
    one_fractions = parent_one_fractions[unique_depth + 1:]
    one_fractions[:unique_depth + 1] = parent_one_fractions[:unique_depth + 1]
    pweights = parent_pweights[unique_depth + 1:]
    pweights[:unique_depth + 1] = parent_pweights[:unique_depth + 1]

    extend_path(
        feature_indexes, zero_fractions, one_fractions, pweights,
        unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index
    )

    split_index = features[node_index]

    # leaf node
    if children_right[node_index] == NO_CHILD:
        for i in range(1, unique_depth + 1):
            w = unwound_path_sum(zero_fractions, one_fractions, pweights, unique_depth, i)
            phi[feature_indexes[i]] += w * (one_fractions[i] - zero_fractions[i]) * values[node_index] * condition_fraction

    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        cleft = children_left[node_index]
        cright = children_right[node_index]
        dataForFeature = x[split_index]
        thresholdForNode = thresholds[node_index]
        differenceFromThreshold = dataForFeature - thresholdForNode

        if differenceFromThreshold <= 0:
            hot_index = cleft
            cold_index = cright
        else:
            hot_index = cright
            cold_index = cleft

        if FALLBACK_TO_32_BIT_FLOAT and abs(differenceFromThreshold) < 0.0001 and differenceFromThreshold != 0:
            dataForFeature32Bit = np.core.single(dataForFeature)
            thresholdForNode32Bit = np.core.single(thresholdForNode)
            if dataForFeature32Bit <= thresholdForNode32Bit:
                hot_index = cleft
                cold_index = cright
            else:
                hot_index = cright
                cold_index = cleft
            pass

        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1
        incoming_one_fraction = 1

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        while path_index <= unique_depth:
            if feature_indexes[path_index] == split_index:
                break
            path_index += 1

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind_path(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index)
            unique_depth -= 1

        tree_shap_recursive(
            children_left, children_right, features, thresholds, values, node_sample_weight,
            x, phi, hot_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights, hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, condition_fraction
        )

        tree_shap_recursive(
            children_left, children_right, features, thresholds, values, node_sample_weight,
            x, phi, cold_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights, cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, condition_fraction
        )


# recursive computation of SHAP values for a decision tree, passing in if the data meets the criteria
# instead of the data and the criteria threshold
@numba.jit(
    numba.types.void(
        numba.types.int32[:],
        numba.types.int32[:],
        numba.types.int32[:],
        numba.types.boolean[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.int64,
        numba.types.int64,
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64,
        numba.types.float64,
        numba.types.int64,
        numba.types.float64,
    ),
    nopython=True, nogil=True, cache=CACHE_NUMBA
)
def tree_shap_recursive_meets_criteria(children_left, children_right, features, meets_criteria, values, node_sample_weight,
                                       phi, node_index, unique_depth, parent_feature_indexes,
                                       parent_zero_fractions, parent_one_fractions, parent_pweights, parent_zero_fraction,
                                       parent_one_fraction, parent_feature_index, condition_fraction):
    # stop if we have no weight coming down to us
    if condition_fraction == 0:
        return

    # extend the unique path
    feature_indexes = parent_feature_indexes[unique_depth + 1:]
    feature_indexes[:unique_depth + 1] = parent_feature_indexes[:unique_depth + 1]
    zero_fractions = parent_zero_fractions[unique_depth + 1:]
    zero_fractions[:unique_depth + 1] = parent_zero_fractions[:unique_depth + 1]
    one_fractions = parent_one_fractions[unique_depth + 1:]
    one_fractions[:unique_depth + 1] = parent_one_fractions[:unique_depth + 1]
    pweights = parent_pweights[unique_depth + 1:]
    pweights[:unique_depth + 1] = parent_pweights[:unique_depth + 1]

    extend_path(
        feature_indexes, zero_fractions, one_fractions, pweights,
        unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index
    )

    split_index = features[node_index]

    # leaf node
    if children_right[node_index] == NO_CHILD:
        for i in range(1, unique_depth + 1):
            w = unwound_path_sum(zero_fractions, one_fractions, pweights, unique_depth, i)
            phi[feature_indexes[i]] += w * (one_fractions[i] - zero_fractions[i]) * values[node_index] * condition_fraction

    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        cleft = children_left[node_index]
        cright = children_right[node_index]
        if meets_criteria[node_index]:
            hot_index = cleft
            cold_index = cright
        else:
            hot_index = cright
            cold_index = cleft

        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1
        incoming_one_fraction = 1

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        while path_index <= unique_depth:
            if feature_indexes[path_index] == split_index:
                break
            path_index += 1

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind_path(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index)
            unique_depth -= 1

        tree_shap_recursive_meets_criteria(
            children_left, children_right, features, meets_criteria, values, node_sample_weight,
            phi, hot_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights, hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, condition_fraction
        )

        tree_shap_recursive_meets_criteria(
            children_left, children_right, features, meets_criteria, values, node_sample_weight,
            phi, cold_index, unique_depth + 1,
            feature_indexes, zero_fractions, one_fractions, pweights, cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, condition_fraction
        )


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    import time
    import xgboost
    import shap
    from sklearn.ensemble import RandomForestRegressor


    class TreeExplainer:
        def __init__(self, model: RandomForestRegressor):
            self.trees = [Tree(e.tree_) for e in model.estimators_]
            # Preallocate space for the unique path data
            max_depth = np.max([t.max_depth for t in self.trees]) + 2
            s = (max_depth * (max_depth + 1)) // 2
            self._arr_size = s
            self.feature_indexes = np.zeros(s, dtype=np.int32)
            self.zero_fractions = np.zeros(s, dtype=np.float64)
            self.one_fractions = np.zeros(s, dtype=np.float64)
            self.pweights = np.zeros(s, dtype=np.float64)
            self.baseline_value = sum([t.values[0] for t in self.trees]) / len(self.trees)

        def shap_values(self, X):
            # convert dataframes
            if str(type(X)).endswith("pandas.core.series.Series'>"):
                X = X.values
            elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
                X = X.values

            assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
            assert len(X.shape) == 2, "Instance must have 2 dimensions!"

            # means that are explaining an entire dataset
            phi = np.zeros(X.shape)
            # iterate through the rows
            for i in range(X.shape[0]):
                for t in self.trees:
                    tree_shap_recursive(
                        t.children_left, t.children_right, t.features,
                        t.thresholds, t.values, t.node_sample_weight,
                        X[i, :], phi[i, :], 0, 0, self.feature_indexes, self.zero_fractions, self.one_fractions, self.pweights,
                        1, 1, -1, 1
                    )
            phi /= len(self.trees)
            return phi

        def shap_values_meets_criteria_approach(self, X):
            phi = np.zeros(X.shape, dtype=np.float64)
            for t in self.trees:
                # calculate if the data meets the criteria of each split
                num_nodes = len(t.children_left)
                meets_criteria = np.zeros((X.shape[0], num_nodes), dtype=bool)
                for i in range(num_nodes):
                    f = t.features[i]
                    if f != -1:
                        # means that actually splits, then check the data
                        meets_criteria[:, i] = X[:, f] <= t.thresholds[i]
                # next, run the shap tree method
                calculate_shap_for_tree_meets_criteria(t.children_left, t.children_right, t.features, meets_criteria,
                                                       t.values, t.node_sample_weight, phi, t.max_depth)
            phi /= len(self.trees)
            return phi


    housing = fetch_california_housing()
    X = housing["data"]
    y = housing["target"]
    num_trees = 5
    model_depth = 6

    print("Fitting the random forest model")
    model = RandomForestRegressor(n_estimators=num_trees, max_depth=model_depth, random_state=42)
    model.fit(X, y)

    # causes a problem with 5 trees of depth 6 on random state 42.
    rows_to_explain = np.array([1416, 1539, 3126])
    only_explain_subset = True
    if only_explain_subset:
        x_to_explain = X[rows_to_explain, :]
    else:
        x_to_explain = X

    print("Explaining random forest model with built in shap explainer")
    model_explainer = shap.TreeExplainer(model)

    print("Explaining random forest model")
    ex = TreeExplainer(model)
    shap_values_1 = ex.shap_values(x_to_explain)
    shap_values_2 = ex.shap_values_meets_criteria_approach(x_to_explain)
    shap_values_expected = model_explainer.shap_values(x_to_explain)
    start = time.time()

    print(f"Time for explaining random forest: {time.time() - start}")
    error_from_expected = abs(shap_values_expected - shap_values_1)
    row_error_from_expected = error_from_expected.sum(axis=1)
    rows_with_high_error = row_error_from_expected > 0.05
    if rows_with_high_error.sum() > 0:
        problematic_rows = x_to_explain[rows_with_high_error, :]
        predicted_values = model.predict(problematic_rows)
        problematic_expected = model_explainer.shap_values(problematic_rows)
        problematic_calc = ex.shap_values(problematic_rows)
        error_on_expected = ex.baseline_value + problematic_expected.sum(axis=1) - predicted_values
        error_on_calc = ex.baseline_value + problematic_calc.sum(axis=1) - predicted_values
        if (abs(error_on_expected) > 0.00001).any() or (abs(error_on_calc) > 0.00001).any():
            pass
    error_from_expected = row_error_from_expected.sum()
    print(f"Error from expected: {error_from_expected}")

    # print("Fitting XGBoost")
    # bst = xgboost.train({"learning_rate": 0.01, "max_depth": model_depth}, xgboost.DMatrix(X, label=y), num_trees)
    #
    # print("Explaining based on XGBoost")
    # start = time.time()
    # shap_values = bst.predict(xgboost.DMatrix(X), pred_contribs=True)
    # print(f"Time for explaining XGBoost: {time.time() - start}")
