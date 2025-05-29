from pandas import DataFrame, concat as pd_concat
from overall_helpers import convert_to_iterable
from data_table import DataTable, DataCollection
from data_columns import DataColumnAggregation
from math import comb as math_combination
from itertools import combinations as itertools_combinations


def calculate_shapley_contributions(data: DataFrame, independent_cols: list[str], dependent_cols: list[str]):
    # calculate the shapley contributions for each of the values
    # assumes that each row is a unique combination of the independent column values
    # for each column combination, group the data, and set up the values
    col_agg = {d: DataColumnAggregation.AVG.value for d in dependent_cols}
    # set up an output array for each of the tests
    shap_results = {c: DataFrame(0, index=data.index, columns=dependent_cols) for c in independent_cols}

    # go through each of the combinations
    independent_col_set = set(independent_cols)
    num_independent_cols = len(independent_cols)
    size_all_minus_one = len(shap_results) - 1
    # assume that the score without any of the columns is zero, instead of the average.
    # otherwise, will need to add the average back in
    for n in range(1, num_independent_cols + 1):
        # calculate the weightings
        # note that this is the player averager and the combination
        as_prior_weighting = (1 / n / math_combination(size_all_minus_one, n)) if n < num_independent_cols else 0
        as_posterior_weighting = 1 / n / math_combination(size_all_minus_one, n - 1)
        for curr_comb in itertools_combinations(independent_cols, n):
            included_cols = list(curr_comb)
            # the Shapley value is calculated as:
            #   1/p * sum( (v(S U m) - v(S)) / combination(p-1, size(S) ) for S in combinations of the columns
            # for:
            #   p = number of columns
            #   S = selection of columns at each step, each is a unique combination of columns
            #   U = union operator
            #   m = column being added
            #   v(S) = average value of the dependent variable using the S columns
            #   combination = number of combinations
            #   size = number of columns in S
            # Note that the expected value of this node v(S) will have been the posterior for another calculation,
            # and the prior for all of the other combinations that extend from this
            # first, calculate the value prior to adding any columns
            # note that this is the average value for each of the groups
            # calculate the average value for groups, grouped by the columns in the current set
            # calculate the groups
            avg_group_results = data.groupby(included_cols).agg(col_agg)
            # join to the data frame to put the relevant average on each row
            join_with_independent_cols = data[included_cols].merge(avg_group_results, how="left", on=included_cols).set_index(data.index)
            avg_vals = join_with_independent_cols[dependent_cols]

            # this value will be added to all columns that this extends from (i.e., where is S U m)
            # and will be subtracted from all columns that extend from this (i.e., where is S)
            # subtract from the columns which extend this, which will occur for all columns which are not included within this set
            if as_prior_weighting > 0:
                as_prior = avg_vals * as_prior_weighting
                excluded_cols = independent_col_set.difference(included_cols)
                for col_to_add in excluded_cols:
                    shap_results[col_to_add] -= as_prior

            # add to all columns which extend to this
            as_posterior = avg_vals * as_posterior_weighting
            for added_col in included_cols:
                shap_results[added_col] += as_posterior

    # now that have built out all of the shapley values, add in the column name and value, and return as a large data frame
    output_data: list[DataFrame] = []
    for col, shap_vals in shap_results.items():
        # add in the corresponding variable and value
        shap_vals["Variable"] = col
        shap_vals["Value"] = data[col]
        # rename the dependent columns
        for d in dependent_cols:
            shap_vals[d + " Contribution"] = shap_vals[d]
        # add in the data columns
        for c in data.columns:
            shap_vals[c] = data[c]
        output_data.append(shap_vals)

    shap_insights: DataFrame = pd_concat(output_data, ignore_index=True, axis=0)
    return shap_insights


def return_columns_in_data_table(dt: DataTable, possible_cols: list[str] | str | None):
    return [c for c in convert_to_iterable(possible_cols, str) if c in dt.headers]


def calculate_shapley_contributions_for_file(file_name: str, partition_cols: list[str] | str | None,
                                             independent_cols: list[str], dependent_cols: list[str] | str):
    # calculates the shapley contributions for each of the dependent columns
    # for each of the independent columns, within each partition set

    # first, pull in the data to a data table
    # calculate the average dependent column value for each of the dimension sets
    dc = DataTable(file_name, auto_populate=False)
    partition_col_list = dc.return_columns_in_data_table(partition_cols)
    independent_col_list = dc.return_columns_in_data_table(independent_cols)
    dependent_col_list = dc.return_columns_in_data_table(dependent_cols)
    dc.add_dimensions(partition_col_list)
    dc.add_dimensions(independent_col_list)
    dc.add_measures(dependent_col_list, aggregation_method=DataColumnAggregation.AVG)
    dc.consolidate_data_and_replace()

    if len(partition_col_list) > 0:
        # for each partition (as defined by the partition columns) calculate the shap contributions
        relevant_cols = independent_col_list + dependent_col_list
        grouped_by_partition = dc.df.groupby(partition_cols)
        output_data: list[DataFrame] = []
        for g, idx in grouped_by_partition.groups.items():
            # get the rows within the partition
            relevant_data = dc.df.loc[idx, relevant_cols]
            partition_shap_vals = calculate_shapley_contributions(relevant_data, independent_col_list, dependent_col_list)
            # set the values of each of the partition columns
            if isinstance(g, tuple):
                for p, v in zip(partition_col_list, g):
                    partition_shap_vals[p] = v
            else:
                partition_shap_vals[partition_col_list[0]] = g
            output_data.append(partition_shap_vals)
        shap_vals = pd_concat(output_data, ignore_index=True, axis=0)
    else:
        # calculate the shap contributions for the full table
        shap_vals = calculate_shapley_contributions(dc.df, independent_col_list, dependent_col_list)

    return shap_vals


if __name__ == "__main__":
    s = calculate_shapley_contributions_for_file("Output/RF Hyperparameter Tuning_Binomial Heuristic.csv", "slice",
                                                 ["n_estimators", "max_depth", "feature_ratio", "data_ratio",
                                                  "gradient_boost_iterations", "gradient_boost_learning_rate"],
                                                 ["train", "test"])
    dc = DataCollection(s)
    dc.save_data("RF Hyperparameter Tuning Analysis.hyper")
