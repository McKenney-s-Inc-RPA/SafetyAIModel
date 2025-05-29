from os.path import join as pathJoin, getmtime as getModifiedTime, exists as pathExists
from datetime import date
from risk_based_decision_tree import RiskModelPart, set_up_data_collection_as_model_source, getNWeekTrainingData, default_use_observations, \
    ModelDataSource
from file_creator import keep_large_projects_and_group_other_by_department, large_project_id_attr
from overall_helpers import projectCol, employeeIdCol, foremanIdCol, craftCol, weekCol, positionTypeCol, monthCol, yearCol, branchCol, \
    profitCenterTypeCol, order_data_frame_columns_alphabetically, randomSeed, numInjuriesCol, hoursCol, calculate_risk, profitCenterIdCol
from data_collection import DataCollection
from pandas import DataFrame
from risk_tree_parameters import RiskDecisionTreeParameters
from risk_data_column import RiskDataColumn

# all dimensions that want to report on for SHAP insights
allPossibleDims = [projectCol, employeeIdCol, foremanIdCol, craftCol, weekCol, positionTypeCol,
                   monthCol, yearCol, branchCol, profitCenterTypeCol, profitCenterIdCol, large_project_id_attr]


# this file provides information about models, like:
#   - The Model ID (used to distinguish between models when multiple models are used
#   - The location where the model is stored
#   - The dimensions that the model was optimized to predict
# This file also provides helpers for:
#   - Training models based on the given dimensions
#   - Providing a list of models to run during ongoing predictions

# class which stores information about the models
class ModelRuntimeInfo:
    def __init__(self, modelId: str, savePath: str, dimsOptimizedFor: list[str] | None):
        self.modelId = modelId
        self.savePath = savePath
        self.dimsOptimizedFor = dimsOptimizedFor
        # find a place for the model to stay
        self._loadedModel: RiskModelPart | None = None

    @property
    def modelIdAndLastModified(self):
        # add the time that the model was last modified, so that can distinguish between updates
        # use a colon to separate from the ID, so that can easily determine the Model ID if needed
        if pathExists(self.savePath):
            # get the modified time
            # this provides the seconds since EPOCH as a float
            modSeconds = getModifiedTime(self.savePath)
            modDate = date.fromtimestamp(modSeconds)
            # write as year month day (like 20241122)
            # note that the month and day will write as two digits, so should always have a length of 8
            modDateStr = modDate.strftime("%Y%m%d")
        else:
            # if the path doesn't exist, then just add on a placeholder
            modDateStr = "00000000"
        return self.modelId + ":" + modDateStr

    def get_model(self) -> RiskModelPart:
        return RiskModelPart.from_json_file(self.savePath)

    def save_model(self, model: RiskModelPart):
        model.to_json_file(self.savePath)
        self._loadedModel = model

    def load_model_if_needed(self):
        if self._loadedModel is None:
            self._loadedModel = self.get_model()

    def _get_model_data_source(self, ds: DataCollection | None = None, use_observations=default_use_observations):
        # if no data set is given, then load in the training data
        if ds is None:
            ds = getNWeekTrainingData(2014, 8, 6)

        # set up the model data source
        modelDataSource = set_up_data_collection_as_model_source(ds, use_observations)

        # register the columns that will use for checking the model accuracy
        if self.dimsOptimizedFor is not None:
            modelDataSource.register_test_slice(self.dimsOptimizedFor, self.modelId)
        else:
            modelDataSource.register_test_slice([branchCol, profitCenterTypeCol], "Branch PC")
            modelDataSource.register_test_slice([projectCol], "Project")
            modelDataSource.register_test_slice([projectCol, positionTypeCol], "Project Classification")
            modelDataSource.register_test_slice([positionTypeCol], "Classification")
            modelDataSource.register_test_slice([employeeIdCol], "Employee")
            # add any other columns that want to report on
            # modelDataSource.relevant_dimensions += [craftCol, foremanIdCol, weekCol]
        return modelDataSource

    def _get_empty_model(self, columns: list[RiskDataColumn]):
        return RiskModelPart(f"{self.modelId} Model", numInjuriesCol, hoursCol, "AICR", "Injuries", columns)

    def perform_sweep(self, param_options: dict[str, list[int | float]], ds: DataCollection | None = None,
                      use_observations=default_use_observations, iters=5):
        # build the data source (including the columns to analyze)
        modelDataSource = self._get_model_data_source(ds, use_observations)

        # build out the model
        timeBasedEmployeeModel = self._get_empty_model(modelDataSource.columns)

        # perform the sweep
        modelDataSource.perform_parameter_sweep(timeBasedEmployeeModel, param_options,
                                                data_split_creator=lambda: modelDataSource.split_random_projectwise(0.25, randomSeed),
                                                output_file=f"Output/RF Hyperparameter Tuning for {self.modelId}.csv", iters=iters)

    def train_and_test_model(self, dtParams: RiskDecisionTreeParameters, ds: DataCollection | None = None,
                             use_observations=default_use_observations, save_model=False, saveAnalysisResults=False):
        # build the data source (including the columns to analyze)
        modelDataSource = self._get_model_data_source(ds, use_observations)

        # build out the model
        newModel = self._get_empty_model(modelDataSource.columns)
        train_test_results = modelDataSource.train_and_analyze_random_projectwise(newModel, dtParams, 0.25, randomSeed,
                                                                                  perform_save=saveAnalysisResults)
        print(train_test_results)

        # if want to save, then do so here
        if save_model:
            self.save_model(newModel)
        return newModel, train_test_results

    def train_model(self, dtParams: RiskDecisionTreeParameters, ds: DataCollection | None = None,
                    use_observations=default_use_observations, save_model=False):
        # build the data source (including the columns to analyze)
        modelDataSource = self._get_model_data_source(ds, use_observations)

        # build out the model, and train it using the whole result set
        newModel = self._get_empty_model(modelDataSource.columns)
        newModel.train(modelDataSource.df, dtParams)

        # if want to save, then do so here
        if save_model:
            self.save_model(newModel)
        return newModel

    def test_then_train_model(self, dtParams: RiskDecisionTreeParameters, ds: DataCollection | None = None,
                              use_observations=default_use_observations, save_model=False, saveAnalysisResults=False):
        # perform training on part of the data, to test how good it will be
        # don't save these results, since will save after train on all of the data
        self.train_and_test_model(dtParams, ds, use_observations=use_observations, save_model=False, saveAnalysisResults=saveAnalysisResults)
        # next, train on all of the data, and save if needed
        self.train_model(dtParams, ds, save_model=save_model)

    @property
    def _has_dims_optimized_for(self):
        return self.dimsOptimizedFor is not None and len(self.dimsOptimizedFor) > 0

    def _get_dims_to_keep(self):
        if self._has_dims_optimized_for:
            return self.dimsOptimizedFor
        else:
            return allPossibleDims

    def generate_injury_predictions(self, currData: DataCollection) -> DataFrame:
        # load the model (if needed)
        self.load_model_if_needed()
        # run the data through the model, and save to a CSV File
        this_weeks_injury_predictions = self._loadedModel.add_predicted_risk_and_injuries(currData.df)
        # add the model type (so multiple models can write to the same data source), and order columns in deterministic order
        this_weeks_injury_predictions["Model_ID"] = self.modelIdAndLastModified
        return order_data_frame_columns_alphabetically(this_weeks_injury_predictions)

    def generate_injury_prediction_breakdown(self, currData: DataCollection, modelAnalyzer: ModelDataSource | None = None) -> DataCollection:
        # load the model (if needed)
        self.load_model_if_needed()
        # set up the model analyzer to know which columns to use
        if modelAnalyzer is None:
            modelAnalyzer = set_up_data_collection_as_model_source(currData)
        importantDims = self._get_dims_to_keep()
        # now, break down the values using the SHAP analysis, and save to a CSV file
        predictedInjuryBreakdown = modelAnalyzer.create_shap_insights(self._loadedModel, reporting_dimensions=importantDims, write_risk=True,
                                                                      write_injuries=True)
        # aggregate by all columns except for the hours, injury, and risk columns
        if self._has_dims_optimized_for:
            # add the measures (to sum when aggregating rows)
            predictedInjuryBreakdown.add_measures([self._loadedModel.hours_column, self._loadedModel.shap_predicted_injury_col])
            # allow to recalculate the risk, based on the aggregated rows
            predictedInjuryBreakdown.add_calculated_field(self._loadedModel.shap_predicted_risk_col,
                                                          [self._loadedModel.shap_predicted_injury_col, self._loadedModel.hours_column],
                                                          calculate_risk)
            measure_cols = [self._loadedModel.hours_column, self._loadedModel.shap_predicted_injury_col, self._loadedModel.shap_predicted_risk_col]
            dim_cols = [c for c in predictedInjuryBreakdown.df.columns if c not in measure_cols]
            predictedInjuryBreakdown.add_dimensions(dim_cols)
            # now, reduce down the result
            predictedInjuryBreakdown.consolidate_data_and_replace()
        # set the model ID
        predictedInjuryBreakdown.set_dimension_data_columns("Model_ID", self.modelIdAndLastModified)
        # set a placeholder for any dimensions that are not used
        for d in allPossibleDims:
            if d not in importantDims:
                predictedInjuryBreakdown.set_dimension_data_columns(d, "*")

        # note that some of the columns may have NaN's instead of blanks
        outputData = predictedInjuryBreakdown.df.fillna("")
        return order_data_frame_columns_alphabetically(outputData)


# create the list of items to work with
deptModel = ModelRuntimeInfo("Department", pathJoin("Models", "department_specialized_random_forest.json"), [branchCol, profitCenterTypeCol])
positionTypeModel = ModelRuntimeInfo("Classification-Branch", pathJoin("Models", "positiontype_branch_specialized_random_forest.json"),
                                     [branchCol, positionTypeCol])
craftModel = ModelRuntimeInfo("Craft-Branch", pathJoin("Models", "craft_branch_specialized_random_forest.json"), [branchCol, craftCol])
generalModel = ModelRuntimeInfo("General", pathJoin("Models", "overall_random_forest.json"), None)
projectModel = ModelRuntimeInfo("Project", pathJoin("Models", "project_specialized_random_forest.json"), [large_project_id_attr])
relevantModels = [
    deptModel,
    positionTypeModel,
    craftModel,
    generalModel,
    projectModel
]


def _get_data_set():
    # have 2014, 2019, 2020, and 2021 loaded
    ds = getNWeekTrainingData(2020, 8, 6)
    # add in the large projects / department groups in
    keep_large_projects_and_group_other_by_department(ds)
    return ds


def perform_parameter_sweep():
    ds = _get_data_set()

    # perform the sweep on all models
    sweepOptions = {"feature_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "n_estimators": [5, 7, 9, 11],
                    "max_depth": [2, 3, 4, 5],
                    "selection_percent_of_max": [0.9],
                    "data_ratio": [0.3, 0.5, 0.7, 0.9],
                    # "gradient_boost_iterations": [1, 5, 9, 13]
                    }

    models_to_sweep = [
        deptModel,
        positionTypeModel,
        craftModel,
        generalModel,
        projectModel
    ]

    for m in models_to_sweep:
        m.perform_sweep(sweepOptions, ds, use_observations=True, iters=8)


def calculate_training_and_testing_score(trainScore: float, testScore: float):
    # want the training and testing to be as high as possible, while being close to one another
    # Therefore, reward high train and test scores, and penalize difference between train and test
    return trainScore + testScore - abs(trainScore - testScore)


def calculate_training_and_testing_metrics_score(metrics: list[dict[str, list[str] | float]]):
    # for each of the metrics, calculate the test and training score, and then sum together
    # note that this will be proprotional to the average score, since should always have the same number of metrics between metrics being compared
    return sum([calculate_training_and_testing_score(v["train"], v["test"]) for v in metrics])


def perform_iterative_model_training(use_observations=default_use_observations, save_model=False, saveAnalysisResults=False, model_iters=9):
    ds = _get_data_set()

    def run_iterative_training(m: ModelRuntimeInfo, p: RiskDecisionTreeParameters):
        bestScore = -1e9
        bestModel = None
        bestMetric = None
        # iteratively run the training of the model, to find the best model that fits both the training and test data sets
        for i in range(model_iters):
            createdModel, testTrainMetrics = m.train_and_test_model(p, ds, use_observations=use_observations,
                                                                    save_model=False, saveAnalysisResults=saveAnalysisResults)
            # figure out if the metrics are better than the old set
            iterScore = calculate_training_and_testing_metrics_score(testTrainMetrics)
            if iterScore > bestScore:
                bestScore = iterScore
                bestModel = createdModel
                bestMetric = testTrainMetrics

        # save the model (if requested)
        if save_model:
            m.save_model(bestModel)

        print(f"Selected model for {m.modelId}: {bestMetric}")

        # lastly, return the model
        return bestModel

    # create basic models for testing
    # test ~15%, train ~20%
    run_iterative_training(deptModel, RiskDecisionTreeParameters(feature_ratio=0.5, n_estimators=5,
                                                                 max_depth=5, random_state=randomSeed,
                                                                 data_ratio=0.9,
                                                                 selection_percent_of_max=0.9))

    # test ~40%, train ~60%
    run_iterative_training(positionTypeModel, RiskDecisionTreeParameters(feature_ratio=0.7, n_estimators=7,
                                                                         max_depth=5, random_state=randomSeed,
                                                                         data_ratio=0.7,
                                                                         selection_percent_of_max=0.9))

    # test ~58%, train ~60%
    run_iterative_training(craftModel, RiskDecisionTreeParameters(feature_ratio=0.9, n_estimators=9,
                                                                  max_depth=5, random_state=randomSeed,
                                                                  data_ratio=0.5,
                                                                  selection_percent_of_max=0.9))

    # test ~15%, train ~20% (average of each of the slices)
    run_iterative_training(generalModel, RiskDecisionTreeParameters(feature_ratio=0.9, n_estimators=13,
                                                                    max_depth=5, random_state=randomSeed,
                                                                    data_ratio=0.5,
                                                                    selection_percent_of_max=0.9))

    run_iterative_training(projectModel, RiskDecisionTreeParameters(feature_ratio=0.9, n_estimators=11,
                                                                    max_depth=5, random_state=randomSeed,
                                                                    data_ratio=0.9,
                                                                    selection_percent_of_max=0.9))


if __name__ == "__main__":
    # perform_parameter_sweep()
    perform_iterative_model_training(save_model=True, saveAnalysisResults=False)
