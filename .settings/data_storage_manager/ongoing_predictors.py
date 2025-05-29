from file_creator import createOngoingRollingWindowData
from risk_based_decision_tree import RiskModelPart, set_up_data_collection_as_model_source
from os.path import exists as path_exists, join as path_join
from os import mkdir
from overall_helpers import projectCol, employeeIdCol, foremanIdCol, craftCol, weekCol, positionTypeCol, monthCol, yearCol, branchCol, \
    profitCenterTypeCol, order_data_frame_columns_alphabetically
from model_providers import ModelRuntimeInfo, relevantModels
from pandas import DataFrame, concat as pd_concat


def calculate_current_predictions():
    # gather the last N weeks worth of data to create an "average" week
    # i.e., if 6 weeks used and 60K hours worked, then the "average" week will contain 10K hours.
    # Then, run the model to predict the injuries over the next week.
    # However, note that the "prediction" applies the average week over the next 3 weeks,
    # but that is assumed to be reflective of this upcoming week.
    # Both hours and injuries are scaled for an "average" week so that AICR / RIR comparisons work properly.

    # set up the folder
    predictionsFolderName = "Current Week Predictions"
    predictionsFolderPath = path_join(".", predictionsFolderName)
    if not path_exists(predictionsFolderPath):
        mkdir(predictionsFolderPath)

    # get the current data
    currData = createOngoingRollingWindowData(continue_with_sample=True)

    # set up the model analyzer to know which columns to use
    modelAnalyzer = set_up_data_collection_as_model_source(currData)

    # iterate through the various models, and band together
    injury_predictions_set: list[DataFrame] = []
    prediction_breakdown_set: list[DataFrame] = []

    for m in relevantModels:
        # run the data through the model to get injury predictions
        model_injury_predictions = m.generate_injury_predictions(currData)
        injury_predictions_set.append(model_injury_predictions)

        # next, generate a breakdown of contributions of each of the columns
        model_prediction_breakdown = m.generate_injury_prediction_breakdown(currData, modelAnalyzer)
        prediction_breakdown_set.append(model_prediction_breakdown)

    this_weeks_injury_predictions = pd_concat(injury_predictions_set, ignore_index=True, axis=0)
    this_weeks_injury_predictions.to_csv(path_join(predictionsFolderPath, "current_injury_predictions.csv"), index=False)

    predictedInjuryBreakdown = pd_concat(prediction_breakdown_set, ignore_index=True, axis=0)
    predictedInjuryBreakdown.to_csv(path_join(predictionsFolderPath, "current_injury_predictions_breakdown.csv"), index=False)

    # # load in the model
    # loadedModel: RiskModelPart = RiskModelPart.from_json_file("Models/single_tree_project_based.json")
    #
    # # run the data through the model, and save to a CSV File
    # this_weeks_injury_predictions = loadedModel.add_predicted_risk_and_injuries(currData.df)
    # # add the model type (so multiple models can write to the same data source), and order columns in deterministic order
    # this_weeks_injury_predictions["Model_ID"] = "General"
    # this_weeks_injury_predictions = order_data_frame_columns_alphabetically(this_weeks_injury_predictions)
    # this_weeks_injury_predictions.to_csv(path_join(predictionsFolderPath, "current_injury_predictions.csv"), index=False)
    #
    # # set up the model analyzer to know which columns to use
    # modelAnalyzer = set_up_data_collection_as_model_source(currData)
    # # now, break down the values using the SHAP analysis, and save to a CSV file
    # importantDims = [projectCol, employeeIdCol, foremanIdCol, craftCol, weekCol, positionTypeCol, monthCol, yearCol, branchCol, profitCenterTypeCol]
    # predictedInjuryBreakdown = modelAnalyzer.create_shap_insights(loadedModel, reporting_dimensions=importantDims, write_risk=True,
    #                                                               write_injuries=True)
    # predictedInjuryBreakdown.set_dimension_data_columns("Model_ID", "General")
    # predictedInjuryBreakdown.save_data(path_join(predictionsFolderPath, "current_injury_predictions_breakdown.csv"))

    # return the predictions and breakdown
    return this_weeks_injury_predictions, predictedInjuryBreakdown


if __name__ == "__main__":
    # run the current predictions, which will write to the data files.
    calculate_current_predictions()
