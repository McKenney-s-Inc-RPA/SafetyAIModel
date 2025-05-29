from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from os.path import join as pathJoin
from sklearn.tree import plot_tree, DecisionTreeRegressor, export_text
import matplotlib.pyplot as plt
from overall_helpers import diagnostic_log_message, convert_to_iterable
from decision_tree_extractor import DecisionTreeEnsembleHandler
from json_manager import JsonBase
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict
import shap
# removes warnings that occur with shap about NumbaDeprecationWarning: The nopython keyword argument was not
# supplied to the 'numba.jit' decorator
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


def sort_data_frame(dataSet: DataFrame):
    return dataSet.reindex(sorted(dataSet.columns), axis=1)


class ModelData:
    @classmethod
    def fromFrame(cls, df: DataFrame, used_predictor: str):
        y = df[used_predictor]
        x = df.drop(columns=used_predictor, axis=1)
        return ModelData(x, y, df)

    def __init__(self, x: DataFrame, y: DataFrame, df: DataFrame | None = None):
        self.x: DataFrame = sort_data_frame(x)
        self.y: DataFrame = y
        self.df = df

    def drop_columns(self, cols_to_drop):
        # should only impact the X and df, as the y should remain the same
        self.x = sort_data_frame(self.x.drop(convert_to_iterable(cols_to_drop, str), axis=1, errors="ignore"))
        self.df = sort_data_frame(self.df.drop(convert_to_iterable(cols_to_drop, str), axis=1, errors="ignore"))

    def split_random(self, test_size: float = 0.2, random_state: int | None = None):
        xTrain, xTest, yTrain, yTest, dfTrain, dfTest = train_test_split(self.x, self.y, self.df, test_size=test_size, random_state=random_state)
        return ModelData(xTrain, yTrain, dfTrain), ModelData(xTest, yTest, dfTest)


class TrainingData:
    @classmethod
    def get_data_split_by_time(cls, dataSet: DataFrame, used_predictor: str, year: int, month: int):
        createdData = TrainingData(dataSet, used_predictor)
        createdData.split_by_time(year, month)
        return createdData

    @classmethod
    def get_data_split_random(cls, dataSet: DataFrame, used_predictor: str, test_size: float = 0.2,
                              random_state: int | None = None):
        createdData = TrainingData(dataSet, used_predictor)
        createdData.split_random(test_size, random_state)
        return createdData

    def __init__(self, dataSet: DataFrame, used_predictor: str):
        self.df: DataFrame = sort_data_frame(dataSet)
        self.used_predictor = used_predictor
        self.train_data: ModelData | None = None
        self.test_data: ModelData | None = None

    def split_by_time(self, year: int, month: int):
        yearCol = self.df["Year"]
        monthCol = self.df["Month"]
        train_data = self.df[(yearCol < year) | ((yearCol == year) & (monthCol < month))]
        test_data = self.df[(yearCol > year) | ((yearCol == year) & (monthCol >= month))]
        self.train_data = ModelData.fromFrame(train_data, self.used_predictor)
        self.test_data = ModelData.fromFrame(test_data, self.used_predictor)

    def split_random(self, test_size: float = 0.2, random_state: int | None = None):
        allData = ModelData.fromFrame(self.df, self.used_predictor)
        self.train_data, self.test_data = allData.split_random(test_size, random_state)

    def drop_columns(self, cols_to_drop):
        self.df = self.df.drop(columns=convert_to_iterable(cols_to_drop, str), axis=1, errors="ignore")
        self.train_data.drop_columns(cols_to_drop)
        self.test_data.drop_columns(cols_to_drop)


class TrainedModel(JsonBase):

    @classmethod
    def train(cls, name: str, model, training_data: TrainingData):
        mt = TrainedModel()
        mt.name = name
        mt._model = model
        diagnostic_log_message(f"Training {mt.name}")
        mt._model.fit(training_data.train_data.x, training_data.train_data.y)
        diagnostic_log_message(f"Testing {mt.name}")
        test_data = training_data.test_data
        mt.feature_names = list(test_data.x.columns)
        y_predictions = mt._model.predict(test_data.x)
        mt.r2 = mt._model.score(test_data.x, test_data.y)
        mt.mse = mean_squared_error(test_data.y, y_predictions)
        mt.model = DecisionTreeEnsembleHandler.create_from_model(model, mt.feature_names)
        diagnostic_log_message(mt)
        return mt

    def __init__(self):
        super().__init__()
        self._model: DecisionTreeRegressor | RandomForestRegressor | None = None
        self.model: DecisionTreeEnsembleHandler | None = None
        self.name: str = "Blank Model"
        self.feature_names: List[str] = []
        self.r2 = 0
        self.mse = 0

    def __str__(self):
        modelType = self._model.__class__.__name__
        return f"{self.name} Evaluation on {modelType}\nMSE: {round(self.mse, 5)}\nR-squared:{round(self.r2, 5)}"

    def from_json_after_events(self):
        super().from_json_after_events()
        # build the underlying model
        self._model = self.model.create_model()

    def save_visualizations(self, directory):
        fig = plt.figure(figsize=(55, 15))
        if hasattr(self._model, 'estimators_'):  # Creating the decision tree figures for the random forest
            for i, estimator in enumerate(self._model.estimators_):
                self._save_tree_visual(estimator, directory, f"RandomForest_{self.name}_{i}.png")
        else:  # Creating the decision tree figures for the decision tree
            self._save_tree_visual(self._model, directory, f"DecisionTree_{self.name}.png")

    def _save_tree_visual(self, tree: DecisionTreeRegressor, directory, file_name):
        plot_tree(tree, feature_names=self.feature_names, filled=True, fontsize=12, impurity=True)
        viz_filename = pathJoin(directory, file_name)
        plt.savefig(viz_filename)

    def explain(self, dataset: ModelData):
        explainer = shap.Explainer(self._model)
        shap_values = explainer(dataset.x)
        return shap_values

    # these are adapted from https://mljar.com/blog/extract-rules-decision-tree/
    def _get_tree_as_text(self, tree: DecisionTreeRegressor):
        return export_text(tree, feature_names=self.feature_names)
