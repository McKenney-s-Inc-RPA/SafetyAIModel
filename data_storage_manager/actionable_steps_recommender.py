from smartsheet import Smartsheet
from smartsheet.sheets import Sheets
from smartsheet.models.sheet import Sheet as SmartsheetSheet
from smartsheet.models.cell import Cell as SmartsheetCell
from smartsheet.models.row import Row as SmartsheetRow
from smartsheet.models.column import Column as SmartsheetColumn
from logical_conditions import *

api_key = {
    'access_token': '3zolofrfk5ww2plarjdo5lku8y',
    'user_agent': 'McK Energy Services',
    'sheet_id': 3490948323430276,
    'sheet_hyperlink': 'https://app.smartsheet.com/sheets/M53hc5HCQvf89Pq9VQ6627m32pJh2PjXQ976H7m1',
}

"""Smartsheet logic

Handles pushing data to Smartsheet.

"""


class RecommendedAction:
    def __init__(self, action: str, role: str):
        self.action = action
        self.role = role


class ActionableStepRow:
    def __init__(self):
        self.recommended_action: RecommendedAction | None = None
        self.criteria: ConditionBase | None = None
        self.parent: ActionableStepRow | None = None
        self.children: list[ActionableStepRow] = []

    @property
    def is_parent(self):
        return len(self.children) > 0

    @property
    def does_not_have_parent(self):
        return self.parent is None

    @property
    def is_criteria(self):
        return self.criteria is not None

    @property
    def is_recommended_action(self):
        return self.recommended_action is not None

    def to_text(self):
        if self.is_criteria and self.is_recommended_action:
            return f"When {self.criteria.get_text()}, {self.recommended_action.role} should {self.recommended_action.action}"
        elif self.is_criteria:
            return f"When {self.criteria.get_text()}"
        elif self.is_recommended_action:
            return f"{self.recommended_action.role} should {self.recommended_action.action}"
        else:
            return "Empty"

    def __str__(self):
        return self.to_text()


class ActionsAndCriteriaSet:
    def __init__(self):
        self.sub_items: list[ActionsAndCriteria] = []

    def add_input_rows(self, input_rows: list[ActionableStepRow]):
        # find the rows which do not have a parent, and therefore are the top of the stack
        for r in input_rows:
            if r.does_not_have_parent:
                rowItem = ActionsAndCriteria(self)
                rowItem.add_row(r)

    def create_rows(self):
        for item in self.sub_items:
            for row in item.create_rows():
                yield row


class ActionsAndCriteria:
    def __init__(self, parent_set: ActionsAndCriteriaSet):
        self.parent_set = parent_set
        parent_set.sub_items.append(self)
        # actions that will be recommended with the same criteria
        self.actions: list[RecommendedAction] = []
        # criteria that will be ANDed together
        self.criteria: list[ConditionBase] = []

    def create_sub_criteria(self):
        # all criteria from the prior list are kept
        # however, all actions are cleared out
        copy_obj = ActionsAndCriteria(self.parent_set)
        copy_obj.criteria.extend(self.criteria)
        return copy_obj

    def create_rows(self):
        if len(self.actions) > 0 and len(self.criteria) > 0:
            # AND all of the criteria together
            full_criteria = BooleanCondition.get_condition_list(self.criteria, all_required=True)
            # for each action, return a new row
            for action in self.actions:
                newRow = ActionableStepRow()
                newRow.criteria = full_criteria
                newRow.recommended_action = action
                yield newRow

    def add_row(self, currRow: ActionableStepRow):
        # note that the input rows can be one of the following:
        # also note that multiple criteria must all be satisfied (i.e., ANDed together)
        #   1-1 relationship: Both the recommended action and criteria are on the row
        #   1-N relationship: 1 recommended action with multiple criteria
        #       Each of the criteria will land on a separate child row
        #   N-1 Relationship: multiple recommended actions with 1 criteria
        #       Each of the recommended actions will land on a separate child row
        #   N-N Relationship: multiple recommended actions with multiple criteria
        #       Each of the recommended actions will land on a separate child row
        #       Each of the criteria will land on a separate child row
        #   Combinations of these
        # Using this structure, criteria apply to:
        #   - If have child rows, then apply to all child rows
        #   - If do NOT have child rows, but has a parent, then apply to all rows on the same level
        #   - If do NOT have child rows and NO parent, then apply just to the own row
        # first, find the items without any parents, as these will be the roots for any logical trees
        # This structure means that the following occurs:
        #   Any criteria on a row, or any child rows with ONLY criteria (and no children) get ANDed together
        #   Any actions on a row, or any child rows with ONLY actions (and no children), get proposed using the same criteria
        if currRow.is_criteria:
            self.criteria.append(currRow.criteria)

        if currRow.is_recommended_action:
            self.actions.append(currRow.recommended_action)

        rows_which_extend: list[ActionableStepRow] = []
        for child in currRow.children:
            if not child.is_parent:
                if child.is_criteria and not child.is_recommended_action:
                    self.criteria.append(child.criteria)
                elif child.is_recommended_action and not child.is_criteria:
                    self.actions.append(child.recommended_action)
                else:
                    rows_which_extend.append(child)
            else:
                rows_which_extend.append(child)

        # for all children which will extend this, perform the extension
        for child_row in rows_which_extend:
            this_extension = self.create_sub_criteria()
            this_extension.add_row(child_row)


def getValueFromSmartsheetCell(sourceRow: SmartsheetRow, colID: int):
    associatedCell: SmartsheetCell = sourceRow.get_column(colID)
    cellVal = associatedCell.value
    return cellVal


def get_actionable_step_smartsheet_rows():
    apiModel = Smartsheet(access_token=api_key['access_token'], user_agent=api_key['user_agent'])
    apiModel.errors_as_exceptions(True)
    sheets = Sheets(apiModel)
    actionableStepSmartsheet: SmartsheetSheet = sheets.get_sheet(api_key['sheet_id'], page_size=5000)

    # get the map of the column names to IDs
    column_map: dict[str, int] = {}
    for column in actionableStepSmartsheet.columns:
        assert isinstance(column, SmartsheetColumn), "Input warranty column (type '{0}') is not the correct type".format(type(column))
        column_map[column.title] = column.id_

    # get the IDs of the columns commonly used
    actionColId = column_map["Recommended Action"]
    roleColId = column_map["Responsible Party"]
    criteriaColumnColId = column_map["Criteria - Column"]
    criteriaLowerBoundColId = column_map["Criteria - Lower Bound"]
    criteriaUpperBoundColId = column_map["Criteria - Upper Bound"]
    criteriaIncludedItemsColId = column_map["Criteria - Included Items"]
    criteriaBoundsAsOrColId = column_map["Criteria Bounds as OR"]

    allRows: list[ActionableStepRow] = []
    rowsById: dict[int, ActionableStepRow] = {}
    parentToChildIdMap: dict[int, list[int]] = {}
    currRow: SmartsheetRow
    for currRow in actionableStepSmartsheet.rows:
        # pull relevant data from the row
        action: str = getValueFromSmartsheetCell(currRow, actionColId)
        role: str = getValueFromSmartsheetCell(currRow, roleColId)
        criteriaColumn = getValueFromSmartsheetCell(currRow, criteriaColumnColId)
        criteriaLowerBound = getValueFromSmartsheetCell(currRow, criteriaLowerBoundColId)
        criteriaUpperBound = getValueFromSmartsheetCell(currRow, criteriaUpperBoundColId)
        criteriaIncludedItems = getValueFromSmartsheetCell(currRow, criteriaIncludedItemsColId)
        criteriaBoundsAsOr = getValueFromSmartsheetCell(currRow, criteriaBoundsAsOrColId)

        # create the row
        newActionableRow = ActionableStepRow()
        allRows.append(newActionableRow)
        rowsById[currRow.id_] = newActionableRow

        # connect to parent
        # connect by ID in case the parent ID hasn't been seen yet
        # after all rows are encountered, then created the connections
        if currRow.parent_id is not None:
            childrenForParent: list[int]
            if currRow.parent_id in parentToChildIdMap:
                childrenForParent = parentToChildIdMap[currRow.parent_id]
            else:
                childrenForParent = []
                parentToChildIdMap[currRow.parent_id] = childrenForParent
            childrenForParent.append(currRow.id_)

        # add the action, if there is one
        # note that the action must be given
        if action is not None:
            actionData = RecommendedAction(action, role)
            newActionableRow.recommended_action = actionData

        # add the criteria, if there is one
        if criteriaColumn is not None:
            rowCriteria: ConditionBase | None = None
            # if is an included item, then check here
            if criteriaIncludedItems is not None:
                # each of the criteria should be split by a new line character
                options = criteriaIncludedItems.split("\n")
                rowCriteria = ListCondition.get_conditions_for_list(criteriaColumn, options, True)
            elif criteriaUpperBound is not None and criteriaLowerBound is not None:
                if criteriaBoundsAsOr:
                    lessThanCondtion = ValueCondition(criteriaColumn, ConditionOperations.LESS_THAN_EQUAL, criteriaUpperBound)
                    greaterThanCondition = ValueCondition(criteriaColumn, ConditionOperations.GREATER_THAN_EQUAL, criteriaLowerBound)
                    rowCriteria = BooleanCondition([lessThanCondtion, greaterThanCondition], all_required=False)
                else:
                    rowCriteria = RangeCondition(criteriaColumn, criteriaLowerBound, True, criteriaUpperBound, True)
            elif criteriaUpperBound is not None:
                rowCriteria = ValueCondition(criteriaColumn, ConditionOperations.LESS_THAN_EQUAL, criteriaUpperBound)
            elif criteriaLowerBound is not None:
                rowCriteria = ValueCondition(criteriaColumn, ConditionOperations.GREATER_THAN_EQUAL, criteriaLowerBound)
            newActionableRow.criteria = rowCriteria

    # lastly, connect parent to children
    for parentId, childIds in parentToChildIdMap.items():
        parentRow = rowsById[parentId]
        for childId in childIds:
            childRow = rowsById[childId]
            childRow.parent = parentRow
            parentRow.children.append(childRow)
    return allRows


def build_actions_and_criteria():
    # converts the smartsheet row inputs into a recommended action with associated criteria
    input_rows = get_actionable_step_smartsheet_rows()
    setHandler = ActionsAndCriteriaSet()
    setHandler.add_input_rows(input_rows)
    return list(setHandler.create_rows())


if __name__ == "__main__":
    data = build_actions_and_criteria()
    print(len(data))
