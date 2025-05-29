import numpy
import pandas as pd
from overall_helpers import *
import numpy as np
from data_table import DataTable, DataFolder, CSVParserOptions
from data_collection import DataCollection, JoinType
from data_columns import DataColumnAggregation
import time
from datetime import date, timedelta
from date_iterator import MonthIterator
from progress_bar import ProgressBar
from dynamics_integrator import get_safety_data_results

# Create new folder with current date and time
current_datetime = time.strftime('%Y-%m-%d %H_%M')

root_folder = DataFolder(".", None)
data_sources_folder = root_folder.get_folder("Data Sources")
generated_folder = data_sources_folder.get_folder("Generated and Complete")
injury_folder = data_sources_folder.get_folder("Injuries")
employee_folder = data_sources_folder.get_folder("Employee")
project_folder = data_sources_folder.get_folder("Project Data")
change_order_folder = project_folder.get_folder("Change Orders")
observations_folder = data_sources_folder.get_folder("Observations")
labor_folder = data_sources_folder.get_folder("Labor Hours")
rework_folder = data_sources_folder.get_folder("Rework")
driving_distance_folder = employee_folder.get_folder("Driving Distances")

# flags for keeping track of the debug modes
injuryIdCol = "Master Injury ID"


def readInjuryFile(include_injury_id=False) -> DataTable:
    injuryFile = injury_folder.get_data_source("Injury Responsibility")
    injuryData = injuryFile.get_data(name="Injuries")
    # for the injuries, only keep the injury ID if requested
    if include_injury_id:
        injuryData.add_dimensions([injuryIdCol])
    else:
        injuryData.remove_data_column(injuryIdCol)
    # non-common dimensions
    injuryData.add_dimensions(["Craft Responsible for Injury", "Cause of Injury", "Classification",
                               "Affected Body Part"])
    # set up the default branch / profit center type
    profitCenterColName = "Profit Center - Original"
    branchAndProfitCenterDesignations = injury_folder.get_data_source("Department Types").get_data()
    branchLookup = branchAndProfitCenterDesignations.create_lookup(profitCenterColName, "Branch")
    profitCenterTypeLookup = branchAndProfitCenterDesignations.create_lookup(profitCenterColName, "Department Type")
    profitCenterColVals = injuryData.df[profitCenterColName]
    # set the default branch value
    injuryData.set_dimension_data_columns(branchCol, profitCenterColVals.replace(branchLookup).replace(np.nan, "Unknown"))
    # set the default profit center type lookup
    injuryData.set_dimension_data_columns(profitCenterTypeCol, profitCenterColVals.replace(profitCenterTypeLookup).replace(np.nan, "Unknown"))
    # remove the columns related to the profit center columns that aren't used
    injuryData.remove_data_columns(["Profit Center ID", "Profit Center Description", "Profit Center - Original"], False)
    # change the date to the given week for easier comparisons
    injuryData.convert_to_week(dateCol)
    injuryData.auto_assign_measures()
    return injuryData


def readChangeOrderFile() -> DataTable:
    changeOrderDataTable = change_order_folder.get_data_source("Change Order Data").get_data(auto_populate=True)
    changeOrderDataTable.rename_measures(prefix="Change Order ")
    return changeOrderDataTable


def readObservationsFile(max_rows_to_sample: int | None = None) -> DataTable:
    obsDataTable = observations_folder.get_data_source("Observation Metrics", max_rows_to_sample=max_rows_to_sample).get_data(auto_populate=True)
    obsDataTable.rename_measures(suffix=" Observations")
    return obsDataTable


def readHoursFile(max_rows_to_sample: int | None = None) -> DataTable:
    # for the field hours, add the foreman hours if the foreman id matches the employee id
    fieldHoursWithTaskData = labor_folder.get_data_source("Field Hours by Task", max_rows_to_sample=max_rows_to_sample).get_data(auto_populate=True)
    hoursDF = fieldHoursWithTaskData.df
    foremanHoursData = np.where(hoursDF[foremanIdCol] == hoursDF[employeeIdCol], hoursDF[hoursCol], 0)
    fieldHoursWithTaskData.set_measure_data_column("Foreman Hours", foremanHoursData)
    # create the calculated fields
    fieldHoursWithTaskData.add_calculated_field('% Foreman to Field Labor', ["Foreman Hours", hoursCol],
                                                lambda foremanHours, hours: foremanHours / hours)
    return fieldHoursWithTaskData


def readReworkFile(max_rows_to_sample: int | None = None) -> DataTable:
    reworkTable = rework_folder.get_data_source("Rework Events", max_rows_to_sample=max_rows_to_sample).get_data(auto_populate=True)
    return reworkTable


def readTravelFile(max_rows_to_sample: int | None = None) -> DataTable:
    # note that the travel file was pulled for 2018 and after
    travelTable = DataTable("Employee Working Distances Updated.xlsx", auto_populate=False, max_rows_to_sample=max_rows_to_sample)
    # add non-common dimensions
    travelTable.add_dimensions(["Project Address", "Home Address"])
    # aggregate distance and duration as averages, so that individual distances can be compared
    # however, total distance and duration can be calculated by multiplying by the number of days
    travelTable.add_measures(["Distance", "Duration"], aggregation_method=DataColumnAggregation.AVG)
    travelTable.auto_assign_measures()
    # remove any long distances travelled
    # remove any records where the values weren't found
    travelTable.filter("Distance").is_between_exclusive(0, 200)
    return travelTable


def readProjectActualsByCraftFile(max_rows_to_sample: int | None = None) -> DataTable:
    reworkTable = project_folder.get_data_source("Project Actuals and Revised by Craft",
                                                 max_rows_to_sample=max_rows_to_sample).get_data(auto_populate=True)
    return reworkTable


def readProjectActualsFile(max_rows_to_sample: int | None = None) -> DataTable:
    reworkTable = project_folder.get_data_source("Project Actuals and Revised",
                                                 max_rows_to_sample=max_rows_to_sample).get_data(auto_populate=True)
    return reworkTable


def readEmployeeClassificationHistory(max_rows_to_sample: int | None = None,
                                      aggregation_method: DataColumnAggregation = DataColumnAggregation.AVG) -> DataCollection:
    # by default, aggregate the position as a minimum so that keeps one of the values
    def create_history():
        # specify commonly used columns
        startDateAttr = "Start Date"
        hireDateAttr = "HireDate"
        endDateAttr = "End Date"

        jobTitleAttr = "Job Title"
        jobIdAttr = "Job ID"
        useJobTitle = True
        jobAttrToUse = jobTitleAttr if useJobTitle else jobIdAttr

        employeeClassificationFile = employee_folder.get_sub_path("Employee Classification History.xlsx")
        employeeHistory = DataTable(employeeClassificationFile, auto_populate=False,
                                    max_rows_to_sample=max_rows_to_sample, sheet_name="Classification Periods")
        classificationTable = DataTable(employeeClassificationFile, auto_populate=False,
                                        sheet_name="Classifications")

        # set up the employee history table
        employeeHistory.add_dimensions([endDateAttr, startDateAttr, jobAttrToUse, hireDateAttr])
        employeeHistory.remove_data_column(jobIdAttr if useJobTitle else jobTitleAttr)
        employeeHistory.convert_to_date([endDateAttr, startDateAttr, hireDateAttr])
        employeeHistory.auto_assign_measures()
        # create the classification lookup
        classificationTable.add_dimensions([jobTitleAttr, jobIdAttr, positionTypeCol])
        classificationLookup = classificationTable.create_lookup(jobAttrToUse, positionTypeCol)
        # split on a week-by-week-basis
        # go through each employee, group the data together (sorting on start date), and calculate the metrics
        grouped_by_employee = employeeHistory.df.groupby(employeeIdCol)
        newData = []
        startDateAnalysis = date(2010, 1, 1)
        pb = ProgressBar(len(grouped_by_employee.groups))
        for empId, rowIds in grouped_by_employee.groups.items():
            employee_history: pd.DataFrame = employeeHistory.df.loc[rowIds]
            sorted_history = employee_history.sort_values(startDateAttr)
            timeAtMck = 0
            timeSinceLastHired = 0
            timesInPositionsSinceLastHired: Dict[str, float] = {}
            timesInPositionsAtMcK: Dict[str, float] = {}
            timesInPositionTypeAtMcK: Dict[str, float] = {}
            timesInPositionTypeSinceLastHired: Dict[str, float] = {}
            pb.update()
            for idx, currRow in sorted_history.iterrows():
                startDate: date = currRow[startDateAttr].date()
                hireDate: date = currRow[hireDateAttr].date()
                position = currRow[jobAttrToUse]
                positionType = classificationLookup.get(position, "Unknown")
                # if just hired, then reset the clock
                if hireDate >= startDate:
                    timeSinceLastHired = 0
                    timesInPositionsSinceLastHired = {}
                    timesInPositionTypeSinceLastHired = {}

                # get the times in the positions
                timeInPositionSinceLastHired = timesInPositionsSinceLastHired.get(position, 0)
                timeInPositionAtMcK = timesInPositionsAtMcK.get(position, 0)
                timeInPositionTypeSinceLastHired = timesInPositionTypeSinceLastHired.get(positionType, 0)
                timeInPositionTypeAtMcK = timesInPositionTypeAtMcK.get(positionType, 0)

                # if starts before the analysis time, then cache the data
                endDate: date = currRow[endDateAttr].date()
                if startDate < startDateAnalysis:
                    endDateForPrecomputing = min(endDate, startDateAnalysis)
                    timeElapsed = (endDateForPrecomputing - startDate).days / 365
                    timeAtMck += timeElapsed
                    timeSinceLastHired += timeElapsed
                    timeInPositionSinceLastHired += timeElapsed
                    timeInPositionAtMcK += timeElapsed
                    timeInPositionTypeSinceLastHired += timeElapsed
                    timeInPositionTypeAtMcK += timeElapsed
                    startDate = startDateAnalysis

                # move through the time periods, and log the data
                timeIterator = MonthIterator(startDate, endDate)
                while timeIterator.move_forward():
                    # create the new record
                    newData.append([empId, timeIterator.month, timeIterator.year, position, positionType,
                                    round(timeAtMck, 2), round(timeSinceLastHired, 2), round(timeInPositionAtMcK, 2),
                                    round(timeInPositionSinceLastHired, 2), round(timeInPositionTypeAtMcK, 2),
                                    round(timeInPositionTypeSinceLastHired, 2)])
                    # add to all of the times
                    timeAtMck += timeIterator.time_in_period
                    timeSinceLastHired += timeIterator.time_in_period
                    timeInPositionAtMcK += timeIterator.time_in_period
                    timeInPositionSinceLastHired += timeIterator.time_in_period
                    timeInPositionTypeSinceLastHired += timeIterator.time_in_period
                    timeInPositionTypeAtMcK += timeIterator.time_in_period

                # save off the times in the positions
                timesInPositionsSinceLastHired[position] = timeInPositionSinceLastHired
                timesInPositionsAtMcK[position] = timeInPositionAtMcK
                timesInPositionTypeAtMcK[positionType] = timeInPositionTypeAtMcK
                timesInPositionTypeSinceLastHired[positionType] = timeInPositionTypeSinceLastHired
        pb.finish()
        # create the new data frame
        return pd.DataFrame(newData, columns=[employeeIdCol, monthCol, yearCol, positionCol, positionTypeCol,
                                              "Time at McK", "Time Since Last Hired", "Time in Position",
                                              "Time in Position Since Last Hired", "Time in Position Type",
                                              "Time in Position Type Since Last Hired"])

    # use the cached version if exists and is desired
    outputHistory = employee_folder.get_data_source("Tenure by Month", max_rows_to_sample=max_rows_to_sample,
                                                    creator_func=create_history).get_data()

    # set up the dimensions and measures
    outputHistory.add_dimensions([positionCol, positionTypeCol])
    outputHistory.auto_assign_measures()
    # aggregate the times by the average time
    # this is because if an employee goes from one role to another, then should say that is the average of the two roles
    # however, allow to change the aggregation method, in case are aggregating on non-month basis
    # in which case, the minimum / max may be more appropriate
    outputHistory.add_measures(["Time Since Last Hired", "Time in Position",
                                "Time in Position Since Last Hired", "Time in Position Type",
                                "Time in Position Type Since Last Hired"], aggregation_method=aggregation_method)
    return outputHistory


def get_difference_array(array: np.ndarray):
    # create a zero array, and set the last values to the given differences
    full_diff = np.zeros(array.shape)
    if array.shape[0] > 1:
        # calculate the difference.
        diff_array = np.diff(array)
        full_diff[1:] = diff_array
    return full_diff


def get_cumulative_sum_and_resetting_sum(original_array: np.ndarray, reset_mask: np.ndarray, reset_to_zero=True):
    # gets the cumulative sum for the original array, as well as resets the value when the reset mask is true
    # when reset to zero is true, then will reset each value of the reset mask to 0.
    # Otherwise, will reset to the original value
    # for example, assume an original array of [1, 1, 1, 1, 1, 1]
    # with reset mask of [0, 0, 1, 0, 1, 0]
    # First, calculate the cumulative sum, or [1, 2, 3, 4, 5, 6]
    cum_sum_arr: np.ndarray = original_array.cumsum()
    # get the values where it resets
    cum_sum_at_reset = cum_sum_arr[reset_mask]
    if cum_sum_at_reset.shape[0] == 0:
        # if doesn't ever reset, then means that the resetting will be the same as the cumulative sum
        return cum_sum_arr, cum_sum_arr

    # next, calculate the offset values
    # Start with the cumulative values at the reset indices, or [0, 0, 3, 0, 5, 0]
    # get the differences between these values, so that won't reset each time
    reset_adjustments = get_difference_array(cum_sum_at_reset)
    reset_adjustments[0] = cum_sum_at_reset[0]
    # if not resetting to zero (i.e., resetting to the original value), then subtract the original value from these
    # values. When this mask is subtracted from the cumulative sum, then will leave behind the original value
    #   For given example, if not resetting to zero, then this would be [0, 0, 2, 0, 4, 0],
    #   leading to a cumulative sum of [0, 0, 2, 2, 4, 4],
    #   leading to a reset sum of [1, 2, 1, 2, 1, 2]
    if not reset_to_zero:
        target_initial_amount = original_array[reset_mask]
        # follow a similar approach to the adjustments. This will add an initial amount for an adjustment,
        # and then subtract off the initial amount, and add the initial amount for the next item
        target_initial_adjustment = get_difference_array(target_initial_amount)
        target_initial_adjustment[0] = target_initial_amount[0]
        reset_adjustments -= target_initial_adjustment
    reset_offsets_seeds = np.zeros(original_array.shape)
    reset_offsets_seeds[reset_mask] = reset_adjustments

    # then, take the cumulative sum, so that get [0, 0, 3, 3, 5, 5]
    reset_offsets = reset_offsets_seeds.cumsum()
    # then, can remove from the cumulative sum to get [1, 2, 0, 1, 0, 1]
    reset_values = cum_sum_arr - reset_offsets
    return cum_sum_arr, reset_values


def readTenureOnProjects(max_rows_to_sample: int | None = None, keep_as_weeks=True) -> DataCollection:
    totalTimeOnProjectAttr = "Total Time on Project"
    continuousTimeOnProjectAttr = "Continuous Time on Project"
    totalWeeksOnProjectAttr = "Total Weeks on Project"
    continuousWeeksOnProjectAttr = "Continuous Weeks on Project"

    def create_timeline():
        # read in the time on each project
        # read in only the employee hours, as don't need task
        # also, keep the date, so will know if are absent for a while
        hoursFile = labor_folder.get_data_source("Field Hours by Task", debug_mode=DEBUG_DEFAULT, max_rows_to_sample=max_rows_to_sample).get_data()
        hoursFile.add_dimensions([projectCol, employeeIdCol, foremanIdCol, weekCol])
        hoursFile.add_measures([hoursCol], aggregation_method=DataColumnAggregation.SUM)
        hoursFile.keep_data_columns([projectCol, employeeIdCol, foremanIdCol, weekCol, hoursCol])
        hoursFile.df[weekCol] = pd.to_datetime(hoursFile.df[weekCol])
        numHoursRows = len(hoursFile.df)
        hoursFile.set_measure_data_column(totalTimeOnProjectAttr, np.zeros(numHoursRows))
        hoursFile.set_measure_data_column(continuousTimeOnProjectAttr, np.zeros(numHoursRows))
        hoursFile.set_measure_data_column(totalWeeksOnProjectAttr, np.zeros(numHoursRows))
        hoursFile.set_measure_data_column(continuousWeeksOnProjectAttr, np.zeros(numHoursRows))

        grouped_by_employee = hoursFile.df.groupby([employeeIdCol, projectCol])
        pb = ProgressBar(len(grouped_by_employee.groups))
        for rowIds in grouped_by_employee.groups.values():
            pb.update()
            employee_history: pd.DataFrame = hoursFile.df.loc[rowIds]
            sorted_history = employee_history.sort_values(weekCol)
            sorted_rowIds = sorted_history.index
            # figure out where the weeks have a large break in them
            # a large break is defined as a "skip" of 4 weeks
            # calculate the difference. Will return as nanosecond difference, so convert to weeks
            weeksDifference = get_difference_array(sorted_history[weekCol].values) / 604800000000000
            # set the first difference as one, so that will start counting at 1 below
            weeksDifference[0] = 1
            # lastly, flag any weeks where the skip is greater than the max
            skipWeeks = weeksDifference >= 4

            # get the number of hours
            hoursWorkedOnProject: pd.Series = sorted_history[hoursCol]
            # get the total time on project, and the continuous time on project, and set within the data frame
            cumulative_time, continuous_time = get_cumulative_sum_and_resetting_sum(hoursWorkedOnProject.values,
                                                                                    skipWeeks, False)
            hoursFile.df.loc[sorted_rowIds, totalTimeOnProjectAttr] = cumulative_time
            hoursFile.df.loc[sorted_rowIds, continuousTimeOnProjectAttr] = continuous_time

            # calculate the number of weeks worked on the project
            weeksWorked = np.ones(len(hoursWorkedOnProject))
            # don't include any weeks with duplicate dates
            weeksWorked[weeksDifference == 0] = 0
            cumulative_weeks, continuous_weeks = get_cumulative_sum_and_resetting_sum(weeksWorked, skipWeeks, False)
            hoursFile.df.loc[sorted_rowIds, totalWeeksOnProjectAttr] = cumulative_weeks
            hoursFile.df.loc[sorted_rowIds, continuousWeeksOnProjectAttr] = continuous_weeks
        pb.finish()
        return hoursFile

    outputHistory = labor_folder.get_data_source("Time on Projects", max_rows_to_sample=max_rows_to_sample,
                                                 creator_func=create_timeline).get_data()
    # average out the total time and continuous time on projects
    outputHistory.add_measures([totalTimeOnProjectAttr, continuousTimeOnProjectAttr, totalWeeksOnProjectAttr,
                                continuousWeeksOnProjectAttr], aggregation_method=DataColumnAggregation.AVG)
    outputHistory.auto_assign_measures(keep_dates=keep_as_weeks)
    return outputHistory


def build_rework_metrics_file():
    reworkFile = readReworkFile()
    injuryFile = readInjuryFile()
    # find all projects with time after 2019
    hoursFile = readHoursFile()
    projectsWithTimeAfter2019 = hoursFile.copy()
    projectsWithTimeAfter2019.keep_data_columns([branchCol, profitCenterTypeCol, yearCol, projectCol, hoursCol])
    projectsWithTimeAfter2019.filter_remove_before_year(2019)
    projectsWithTimeAfter2019.filter(hoursCol).is_greater_than(200)
    projectsWithTimeAfter2019.remove_data_columns([yearCol, hoursCol])
    # only read in hours for projects with any time after 2019
    allHoursForProjectsWithAnyHoursAfter2019 = hoursFile.join(projectsWithTimeAfter2019, JoinType.INNER)
    allHoursForProjectsWithAnyHoursAfter2019.remove_data_columns(['Foreman Hours', '% Foreman to Field Labor'])
    allHoursForProjectsWithAnyHoursAfter2019.rename_column(hoursCol, "Field Hours after 2019")
    injuryFile.filter_remove_before_year(2019)
    reworkFile.filter_remove_before_year(2019)
    projectActualFile = readProjectActualsFile()
    projectActualFile.filter("Actual Hours").is_greater_than(10)
    projectActualFile.filter("Actual Cost").is_greater_than(10)
    # outer join the rework and injuries, and then join to the projects file
    # only join on projects that have injuries or rework for
    rework_metrics_file = injuryFile \
        .join(reworkFile, JoinType.OUTER) \
        .join(allHoursForProjectsWithAnyHoursAfter2019, JoinType.OUTER) \
        .join(projectActualFile, JoinType.INNER)
    rework_metrics_file.add_calculated_field("Rework Rate", ["Rework Cost", "Actual Cost"], lambda rc, ac: rc / ac)
    print(rework_metrics_file)
    rework_folder.save_data("Rework Injury Metrics", rework_metrics_file)


def build_travel_time_analysis_file():
    travel_file = readTravelFile()
    injury_file = readInjuryFile()
    injury_file.filter_remove_before_year(2018)
    # keep all travel data, but only merge in the injuries for the people who have data for
    injury_and_travel_metrics = travel_file.join(injury_file, JoinType.LEFT)
    print(injury_and_travel_metrics)
    driving_distance_folder.save_data("Travel Injury Metrics", injury_and_travel_metrics)


def build_employee_files(max_rows_to_sample: int | None = None):
    # read in the injuries
    injuryFile = readInjuryFile(True)

    # read in the amount of time each employee has been on each project (tenure)
    hoursFile = readTenureOnProjects(max_rows_to_sample)
    hoursFile.name = "Project Tenure"

    # first, join the tenure on the projects with the injuries
    # outer join, so that know what tenure people were when they were injured, regardless of project time
    employee_injuries = injuryFile.copy("Employee Injuries")
    # remove employee injuries that don't have an employee set
    employee_injuries.filter(employeeIdCol).is_not_equal_to("")
    projectTenureAndInjury = hoursFile.join(employee_injuries, JoinType.OUTER,
                                            joining_dimensions=[monthCol, yearCol, projectCol, employeeIdCol],
                                            extra_dimensions=[injuryIdCol],
                                            source_indicator=True)

    # read in the positions
    # use the minimum time, so that if someone switches roles, the minimum role is considered
    position_file = readEmployeeClassificationHistory(max_rows_to_sample=max_rows_to_sample,
                                                      aggregation_method=DataColumnAggregation.MIN)
    position_file.name = "Employee Tenure"
    # only take on the joining dimensions
    position_file.add_measures([positionCol, positionTypeCol], aggregation_method=DataColumnAggregation.MIN)
    position_file.keep_dimensions([monthCol, yearCol, employeeIdCol])
    position_file.add_dimensions([positionCol, positionTypeCol])

    # next, join in the position tenure information
    empHoursInjuriesAndClassifications = projectTenureAndInjury.join(position_file, JoinType.LEFT,
                                                                     joining_dimensions=[monthCol, yearCol, employeeIdCol],
                                                                     extra_dimensions=[positionCol, positionTypeCol, projectCol, injuryIdCol],
                                                                     source_indicator=True)
    empHoursInjuriesAndClassifications.split_project_number()
    emp_saved_data = employee_folder.save_data("Tenure vs Injuries", empHoursInjuriesAndClassifications, debug_mode=False)
    emp_saved_data.save_both = True

    # now, filter only the foreman data, and create the data based on these data as well
    foremanTenure = hoursFile.copy("Foreman Hours")
    # filter where the employee id is the same as the foreman id
    foreman_data = foremanTenure.df[foremanIdCol]
    is_foreman = np.logical_or(foreman_data == foremanTenure.df[employeeIdCol], foreman_data == "")
    foremanTenure.filter_where(is_foreman)
    foremanTenure.rename_column(hoursCol, "Foreman Hours")
    # get the field hours, and join in
    fieldHours = readHoursFile()
    fieldHours.name = "Field Hours"
    fieldHours.keep_data_columns([projectCol, yearCol, monthCol, foremanIdCol, hoursCol])
    fmHoursAndFieldHours = foremanTenure.join(fieldHours, JoinType.LEFT,
                                              joining_dimensions=[monthCol, yearCol, projectCol, foremanIdCol])
    fmHoursAndFieldHours.name = "Project Tenure"

    # first, join the tenure on the projects with the injuries
    # outer join, so that know what tenure people were when they were injured, regardless of project time
    foreman_responsible_injuries = injuryFile.copy("Foreman Responsible Injuries")
    # remove where don't know the foreman
    foreman_responsible_injuries.filter(foremanIdCol).is_not_equal_to("")
    fmProjectTenureAndInjury = fmHoursAndFieldHours.join(foreman_responsible_injuries, JoinType.OUTER,
                                                         joining_dimensions=[monthCol, yearCol, projectCol, foremanIdCol],
                                                         extra_dimensions=[injuryIdCol],
                                                         source_indicator=True)
    # rename the position to foreman id column
    foreman_position_history = position_file.copy("Foreman Tenure")
    foreman_position_history.rename_column(employeeIdCol, foremanIdCol)
    # next, join in the position tenure information
    fmHoursInjuriesAndClassifications = fmProjectTenureAndInjury.join(foreman_position_history, JoinType.LEFT,
                                                                      joining_dimensions=[monthCol, yearCol, foremanIdCol],
                                                                      extra_dimensions=[positionCol, positionTypeCol, projectCol, injuryIdCol],
                                                                      source_indicator=True)
    fmHoursInjuriesAndClassifications.split_project_number()
    fm_saved_data = employee_folder.save_data("Foreman Tenure vs Injuries", fmHoursInjuriesAndClassifications, debug_mode=False)
    fm_saved_data.save_both = True


def buildProjectActualsWithEmployeeData():
    use_source_indicator = False
    # get the field hours file, broken up by month and year
    hoursFile = labor_folder.get_data_source("Field Hours by Task").get_data(name="Field Hours")
    hoursFile.add_dimensions([projectCol, employeeIdCol, foremanIdCol, monthCol, yearCol, taskCodeIdCol])
    hoursFile.add_measures([hoursCol], aggregation_method=DataColumnAggregation.SUM)
    # convert task to craft - dont perform consolidation yet
    hoursFile.split_task_code(False)
    # consolidate data
    hoursFile.keep_data_columns([projectCol, employeeIdCol, foremanIdCol, monthCol, yearCol, craftCol, hoursCol])
    hoursFile.convert_common_ids_to_string()

    # build out the foreman column, filling in any blanks
    foreman_ids = hoursFile.df[foremanIdCol]
    is_foreman = np.logical_or(foreman_ids == hoursFile.df[employeeIdCol], foreman_ids == "")
    hoursFile.df[foremanIdCol].where(~is_foreman, other=hoursFile.df[employeeIdCol], inplace=True)

    # join in the injuries
    # note that an employee could have worked under multiple foremen in the given time period
    injuryFile = readInjuryFile(True)
    injuryFile.filter(employeeIdCol).is_not_equal_to("")
    injuryFile.keep_dimensions([monthCol, yearCol, projectCol, employeeIdCol, foremanIdCol])
    hours_by_emp = hoursFile.df.groupby([monthCol, yearCol, projectCol, employeeIdCol])
    injuries_by_emp = injuryFile.df.groupby([monthCol, yearCol, projectCol, employeeIdCol])
    # also, the foremen with injuries attributed may not be the foremen who entered the time
    # because of this, spread any injuries attributed to non-present foremen across the foremen who are present,
    # weighted by the hours worked by the present foremen
    recorded_injuries = []
    injuryCols = [numInjuriesCol, numRecordablesCol]
    for injury_key, injuryRowIds in injuries_by_emp.groups.items():
        if injury_key in hours_by_emp.groups:
            hours_indices = hours_by_emp.groups[injury_key]
            m, y, p, e = injury_key
            hoursByForeman = group_data_frame(hoursFile.df.loc[hours_indices], foremanIdCol, {hoursCol: "sum"})
            total_hours = hoursByForeman[hoursCol].sum()
            # build the ratio of the hours that the foreman submitted time for this employee
            foreman_ratio_lookup = {row[foremanIdCol]: row[hoursCol] / total_hours for idx, row in hoursByForeman.iterrows()}
            foreman_values_issued = {f: {c: 0 for c in injuryCols} for f in foreman_ratio_lookup.keys()}
            for i in injuryRowIds:
                injuryForeman = injuryFile.df.loc[i, foremanIdCol]
                for c in [numInjuriesCol, numRecordablesCol]:
                    numInjuries = injuryFile.df.loc[i, c]
                    if injuryForeman in foreman_ratio_lookup:
                        foreman_values_issued[injuryForeman][c] += numInjuries
                    else:
                        for f, r in foreman_ratio_lookup.items():
                            foreman_values_issued[f][c] += numInjuries * r
            for f, d in foreman_values_issued.items():
                new_row_data = {monthCol: m, yearCol: y, projectCol: p, employeeIdCol: e, foremanIdCol: f}
                for c in injuryCols:
                    new_row_data[c] = d[c]
                recorded_injuries.append(new_row_data)
    injuryFile.set_data_frame(pd.DataFrame.from_records(recorded_injuries))

    # add an injury row ID so can tell how many rows the injury goes to
    # note that this is different from the injury ID, as an employee may get injured more than once for a month / year / project / foreman
    injuryRowIdCol = "Injury Row ID"
    injuryFile.set_dimension_data_columns(injuryRowIdCol, list(range_one_based(len(injuryFile.df))))
    # if has a match with the foreman, month, year, and project, then match up
    # otherwise, then assign the unmatched values to the records for the employee for the given month
    output_data = hoursFile.join(injuryFile, JoinType.LEFT,
                                 joining_dimensions=[monthCol, yearCol, projectCol, employeeIdCol, foremanIdCol],
                                 extra_dimensions_left=[foremanIdCol, craftCol],
                                 extra_dimensions_right=[injuryRowIdCol],
                                 source_indicator=use_source_indicator)
    # get the row ids, and how many hours fall into each row id
    included_injury_row_ids = output_data.df[injuryRowIdCol]
    duplicated_injury_rows_mask = (included_injury_row_ids > 0) & included_injury_row_ids.duplicated()
    # figure out the number of hours per injury row id
    injuries_needing_weighting = output_data.df.loc[duplicated_injury_rows_mask]
    # should weight by the number of hours in the record
    total_hours_for_injury = injuries_needing_weighting[hoursCol].groupby(injuries_needing_weighting[injuryRowIdCol]).transform("sum")
    output_data.df.loc[duplicated_injury_rows_mask, numInjuriesCol] = injuries_needing_weighting[hoursCol] / \
                                                                      total_hours_for_injury * injuries_needing_weighting[numInjuriesCol]
    output_data.df.loc[duplicated_injury_rows_mask, numRecordablesCol] = injuries_needing_weighting[hoursCol] / \
                                                                         total_hours_for_injury * injuries_needing_weighting[numRecordablesCol]

    # # analyze the missing injuries
    # # find the injury file dimensions which didn't have a match
    # unique_injury_row_ids = set(included_injury_row_ids.unique())
    # missing_injury_ids = list(set(range(1, len(injuryFile.df) + 1)).difference(unique_injury_row_ids))
    # missing_injury_dimensions = injuryFile.df.loc[injuryFile.df[injuryRowIdCol].isin(missing_injury_ids), list(injuryFile.dimension_names)]
    # should_check = np.ones(len(hoursFile.df), dtype=bool)
    # for d in [monthCol, yearCol, projectCol, employeeIdCol]:
    #     injury_dim_values = missing_injury_dimensions[d].unique()
    #     is_relevant = hoursFile.df[d].isin(injury_dim_values)
    #     should_check = np.logical_and(should_check, is_relevant)
    # # now, find the values
    # hours_for_missing_data = hoursFile.df.loc[should_check]

    output_data.remove_data_column(injuryRowIdCol)

    # build the foreman hours on the job
    foreman_hours = hoursFile.copy("Foreman Hours")
    foreman_hours.filter_where(is_foreman)
    foreman_hours.keep_dimensions([projectCol, foremanIdCol, monthCol, yearCol, craftCol])
    foreman_hours.rename_measures(prefix="Foreman ")
    output_data.join_simple_inplace(foreman_hours, source_indicator=use_source_indicator)

    # add in the foreman observations
    foremanObservations = readObservationsFile()
    foremanObservations.name = "Foreman Observations"
    foremanObservations.rename_measures(prefix="Foreman ")
    foremanObservations.keep_dimensions([foremanIdCol, projectCol, yearCol, monthCol])
    # should the foreman observations be normalized in some way, like observations per week?
    output_data.join_simple_inplace(foremanObservations, source_indicator=use_source_indicator)

    # join in the amount of time each employee has been on each project (tenure)
    projectTenure = readTenureOnProjects()
    projectTenure.name = "Project Tenure"
    projectTenure.remove_data_columns([hoursCol, weekCol, branchCol, profitCenterTypeCol])
    output_data.join_simple_inplace(projectTenure, source_indicator=use_source_indicator)

    # join in the foreman time on projects
    foremanProjectTenure = projectTenure.copy("Foreman Project Tenure")
    foremanProjectTenure.rename_column(employeeIdCol, foremanIdCol)
    foremanProjectTenure.rename_measures(prefix="Foreman ")
    output_data.join_simple_inplace(foremanProjectTenure, source_indicator=use_source_indicator)

    # join in the employee tenure
    # use the minimum time, so that if someone switches roles, the minimum role is considered
    position_file = readEmployeeClassificationHistory(aggregation_method=DataColumnAggregation.MIN)
    position_file.name = "Employee Tenure"
    # only take on the joining dimensions
    position_file.add_measures([positionCol, positionTypeCol], aggregation_method=DataColumnAggregation.MIN)
    position_file.keep_dimensions([monthCol, yearCol, employeeIdCol])
    position_file.add_dimensions([positionCol, positionTypeCol])
    output_data.join_simple_inplace(position_file, source_indicator=use_source_indicator)

    # join in the foreman tenure
    foreman_tenure = position_file.copy("Foreman Tenure")
    foreman_tenure.rename_columns({employeeIdCol: foremanIdCol,
                                   positionCol: "Foreman " + positionCol,
                                   positionTypeCol: "Foreman " + positionTypeCol})
    foreman_tenure.rename_measures(prefix="Foreman ")
    output_data.join_simple_inplace(foreman_tenure, source_indicator=use_source_indicator)

    # build in the position type distribution on the craft and project level
    position_file_only_positions = position_file.copy("Employee Positions")
    position_file_only_positions.keep_measures([])
    positionAndHours = hoursFile.join_simple(position_file_only_positions, join_type=JoinType.LEFT)
    # keep only the hours per position type on projects and crafts
    positionAndHours.keep_dimensions([projectCol, craftCol, positionTypeCol, hoursCol])
    # only keep specific position types. All others should fall in an "other" bucket
    positionTypesToKeep = ['Apprentice', 'Classified Worker', 'Foreman', 'Helper', 'Journeyman', 'Technician', 'Tradesman']
    posTypeValues = positionAndHours.df[positionTypeCol]
    posTypeValues.where(posTypeValues.isin(positionTypesToKeep), "Other", inplace=True)
    # create a copy of the hours column for the total hours
    total_hours_col = hoursCol + "_Total"
    positionAndHours.set_measure_data_column(total_hours_col, positionAndHours.df[hoursCol])
    # perform the pivot
    positionTypeColList = positionAndHours.pivot_dimension(positionTypeCol, hoursCol)
    # create a copy to calculate percentages per craft
    crewMixByCraft = positionAndHours.copy("Crew Mix by Craft")
    crewMixByCraft.normalize(positionTypeColList, total_hours_col, col_prefix="Craft ", col_suffix=" %", factor=100)
    output_data.join_simple_inplace(crewMixByCraft, source_indicator=use_source_indicator)
    # consolidate by project
    crewMixByProject = positionAndHours.copy("Crew Mix by Project")
    crewMixByProject.remove_data_column(craftCol)
    crewMixByProject.normalize(positionTypeColList, total_hours_col, col_prefix="Project ", col_suffix=" %", factor=100)
    output_data.join_simple_inplace(crewMixByProject, source_indicator=use_source_indicator)

    # build the project craft actual vs revised file
    projectCraftData = buildProjectCraftFile(use_source_indicator)
    # remove branch and profit center type, as these will be calculated later
    projectCraftData.remove_data_columns([branchCol, profitCenterTypeCol])
    output_data.join_simple_inplace(projectCraftData, source_indicator=use_source_indicator)

    # break out any calculated fields as needed
    output_data.split_project_number()

    resultFile = generated_folder.save_data("All Data", output_data)
    resultFile.save_both = True
    return output_data


def buildProjectCraftEmployeeActualsFile():
    # read in the actuals data by month
    # note that want to know the actual values by craft
    projectCraftEmployeeActuals = project_folder.get_data_source("Project Craft Employee Actual Amount and Hours").get_data()
    projectCraftEmployeeActuals.add_dimensions([employeeIdCol, projectCol, yearCol, monthCol, craftCol])
    projectCraftEmployeeActuals.auto_assign_measures()
    return projectCraftEmployeeActuals


def buildProjectCraftRevisedFile():
    # read in the project & craft level data
    projectCraftFile = project_folder.get_data_source("Project Craft Revised Data")
    projectCraftData = projectCraftFile.get_data()
    projectCraftData.add_dimensions([projectCol, craftCol, "Customer ID", "Master Customer Name", "Customer Name"])
    projectCraftData.auto_assign_measures()
    return projectCraftData


def buildProjectCraftFile(use_source_indicator=False):
    projectCraftEmployeeActuals = buildProjectCraftEmployeeActualsFile()
    # perform filtering
    # note that do here, as will join into this data file, as should always need actuals for people to be working on a project
    # keep only the projects that belong to a department
    projectCraftEmployeeActuals.filter("Branch").is_in(["ATL", "CLT"])
    # remove before the injuries were really tracked
    projectCraftEmployeeActuals.filter_remove_before_year(2012)
    # remove where there isn't an employee
    projectCraftEmployeeActuals.filter(employeeIdCol).is_not_equal_to("ua")
    # remove where doesn't have a project number
    projectCraftEmployeeActuals.filter(projectCol).is_not_equal_to("00000000")
    # only keep actual hours and rework cost
    # also, remove the employee ID before aggregation by craft and project, so that isn't aggregated later
    projectCraftEmployeeActuals.remove_data_columns(["Actual Cost", "Rework Hours", employeeIdCol])
    # tabulate by project and craft
    projectCraftActualsFile = projectCraftEmployeeActuals.copy("Project and Craft by Month")
    projectCraftActualsFile.rename_measures("Craft ")
    # cumulative sum of the cost, hours, and rework for each project & craft by year and month
    projectCraftActualsFile.cumsum([projectCol, craftCol], [yearCol, monthCol], None, output_prefix="JTD ")
    # calculate by month
    dataByProjectMonth = projectCraftEmployeeActuals.copy("Project and Craft by Month")
    dataByProjectMonth.rename_measures("Project ")
    # cumulative sum of the cost, hours, and rework for each project & craft by year and month
    dataByProjectMonth.cumsum([projectCol], [yearCol, monthCol], None, output_prefix="JTD ")
    # join the project and craft data
    projectCraftActualsFile.join_simple_inplace(dataByProjectMonth, source_indicator=use_source_indicator)

    # calculate the revised (budget) data
    projectCraftRevisedFile = buildProjectCraftRevisedFile()
    projectCraftRevisedFile.name = "Revised Craft"
    # remove where doesn't have a project number
    projectCraftRevisedFile.filter(projectCol).is_not_equal_to("00000000")
    # remove projected data
    projectCraftRevisedFile.remove_data_columns(["Projected Hours", "Projected Cost"])
    # create a copy
    projectRevisedData = projectCraftRevisedFile.copy("Revised Project")
    projectRevisedData.keep_dimensions([projectCol])
    # rename the craft columns in the original frame
    projectCraftRevisedFile.rename_measures(prefix="Craft ")
    projectRevisedData.rename_measures(prefix="Project ")
    # merge the data back together
    projectCraftRevisedFile.join_simple_inplace(projectRevisedData, source_indicator=use_source_indicator)

    # only keep data for which there is actual data
    # note that there may not be a revised budget (especially in cases of rework on a craft)
    # as the budget may not have been created appropriately
    actualsAndRevised = projectCraftActualsFile.join_simple(projectCraftRevisedFile, JoinType.LEFT, source_indicator=use_source_indicator)

    # create the percentages of the budget, valued from 0 - 100 (or higher if needed)
    # iterate through all of the permutations of the values given
    for valueCol in ["Actual Hours", "Rework Cost"]:
        for metricType in ["Craft", "Project"]:
            normalizerCol = metricType + " Revised " + ("Hours" if valueCol.endswith("Hours") else "Cost")
            for value_prefix in [metricType, "JTD " + metricType]:
                sourceValCol = value_prefix + " " + valueCol
                calc_col_name = "% Budget " + sourceValCol
                # send to a high number if doesn't have a budget set.
                actualsAndRevised.add_calculated_field(calc_col_name, [sourceValCol, normalizerCol], lambda n, d: n / d * 100, 10)

    project_folder.save_data("Project Actual and Revised by Employee Craft Calculated", actualsAndRevised)

    return actualsAndRevised


rolling_dim_cols = [foremanIdCol, employeeIdCol, projectCol, craftCol, weekCol]
rolling_injury_metric_cols = [numInjuriesCol, numRecordablesCol]
rolling_num_entries_col = "# Entries"
rolling_window_num_weeks = 6
rolling_window_prediction_weeks = 3


def calculateBudgetPercentage(actual, revised):
    return actual / revised * 100


def get_prior_monday(given_date: date):
    # get the most recent Monday - Monday should have a weekday of 0
    # Subtract of the weekday to get back to Monday
    return given_date - timedelta(days=given_date.weekday())


def getSafetyDataColumnTypes():
    safetyDataOptions = CSVParserOptions()
    safetyDataOptions.date_columns += ['Week of Injury', 'Week', 'MostRecentHireDate', 'PositionStartDate', 'PositionEndDate']
    floatDataType = "Float64"
    intDataType = floatDataType  # "Int32", removed to hopefully remove large memory errors
    obsDataType = floatDataType  # "uint16", removed to hopefully remove large memory errors
    categoryType = "category"
    safetyDataOptions.column_types = {'ForemanEmpID': categoryType,
                                      'CrewEmpID': categoryType,
                                      'ProjectID': categoryType,
                                      'Craft': categoryType,
                                      'Month': categoryType,
                                      'Year': categoryType,
                                      'CrewHours': floatDataType,
                                      'JTDHours': floatDataType,
                                      'JTDWeeksOnProject': intDataType,
                                      'FMNHours': floatDataType,
                                      'FMNJTDHours': floatDataType,
                                      'FMNJTDWeeksOnProject': intDataType,
                                      'SafetyObservations': obsDataType,
                                      'TotalNumForemanObs': obsDataType,
                                      'PTPs': obsDataType,
                                      'ATRisk': obsDataType,
                                      'Training': obsDataType,
                                      'CustomerID': categoryType,
                                      'CustomerName': categoryType,
                                      'MasterCustomerName': categoryType,
                                      'RevisedContractAmount': floatDataType,
                                      'BuildingType': categoryType,
                                      'Closed': 'Int8',
                                      'Position': categoryType,
                                      'PositionType': categoryType,
                                      'YearsAtMcK': floatDataType,
                                      'YearsSinceLastHired': floatDataType,
                                      'YearsInPosition': floatDataType,
                                      'YearsInPositionSinceLastHired': floatDataType,
                                      'YearsInPositionType': floatDataType,
                                      'YearsInPositionTypeSinceLastHired': floatDataType,
                                      'FMNPosition': categoryType,
                                      'FMNPositionType': categoryType,
                                      'FMNYearsAtMcK': floatDataType,
                                      'FMNYearsSinceLastHired': floatDataType,
                                      'FMNYearsInPosition': floatDataType,
                                      'FMNYearsInPositionSinceLastHired': floatDataType,
                                      'FMNYearsInPositionType': floatDataType,
                                      'FMNYearsInPositionTypeSinceLastHired': floatDataType,
                                      'TotalProjectHours': floatDataType,
                                      'ProjectApprenticeHrs': floatDataType,
                                      'ProjectClassifiedWorkerHrs': floatDataType,
                                      'ProjectForemanHrs': floatDataType,
                                      'ProjectHelperHrs': floatDataType,
                                      'ProjectJourneyManHrs': floatDataType,
                                      'ProjectTechnicianHrs': floatDataType,
                                      'ProjectTradesmanHrs': floatDataType,
                                      'ProjectOfficeHrs': floatDataType,
                                      'ProjectOtherHrs': floatDataType,
                                      'TotalCraftHours': floatDataType,
                                      'CraftApprenticeHrs': floatDataType,
                                      'CraftClassifiedWorkerHrs': floatDataType,
                                      'CraftForemanHrs': floatDataType,
                                      'CraftHelperHrs': floatDataType,
                                      'CraftJourneyManHrs': floatDataType,
                                      'CraftTechnicianHrs': floatDataType,
                                      'CraftTradesmanHrs': floatDataType,
                                      'CraftOfficeHrs': floatDataType,
                                      'CraftOtherHrs': floatDataType,
                                      'ProjectReworkCost': floatDataType,
                                      'ProjectReworkHours': floatDataType,
                                      'RevisedProjectHours': floatDataType,
                                      'RevisedProjectCost': floatDataType,
                                      '% Budget Project Actual Hours': floatDataType,
                                      '% Budget Project Rework Cost': floatDataType,
                                      '% Budget Project Rework Hours': floatDataType,
                                      'JTDProjectReworkCost': floatDataType,
                                      'JTDProjectReworkHours': floatDataType,
                                      'JTDProjectActualHours': floatDataType,
                                      '% Budget JTD Project Actual Hours': floatDataType,
                                      '% Budget JTD Project Rework Cost': floatDataType,
                                      '% Budget JTD Project Rework Hours': floatDataType,
                                      'CraftReworkCost': floatDataType,
                                      'CraftReworkHours': floatDataType,
                                      'RevisedCraftCost': floatDataType,
                                      'RevisedCraftHours': floatDataType,
                                      '% Budget Craft Actual Hours': floatDataType,
                                      '% Budget Craft Rework Cost': floatDataType,
                                      '% Budget Craft Rework Hours': floatDataType,
                                      'JTDCraftReworkCost': floatDataType,
                                      'JTDCraftReworkHours': floatDataType,
                                      'JTDCraftActualHours': floatDataType,
                                      '% Budget JTD Craft Actual Hours': floatDataType,
                                      '% Budget JTD Craft Rework Cost': floatDataType,
                                      '% Budget JTD Craft Rework Hours': floatDataType,
                                      'MasterInjuryID': "string",
                                      '# Recordables': floatDataType,
                                      '# Injuries': floatDataType,
                                      'InjuredEmployee': floatDataType,
                                      'Affected Body Part': categoryType,
                                      'Cause of Injury': categoryType,
                                      'Classification': categoryType,
                                      'EmpCraft': categoryType}
    return safetyDataOptions


def createRollingWindowData(originalDataset: DataFrame, weekOffsetIterator, newName: str, retain_original_week: bool = False):
    datasets = []
    weekAttr = "Week"
    metricWeekAttr = "Analysis Week"
    for weekOffset in weekOffsetIterator:
        allDataCopy = originalDataset.copy()
        allDataCopy[metricWeekAttr] = allDataCopy[weekAttr] + pd.DateOffset(days=7 * weekOffset)
        allDataCopy[rolling_num_entries_col] = 1
        datasets.append(allDataCopy)
    rollingWindowData = DataCollection(pd.concat(datasets), newName)
    # remove the week item, and set the analysis seek to the new week
    if retain_original_week:
        rollingWindowData.rename_column(weekAttr, "Original Week")
    else:
        rollingWindowData.remove_data_column(weekAttr)
    rollingWindowData.rename_column(metricWeekAttr, weekCol)
    # add the dimensions
    rollingWindowData.add_dimensions(rolling_dim_cols, warn_if_not_found=False)
    return rollingWindowData


def getRecentSafetyDataForAnalysis(given_date: date = None):
    # Converts the current dataset (looking back a few weeks) to the format that can be analyzed by the model
    # first, set the value for all weeks to the analysis date
    # default to today if a date is not given
    if given_date is None:
        given_date = date.today()
    # find the prior Monday, as that will be where all dates are relative to.
    prior_monday = get_prior_monday(given_date)
    # set the end date to the prior Sunday
    end_date = prior_monday - timedelta(days=1)
    # set the start date to the Monday from a few weeks back
    start_date = get_prior_monday(prior_monday - timedelta(days=(rolling_window_num_weeks * 7)))
    # get the results
    originalDataset = get_safety_data_results(start_date, end_date)
    allDataCopy = originalDataset.copy()
    return allDataCopy


def getTrainingSafetyDataForAnalysis(startYear):
    def createDatasetSinceYear():
        # start with the first monday of the year, starting with 1/7/XX and looking back to the first monday
        # this is because if the 1/7 is a Monday, then will be the first Monday of the year.
        # Otherwise, the first Monday must be 1/1 - 1/6.
        start_date = get_prior_monday(date(startYear, 1, 7))
        # find the prior Monday, as that will be where all dates are relative to.
        prior_monday = get_prior_monday(date.today())
        # set the end date to the prior Sunday
        end_date = prior_monday - timedelta(days=1)
        originalDataset = get_safety_data_results(start_date, end_date)
        ds = DataCollection(originalDataset)
        # rename any columns as needed
        rename_columns_from_dynamics_data(ds)
        # round the data (for more succinct file storage)
        round_rolling_window_data(ds)
        # generate the output file
        return ds

    # grab the data, creating based on the given metrics if needed
    dataSinceYear = generated_folder.get_data_source(f"Safety AI {startYear} - Present",
                                                     creator_func=createDatasetSinceYear).get_data(parserOptions=getSafetyDataColumnTypes())
    return dataSinceYear


def getRecentSafetyDataForAnalysis_SAMPLE():
    # read in the sample file
    sourceFile = generated_folder.get_sub_path("Example Rolling Safety Window Data.xlsx")
    allData = DataTable(sourceFile, auto_populate=False, sheet_name="Sheet1")
    # the sample file contains data from 2/1 (a Thursday), to 3/13 (a Wednesday)
    # because of this, only keep data between 2/5/2024 (Monday) and 3/10/2024 (Sunday)
    weekData = pd.to_datetime(allData.df[weekCol]).dt.date
    startKeepDate = date(2024, 2, 5)
    endKeepDate = date(2024, 3, 10)
    allData.filter_where((startKeepDate <= weekData) & (weekData <= endKeepDate))
    return allData.df


def createOngoingRollingWindowData(given_date: date = None, continue_with_sample=False):
    # try to get the current data; fallback to sample if needed
    try:
        allDataCopy = getRecentSafetyDataForAnalysis(given_date)
    except Exception as ex:
        # if should continue with the sample (if for example, running in debug mode), then create the sample
        if continue_with_sample:
            diagnostic_log_message(f"Problem running on ongoing basis: {ex}")
            allDataCopy = getRecentSafetyDataForAnalysis_SAMPLE()
        else:
            raise ex
    # set the week to the last week within the analysis set, which will be the last included monday
    # this is so that is aggregated for the given week, like how the training data is aggregated by 6-week rolling window.
    last_week_in_set = allDataCopy[weekCol].max()
    allDataCopy[weekCol] = last_week_in_set
    # add the number of entries
    allDataCopy[rolling_num_entries_col] = 1
    # next, create the data collection
    rollingWindowData = DataCollection(allDataCopy, "Current Results")
    # rename the columns
    rename_columns_from_dynamics_data(rollingWindowData)
    # add the dimensions
    rollingWindowData.add_dimensions(rolling_dim_cols)
    # finish off by consolidating the data
    applicableData = convert_rolling_data_to_applicable_data_with_recent_injuries(rollingWindowData, rolling_window_num_weeks)
    add_in_rolling_window_calculated_fields(applicableData)
    outputSet = createAuxilliaryColumnsForRollingData(applicableData)
    keep_large_projects_and_group_other_by_department(outputSet)
    return outputSet


position_cols = ["Apprentice", "ClassifiedWorker", "Foreman", "Helper", "JourneyMan", "Technician", "Tradesman", "Office", "Other"]
totalProjectHoursCol = "TotalProjectHours"
totalCraftHoursCol = "TotalCraftHours"
summary_project_hours_cols = [totalProjectHoursCol, "ProjectReworkHours"]
summary_project_cost_cols = ["ProjectReworkCost"]
summary_craft_hours_cols = [totalCraftHoursCol, "CraftReworkHours"]
summary_craft_cost_cols = ["CraftReworkCost"]
jtd_project_hours_cols = ["JTDProjectReworkHours", "JTDProjectActualHours"]
jtd_project_cost_cols = ["JTDProjectReworkCost"]
jtd_craft_hours_cols = ["JTDCraftReworkHours", "JTDCraftActualHours"]
jtd_craft_cost_cols = ["JTDCraftReworkCost"]


def rename_columns_from_dynamics_data(sourceFile: DataCollection):
    # rename the common columns
    sourceFile.rename_column("CrewHours", hoursCol)
    sourceFile.rename_column("ProjectID", projectCol)
    sourceFile.rename_column("ForemanEmpID", foremanIdCol)
    sourceFile.rename_column("CrewEmpID", employeeIdCol)
    sourceFile.rename_column("Craft", craftCol)
    # some weeks start on a Tuesday, whereas other weeks start on a Monday
    # transition all weeks to start on the Monday
    weekData = pd.to_datetime(sourceFile.df["Week"])
    sourceFile.df["Week"] = weekData + pd.to_timedelta(-weekData.dt.dayofweek, unit="D")


def round_rolling_window_data(data: DataCollection):
    # round percentages to 2 decimal places (reported from 0 - 100)
    percent_col_text = "%"
    data.round_columns_containing(2, percent_col_text)
    # round hours to 1 decimal place, excluding percentages
    data.round_columns_containing(1, "Hours", percent_col_text)
    data.round_columns_containing(1, "Hrs", percent_col_text)
    # round weeks on project to 1 decimal place
    data.round_columns_containing(1, "Weeks")
    # round observations to 1 decimal place
    data.round_columns_containing(1, "Obs")
    # round costs to 2 decimal places, excluding any percentage columns
    data.round_columns_containing(1, "Cost", percent_col_text)
    # round tenure to 2 decimal places
    data.round_columns_containing(1, "Years")


def add_in_rolling_window_calculated_fields(data: DataCollection):
    # add in calculated fields
    valueWhenFailed = numpy.nan
    # add in the percentage of the total
    for p in position_cols:
        data.add_calculated_field(f"Project {p} %", [f"Project{p}Hrs", totalProjectHoursCol],
                                  calculateBudgetPercentage, replacement_value=valueWhenFailed)
        data.add_calculated_field(f"Craft {p} %", [f"Craft{p}Hrs", totalCraftHoursCol],
                                  calculateBudgetPercentage, replacement_value=valueWhenFailed)
    # add percentage of craft budget hours and cost
    data.add_calculated_field("% Budget Craft Cost Budget", ["RevisedCraftCost", "RevisedProjectCost"], calculateBudgetPercentage,
                              replacement_value=valueWhenFailed)
    data.add_calculated_field("% Budget Craft Hours Budget", ["RevisedCraftHours", "RevisedProjectHours"], calculateBudgetPercentage,
                              replacement_value=valueWhenFailed)
    # add % of budget for rework metrics
    for c in (summary_project_hours_cols + jtd_project_hours_cols):
        data.add_calculated_field("% Budget " + c, [c, "RevisedProjectHours"], calculateBudgetPercentage, replacement_value=valueWhenFailed)
    for c in (summary_project_cost_cols + jtd_project_cost_cols):
        data.add_calculated_field("% Budget " + c, [c, "RevisedProjectCost"], calculateBudgetPercentage, replacement_value=valueWhenFailed)
    for c in (summary_craft_hours_cols + jtd_craft_hours_cols):
        data.add_calculated_field("% Budget " + c, [c, "RevisedCraftHours"], calculateBudgetPercentage, replacement_value=valueWhenFailed)
    for c in (summary_craft_cost_cols + jtd_craft_cost_cols):
        data.add_calculated_field("% Budget " + c, [c, "RevisedCraftCost"], calculateBudgetPercentage, replacement_value=valueWhenFailed)


def aggregate_rolling_data_in_lossless_format(data: DataCollection, columns: str | list[str], num_times_row_repeats: int):
    # set up the data so that when is aggregated in a rolling format, can be summed to equal the original value
    # For example, if rolling the following into 3 week windows:
    #   - Week 1: 5
    #   - Week 2: 6
    #   - Week 3: 3
    #   - Week 4: 2
    #   - Week 5: 4
    #   - Sum: 20
    # Then when summing the output of the rolling windows becomes the result of the sum of the original
    # This is particularly important when data may be reported again (like hours or injuries)
    # The sum of the output of rolling windows becoming the result of the sum of the original can only occur if each row
    # only appears once, or is weighted as if it appears once.
    # Can't use an average, because the rolling windows would become the following:
    #   - Week 1: 5 (5)
    #   - Week 2: 5.5 [(5 + 6) / 2]
    #   - Week 3: 4.666 [(5 + 6 + 3) / 3]
    #   - Week 4: 3.666 [(6 + 3 + 2) / 3]
    #   - Week 5: 3 [(3 + 2 + 4) / 3]
    #   Sum: 21.8332
    # Note that this is because 5 is weighted as 1 (in the first week), 0.5 (in the second week), and 0.333 (in the third week)
    # Instead, can divide all values by the window size (effectively the same as adding 0 buffer weeks as Week -1 and Week 0)
    # This will ensure that the values are all reported as the same amount
    # First, go through each column, and divide the value by the window size
    column_list = convert_to_iterable(columns, str)
    for c in column_list:
        data.df[c] = data.df[c] / num_times_row_repeats
    # next, set up the columns to sum the values
    data.add_measures(column_list, aggregation_method=DataColumnAggregation.SUM)


def convert_rolling_data_to_applicable_data(data: DataCollection, num_times_row_repeats: int):
    # aggregate all of the fields
    # use average, so that the input and output have similar values
    rowAggrMethod = DataColumnAggregation.AVG
    # sum up the number of entries
    data.add_measure(rolling_num_entries_col, aggregation_method=DataColumnAggregation.SUM)
    # aggregate the hours using a method so that the sum of result exactly equals the sum of the input amount
    # for now, only need to do the crew hours (as all other columns will be used as metrics, but shouldn't be aggregated)
    aggregate_rolling_data_in_lossless_format(data, hoursCol, num_times_row_repeats)
    # aggregate the hours and observations
    data.add_measures(["FMNHours", "SafetyObservations", "TotalNumForemanObs", "PTPs", "ATRisk", "Training"],
                      aggregation_method=rowAggrMethod)
    # aggregate JTD metrics
    data.add_measures(["JTDHours", "JTDWeeksOnProject", "FMNJTDHours", "FMNJTDWeeksOnProject"],
                      aggregation_method=DataColumnAggregation.MAX)
    # aggregate "dimensions" that should be treated as measures
    data.add_measures(["CustomerID", "CustomerName", "MasterCustomerName", "BuildingType", "Position", "PositionType",
                       "FMNPosition", "FMNPositionType", "Closed"],
                      aggregation_method=DataColumnAggregation.MAX)
    # aggregate tenure metrics - use the minimum for the least amount of time
    data.add_measures(["YearsAtMcK", "YearsSinceLastHired", "YearsInPosition", "YearsInPositionSinceLastHired",
                       "YearsInPositionType", "YearsInPositionTypeSinceLastHired",
                       "FMNYearsAtMcK", "FMNYearsSinceLastHired", "FMNYearsInPosition", "FMNYearsInPositionSinceLastHired",
                       "FMNYearsInPositionType", "FMNYearsInPositionTypeSinceLastHired"], aggregation_method=DataColumnAggregation.MIN)
    # aggregate data for number of hours on project / craft

    # add the total values
    data.add_measures([f"Project{p}Hrs" for p in position_cols], aggregation_method=rowAggrMethod)
    data.add_measures([f"Craft{p}Hrs" for p in position_cols], aggregation_method=rowAggrMethod)
    # add injuries from the past few weeks
    data.add_measures(rolling_injury_metric_cols, aggregation_method=rowAggrMethod)
    # aggregate data for revised project rework hours and cost
    data.add_measures(["RevisedProjectHours", "RevisedProjectCost", "RevisedCraftCost", "RevisedCraftHours"],
                      aggregation_method=DataColumnAggregation.MAX)

    # aggregate data for rework cost & hours
    data.add_measures(summary_project_hours_cols + summary_project_cost_cols + summary_craft_hours_cols + summary_craft_cost_cols,
                      aggregation_method=rowAggrMethod)
    # add in the job to date project and craft rework metrics
    data.add_measures(jtd_project_hours_cols + jtd_project_cost_cols + jtd_craft_hours_cols + jtd_craft_cost_cols,
                      aggregation_method=DataColumnAggregation.MAX)

    # rename columns that are not well written
    data.rename_column("ATRisk", "# At Risk Obs")
    data.rename_column("Training", "# Training Obs")
    data.rename_column("SafetyObservations", "# Safety Obs")
    data.rename_column("PositionType", positionTypeCol)
    for m in rolling_injury_metric_cols:
        data.rename_column(m, "Recent " + m)

    # combine all records together
    diagnostic_log_message("Starting the rolling window consolidation")
    data.consolidate_data_and_replace()
    diagnostic_log_message("Completed the rolling window consolidation")
    return data


def add_recent_injuries_to_applicable_data(data: DataCollection):
    # add in the project and craft recent injuries
    recentInjuriesByProjectAndCraft = data.copy("Recent Craft Injuries")
    recentInjuriesByProjectAndCraft.keep_data_columns([projectCol, craftCol, weekCol, "Recent # Recordables", "Recent # Injuries"])
    recentInjuriesByProjectAndCraft.consolidate_data_and_replace()
    # make a project copy before renaming measures
    recentInjuriesByProject = recentInjuriesByProjectAndCraft.copy("Recent Project Injuries")
    recentInjuriesByProject.remove_data_column(craftCol)
    recentInjuriesByProject.consolidate_data_and_replace()
    # rename the injury metrics in both
    recentInjuriesByProjectAndCraft.rename_measures("Craft ")
    recentInjuriesByProject.rename_measures("Project ")
    # join both back into the data
    data.join_simple_inplace(recentInjuriesByProjectAndCraft)
    data.join_simple_inplace(recentInjuriesByProject)
    return data


def convert_rolling_data_to_applicable_data_with_recent_injuries(data: DataCollection, num_times_row_repeats: int):
    applicable_data = convert_rolling_data_to_applicable_data(data, num_times_row_repeats)
    return add_recent_injuries_to_applicable_data(applicable_data)


def createAuxilliaryColumnsForRollingData(data: DataCollection):
    # add the month based on the week reported (which will be the most recent week)
    data.split_date("Week", remove_date=False, perform_consolidation=False)
    data.split_project_number()
    return data


def setUpRollingWindowDatasetWithDimensions(ds: DataCollection):
    # if doesn't have any columns within the lookup, then set up the dimensions
    if len(ds.column_lookup) == 0:
        ds.set_up_from_dimensions(rolling_dim_cols + [yearCol, monthCol, branchCol, profitCenterTypeCol, profitCenterIdCol], False)
        # rollingData.auto_assign_measures()


def getNWeekTrainingData(startingYear: int, numWeeks: int = rolling_window_num_weeks, predictionInjuryWeeks: int = rolling_window_prediction_weeks):
    def createDataForRollingWindowDataset():
        # load in the original data source
        sourceFile = getTrainingSafetyDataForAnalysis(startingYear)
        setUpRollingWindowDatasetWithDimensions(sourceFile)

        # TEST: For now, only look at project 15006007
        # sourceFile.filter(projectCol).is_equal_to("15006007")
        round_rolling_window_data(sourceFile)
        # only include construction data (which will be for ATL and CLT)
        sourceFile.split_project_number(projectCol)
        sourceFile.filter(branchCol).is_in(["ATL", "CLT", "ATL4"])

        # drop columns that won't use inside the analysis
        sourceFile.remove_data_columns(["MasterInjuryID", "InjuredEmployee", "Affected Body Part", "Cause of Injury",
                                        "Classification", "EmpCraft", "Week of Injury",
                                        # note that month, year, branch, and profit center data will be recreated once the
                                        # rolling data is generated
                                        monthCol, yearCol, branchCol, profitCenterTypeCol,
                                        'MostRecentHireDate', 'PositionStartDate', 'PositionEndDate'], False)

        # create the rolling windows
        sourceFile.remove_unused_categories()
        sourceFile.alphabetize_categories()
        allData_df = sourceFile.df
        # analyze the data on a rolling basis, and include how many injuries occurred after that timeframe
        # most columns will be aggregated over the window by sum
        # some columns (like JTD columns, or tenure columns) can be aggregated by either max or average
        # will aggregate the last N weeks.
        # Thus, start with 0 (current week), and count forwards N weeks
        # (since the data will be used by the next N weeks, as they are looking back N weeks)
        # don't remove the injury columns, since can use those to determine likelihood of next injury

        rolling_data = createRollingWindowData(allData_df, range(0, numWeeks), "Rolling Data")
        applicable_data = convert_rolling_data_to_applicable_data(rolling_data, numWeeks)
        data = add_recent_injuries_to_applicable_data(applicable_data)

        # now, combine the injury datasets - include any injury values that will sum up, and any identifying row columns (for merging later)
        # weight the injury by the number of hours within each bucket
        # this allows to not include foreman, craft, and possibly project (helping to not exclude important columns)
        injury_dim_columns = [employeeIdCol, projectCol, weekCol]
        injury_columns_to_keep = rolling_injury_metric_cols + injury_dim_columns
        injury_data = allData_df[injury_columns_to_keep]
        # filter the injuries to only data that contains injuries
        injury_data = injury_data[injury_data[numInjuriesCol] > 0]
        # add in the injury row ID (so can track how many rows actually were kept for this injury)
        injurySourceIdCol = "Injury Source ID"
        injury_data[injurySourceIdCol] = range(len(injury_data))
        # since are trying to predict injuries in the future, then a row will apply to the last few weeks
        # since the row is included within the "lookahead" of those weeks
        # also, skip the current week, since those metrics will be included within the data
        injuriesAfterRollingWindow = createRollingWindowData(injury_data, range(-predictionInjuryWeeks, 0), "Rolling Injury Data")
        aggregate_rolling_data_in_lossless_format(injuriesAfterRollingWindow, rolling_injury_metric_cols, predictionInjuryWeeks)
        # add the injury column ID as a dimension, so is not lost within the consolidation
        injuriesAfterRollingWindow.add_dimensions(injurySourceIdCol)
        injuriesAfterRollingWindow.consolidate_data_and_replace()
        # since the rolling metrics are consolidated via different columns,
        # then consolidate the rolling metrics with the injury dimensions to get the total number of hours for records associated with each injury,
        # then join in the injuries to the rolling metrics, weighted by hours of each row divided by total hours for the injury
        dataAggregatedLikeInjuries = data.copy("Rolling Data aggregated like Injury Data")
        dataAggregatedLikeInjuries.keep_data_columns(injury_dim_columns + [hoursCol])
        totalHoursPerInjuryCol = "Hours Included within Injury Bucket"
        # set up the data aggregated like injuries, so know how many records the injury is reported on
        dataAggregatedLikeInjuries.rename_column(hoursCol, totalHoursPerInjuryCol)

        distributeMissingInjuryWeekWeights = True
        if distributeMissingInjuryWeekWeights:
            # if an employee gets injured at the start of the project,
            # then the injury may not "report back" correctly, as there aren't any weeks to report back to
            # For example, if reporting back 6 weeks, and an injury occurs within the 3rd week, then will only report on weeks 1 and 2 (aka 2 of 6 weeks)
            # Therefore, also weight by the number of records that the injury reports back to (in this case, multiple each by 3 = 6 / 2)
            # However, if a second injury occurs on the 5th week, then will only report on weeks 1, 2, 3, and 4 (aka 4 of 6 weeks)
            # with each record being weighted by 1.5 (= 6 / 4)
            # For the first week, the injury will be equal to the spread-out injury from the 2nd week (aka 1/6) times the row weight (i.e., 3)
            # plus the spread-out injury from the 5th week (aka 1/6) times the row weight (i.e., 1.5)
            # Then, these weights can be added together (0.75 for week 1, 0.75 for week 2, 0.25 for week 3, and 0.25 for week 4),
            # with the sum equaling the original number of injuries (i.e., 2)
            # first, consolidate the rolling data (but without columns), so that know what data is being kept
            dataAggregatedLikeInjuriesWithoutHours = dataAggregatedLikeInjuries.copy()
            # consolidation shouldn't do anything, since are removing a measure
            dataAggregatedLikeInjuriesWithoutHours.remove_data_column(totalHoursPerInjuryCol, perform_consolidation=True)
            # merge in with the rolling injury data, to figure out which weeks are missing
            rollingInjuriesRemaining = injuriesAfterRollingWindow.join_simple(dataAggregatedLikeInjuriesWithoutHours, JoinType.INNER)
            remainingWeeksPerInjury = rollingInjuriesRemaining.df[injurySourceIdCol].value_counts()
            weightingFactor = predictionInjuryWeeks / rollingInjuriesRemaining.df[injurySourceIdCol].map(remainingWeeksPerInjury)
            for injury_col in rolling_injury_metric_cols:
                rollingInjuriesRemaining.df[injury_col] *= weightingFactor
            # use the rolling injuries remaining as the injuries after the rolling window
            injuriesAfterRollingWindow = rollingInjuriesRemaining

        # now that have used the injury source column, can remove from the analysis (and consolidate)
        injuriesAfterRollingWindow.remove_data_column(injurySourceIdCol, perform_consolidation=True)

        # now, join in the injury data, keeping all injuries (to see which don't match)
        injuriesWithTotalHours = injuriesAfterRollingWindow.join_simple(dataAggregatedLikeInjuries, JoinType.INNER)

        # now, join the injury data with all data, weighting the injuries by the number of hours in the record vs total hours on the injury
        # make sure to preserve any dimensions within the rolling data not used in the injury dimensions
        rollingDataWithSubsequentInjuries = data.join(injuriesWithTotalHours, JoinType.OUTER,
                                                      joining_dimensions=injury_dim_columns,
                                                      extra_dimensions_left=[c for c in rolling_dim_cols if c not in injury_dim_columns])

        # weight the injuries based on total hours, in case an injury (which is aggregated by limited dimensions) applies to multiple hours rows
        for injury_col in rolling_injury_metric_cols:
            # set the new injury value
            old_injuries = rollingDataWithSubsequentInjuries.df[injury_col]
            total_hours = rollingDataWithSubsequentInjuries.df[totalHoursPerInjuryCol]
            new_injuries = old_injuries * rollingDataWithSubsequentInjuries.df[hoursCol] / total_hours
            # for any division by zero or empty value problems, set the value to the original injury value
            # this is because likely didn't have a corresponding record within the rolling data when trying to figure total hours
            totalHoursAreZeroMask = total_hours == 0
            new_injuries[totalHoursAreZeroMask] = old_injuries[totalHoursAreZeroMask]
            # now, set the new injury value
            rollingDataWithSubsequentInjuries.df[injury_col] = new_injuries
        rollingDataWithSubsequentInjuries.remove_data_column(totalHoursPerInjuryCol)

        createAuxilliaryColumnsForRollingData(rollingDataWithSubsequentInjuries)

        # round the data as needed, to preserve memory when storing
        round_rolling_window_data(rollingDataWithSubsequentInjuries)
        # generate the output file
        return rollingDataWithSubsequentInjuries

    # grab the data, creating based on the given metrics if needed
    parser_options = CSVParserOptions()
    categoryType = "string"
    parser_options.column_types = {
        'CustomerID': categoryType,
        'CustomerName': categoryType,
        'MasterCustomerName': categoryType,
        'BuildingType': categoryType,
    }
    rollingData = generated_folder.get_data_source(f"Rolling Window Project Data {startingYear} {numWeeks}x{predictionInjuryWeeks}",
                                                   creator_func=createDataForRollingWindowDataset).get_data(parserOptions=parser_options)
    setUpRollingWindowDatasetWithDimensions(rollingData)
    # add in the calculated fields
    add_in_rolling_window_calculated_fields(rollingData)
    return rollingData


# used for projects that are considered "large"
large_project_id_attr = "LargeProjectId"


def keep_large_projects_and_group_other_by_department(ds: DataCollection):
    # If a project averages over XXX hours worked on it per week, then will retain the project ID as is
    # otherwise, will set the project to Small {Department} Projects
    # Note that if aggregated by rolling windows (i.e., 8 week window), then will use the average value
    # first, figure out average hours per project per week
    avg_hours_per_project_per_week = ds.copy()
    avg_hours_per_project_per_week.keep_data_columns([projectCol, weekCol, profitCenterIdCol, hoursCol])
    avg_hours_per_project_per_week.consolidate_data_and_replace()
    # next, decide which projects should be grouped by department
    group_by_department_attr = "Group by Department?"
    avg_hours = avg_hours_per_project_per_week.df[hoursCol]
    avg_hours_above_threshold = avg_hours > 500
    # next, create a new column to hold either the project ID or the department small projects name
    # start off with the department ID (since a lot of rows will be small projects), and then assign projects as necessary
    large_project_id = "Small Projects for Dept: " + avg_hours_per_project_per_week.df[profitCenterIdCol].astype("str")
    large_project_id.loc[avg_hours_above_threshold] = avg_hours_per_project_per_week.df[projectCol].loc[avg_hours_above_threshold]
    avg_hours_per_project_per_week.set_measure_data_column(large_project_id_attr, large_project_id, DataColumnAggregation.MIN)
    # remove the hours column (so that doesn't overwrite the prior data). Don't need to perform consolidation, since want to keep all rows
    avg_hours_per_project_per_week.remove_data_column(hoursCol, False)
    # next, join in the large project ID into the original data set
    ds.join_simple_inplace(avg_hours_per_project_per_week)
    # return the resulting data set
    return ds


if __name__ == "__main__":
    # get data for training
    # training_data = getNWeekTrainingData(2014, 8, 6)
    # getTrainingSafetyDataForAnalysis(2019)
    # getTrainingSafetyDataForAnalysis(2014)
    # getTrainingSafetyDataForAnalysis(2020)
    getTrainingSafetyDataForAnalysis(2021)

    data_for_predictions = createOngoingRollingWindowData(continue_with_sample=False)

    # buildProjectActualsWithEmployeeData()
    # build_employee_files()

    # save all folders and files
    root_folder.build_necessary_structure()
