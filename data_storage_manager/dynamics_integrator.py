"""Database logic
Handles getting data from DYNAMICSSLAPP view
"""
from SecretKeeper import Keeper
from datetime import date
from overall_helpers import diagnostic_log_message
import sqlalchemy
import pandas as pd
import pyodbc


def get_preferred_connector():
    # get the list of the drivers
    potential_drivers = pyodbc.drivers()

    # find the "bset" driver that uses ODBC
    # assume that the connectors will use increasing ODBC number (like ODBC Driver 17 and ODBC Driver 18),
    # so can just find the "biggest" and therefore most recent driver.
    selectedDriver = None
    for d in potential_drivers:
        if d.startswith("ODBC Driver"):
            if selectedDriver is None or d > selectedDriver:
                selectedDriver = d
    if selectedDriver is not None:
        return selectedDriver
    # next, look for any drivers that contain SQL Server
    for d in potential_drivers:
        if "SQL Server" in d:
            return d

    # otherwise, just use the first driver, and hope that it works
    return potential_drivers[0]


def format_date_for_sql(given_date: date):
    # return in the format YYYY-MM-DD, padding month and day with zeros
    # also, wrap within quotes
    return f'"{given_date.year}-{given_date.month:02}-{given_date.day:02}"'


def get_safety_data_results(first_week_start_date: date, last_week_end_date: date, save_path: str = None):
    # the stored procedure will return partial results, unless the start and end dates exactly align with the weeks
    # To not return partial weeks, the start date should be a Monday, and the end date should be a Sunday
    # connect to the database
    diagnostic_log_message(f"Connecting to safety data stored procedure from {first_week_start_date} to {last_week_end_date}")
    # create the stored procedure call
    # use no counts in case SQL Server returns counts with each of the reads.
    sql_start_date = format_date_for_sql(first_week_start_date)
    sql_end_date = format_date_for_sql(last_week_end_date)
    stored_proc_call = f"SET NOCOUNT ON; EXEC [MWA].[Queries].SafetyTrend @StartDate={sql_start_date}, @EndDate={sql_end_date};"

    # read the results into pandas dataframe
    sql_user, sql_pass = Keeper.get_credentials("VKP3GSaPfUrBPEMFb8fbeA")
    connection_str = f'DRIVER={{{get_preferred_connector()}}};SERVER=MCKATLSQL;PORT=1433;DATABASE=MWA;UID={sql_user};PWD={sql_pass};TrustServerCertificate=yes;Encrypt=yes;'
    connection_url = sqlalchemy.engine.URL.create("mssql+pyodbc", query={"odbc_connect": connection_str})
    engine = sqlalchemy.create_engine(connection_url)
    df = pd.read_sql_query(stored_proc_call, engine)
    diagnostic_log_message("Completed reading safety data stored procedure")
    # lastly, write to a CSV (if given)
    if save_path is not None:
        df.to_csv(save_path)
    return df
