from datetime import date
from abc import ABC, abstractmethod


class DateIteratorBase(ABC):
    # allows a user to move forward through dates
    def __init__(self, start_date: date, end_date: date):
        self.start_date = start_date
        self.end_date = end_date
        self.curr_date: date = start_date
        self.time_in_period: float = 0.

    @abstractmethod
    def move_forward(self) -> bool: ...


class MonthIterator(DateIteratorBase):
    def __init__(self, start_date: date, end_date: date):
        super().__init__(start_date, end_date)
        self.month = start_date.month
        self.year = start_date.year
        self.curr_date = start_date
        # initialize the next values to the current values
        # this way, on first iteration of move forward, then stays at current position
        self._next_date = start_date
        self._next_month = self.month
        self._next_year = self.year

    def _calculate_next_date(self):
        # calculate the next month and year
        if self.month == 12:
            self._next_month = 1
            self._next_year = self.year + 1
        else:
            self._next_month = self.month + 1
        self._next_date = date(self._next_year, self._next_month, 1)
        if self._next_date > self.end_date:
            self._next_date = self.end_date

    def skip_forward_to(self, year: int, month: int):
        # only skip forward if not already there

        self.curr_date = date(year, month, 1)
        self.year = year
        self.month = month
        self._next_year = year
        self._next_month = month
        self._next_date = self.curr_date

    def move_forward(self) -> bool:
        # copy the values from the next value
        # calculate the next time
        self.month = self._next_month
        self.year = self._next_year
        self.curr_date = self._next_date
        # make sure that haven't passed the end date
        if self.curr_date >= self.end_date:
            return False
        # get the next value
        self._calculate_next_date()
        # calculate the time until the next date
        self.time_in_period = (self._next_date - self.curr_date).days / 365
        return True
