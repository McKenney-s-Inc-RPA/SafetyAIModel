from __future__ import annotations
from typing import Dict
from math import log10, ceil, inf, nan
from overall_helpers import can_be_int
from numpy import inf as np_inf, nan as np_nan


# these models are to put the tree into an understandable / usable format
def string_has_value(curr_text: str | None):
    return False if curr_text is None else len(curr_text) > 0


def remove_decimal_if_needed(given_value):
    if can_be_int(given_value):
        return int(given_value)
    else:
        return given_value


def str_with_sig_figs(given_number: float, sig_figs: int = 5):
    # gets the number of digits before the decimal point
    # for example, 1023 has 4 digits, and has a log10 of 3.01
    abs_value = abs(given_number)
    if abs_value == 0.:
        return "0"
    maxDigit = int(ceil(log10(abs_value)))
    # go to the digit that want to round to
    # for example, if wanting 3 digits from 1023, then would want to round to the nearest 10 (or -1)
    roundDigit = maxDigit - sig_figs
    roundedNumber = round(given_number, -roundDigit)
    if roundDigit >= 0:
        # means that are wanting to round to a number greater than 1
        # because of this, drop the decimal point
        return str(int(roundedNumber))
    else:
        # remove the 0 after the decimal place if needed
        return str(remove_decimal_if_needed(roundedNumber))


def get_short_text_for_large_number(given_value, num_sig_figs=3):
    if given_value == inf or given_value == np_inf \
            or given_value == -inf or given_value == -np_inf \
            or given_value == nan or given_value == np_nan:
        return str(given_value)
    if given_value < 1e3:
        return str_with_sig_figs(given_value, num_sig_figs)
    elif given_value < 1e6:
        return str_with_sig_figs(given_value / 1e3, num_sig_figs) + "K"
    elif given_value < 1e9:
        return str_with_sig_figs(given_value / 1e6, num_sig_figs) + "M"
    else:
        return str_with_sig_figs(given_value / 1e6, num_sig_figs) + "B"


class TabBuilder:
    def __init__(self, tab_text: str, before_tab_text: str, after_tab_text: str):
        self.lookup_by_num_tabs: Dict[int, str] = {}
        self.tab_text = tab_text
        self.before_tab_text = before_tab_text
        self.after_tab_text = after_tab_text
        self.joined_text = self.before_tab_text + self.after_tab_text
        self.render_tabs = string_has_value(tab_text)

    def build(self, num_tabs: int) -> str:
        if num_tabs not in self.lookup_by_num_tabs:
            str_to_return: str
            if self.render_tabs:
                str_to_return = self.before_tab_text + (self.tab_text * num_tabs) + self.after_tab_text
            else:
                str_to_return = self.joined_text

            self.lookup_by_num_tabs[num_tabs] = str_to_return
            return str_to_return
        return self.lookup_by_num_tabs[num_tabs]


class CodeStyle:
    def __init__(self, indent_text: str = "\t",
                 put_on_new_lines: bool = True,
                 before_condition_text: str = "",
                 after_condition_text: str = "",
                 before_column_text: str = "",
                 after_column_text: str = "",
                 before_value_text: str = "",
                 after_value_text: str = "",
                 else_text: str = "",
                 ending_text: str = ""):
        self.before_value_builder = TabBuilder(indent_text, "", before_value_text)
        self.after_value_text = after_value_text
        # build out the strings that will be used
        # should look something like this:
        # if {column} <= {threshold}:
        #   {true_value}
        # else:
        #   {false_value}
        # {potential end here}
        # before the column may be tabs, so introduce as a tab builder
        self.before_column_builder = TabBuilder(indent_text, "", before_condition_text + before_column_text)
        # before the threshold, close out the column text, and introduce the less than operator
        self.before_threshold_text = after_column_text + " <= "
        # before the true, close out the condition, and introduce a new line (if needed)
        # note that the true statement will provide its own tabs
        potentialNewLineText = "\n" if put_on_new_lines else ""
        self.before_true_text = after_condition_text + potentialNewLineText
        # before the false statement, provide a potential new line, potentially tabs, the else text,
        # and potentially a new line
        self.before_false_builder = TabBuilder(indent_text, potentialNewLineText, else_text + potentialNewLineText)
        # before the ending (if an ending is specified), then add a potential new line, potential tabs,
        # and the closing text
        self.render_ending = string_has_value(ending_text)
        self.before_ending_builder = TabBuilder(indent_text, potentialNewLineText, ending_text)

    def build_value(self, depth, value):
        return self.before_value_builder.build(depth) + str(value) + self.after_value_text

    def build_full_if_statement(self, depth: int, column_name: str, threshold, true_text: str, false_text: str):
        builtStatement = self.before_column_builder.build(depth) + column_name + self.before_threshold_text \
                         + str(threshold) + self.before_true_text + true_text + self.before_false_builder.build(depth) \
                         + false_text
        if self.render_ending:
            return builtStatement + self.before_ending_builder.build(depth)
        else:
            return builtStatement


PYTHON_RENDER_STYLE = CodeStyle(indent_text="\t", put_on_new_lines=True,
                                before_condition_text="if ", after_condition_text=":",
                                before_column_text='df["', after_column_text='"]',
                                before_value_text="return ", after_value_text="",
                                else_text="else:", ending_text="")
PRINT_RENDER_STYLE = CodeStyle(indent_text="\t", put_on_new_lines=True,
                               before_condition_text="", after_condition_text=":",
                               before_column_text='', after_column_text='',
                               before_value_text="value: ", after_value_text="",
                               else_text="otherwise", ending_text="")
TABLEAU_RENDER_STYLE = CodeStyle(indent_text="\t", put_on_new_lines=True,
                                 before_condition_text="IF ", after_condition_text=" THEN",
                                 before_column_text='["', after_column_text='"]',
                                 before_value_text="", after_value_text="",
                                 else_text="ELSE", ending_text="END")
TABLEAU_INLINE_RENDER_STYLE = CodeStyle(indent_text="", put_on_new_lines=False,
                                        before_condition_text="iif(", after_condition_text=",",
                                        before_column_text='["', after_column_text='"]',
                                        before_value_text="", after_value_text="",
                                        else_text=",", ending_text=")")
SQL_INLINE_RENDER_STYLE = CodeStyle(indent_text="", put_on_new_lines=False,
                                    before_condition_text="if(", after_condition_text=",",
                                    before_column_text='`', after_column_text='`',
                                    before_value_text="", after_value_text="",
                                    else_text=",", ending_text=")")
