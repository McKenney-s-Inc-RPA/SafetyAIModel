# helper to wrap text
from __future__ import annotations
from typing import List, Dict, Set
from abc import ABC, abstractmethod


class CharBoundsBase(ABC):
    @abstractmethod
    def is_closing_char(self, char) -> bool: ...

    @abstractmethod
    def should_ignore_start(self, char) -> bool: ...


class CharBounds(CharBoundsBase):
    def __init__(self, start: str, stop: str):
        self.start = start
        self.stop = stop
        self.ignore_starts: Set[str] = set()

    def is_closing_char(self, char) -> bool:
        return self.stop == char

    def should_ignore_start(self, char) -> bool:
        return char in self.ignore_starts


class NoBounds(CharBoundsBase):
    def is_closing_char(self, char) -> bool:
        return False

    def should_ignore_start(self, char) -> bool:
        return False


class TextWrapperSections:
    def __init__(self, offset, wrapping_bounds: CharBoundsBase, parent_section: TextWrapperSections | None = None):
        self.text = ""
        self.offset = offset
        self.length = 0
        self.wrapping_bounds = wrapping_bounds
        self.parent_section = parent_section
        self.sections: List[TextWrapperSections] = []
        if parent_section is not None:
            parent_section.sections.append(self)

    def set_text(self, sub_text):
        self.text = sub_text
        self.length = len(sub_text)


# breaks apart text into sections


class TextWrapper:
    def __init__(self):
        self.char_bounds: List[CharBounds] = []
        self.lookup_by_start_char: Dict[str, CharBounds] = {}
        self.splitters: Dict[str, bool] = {}
        self.sections = []

    def add_char_bounds(self, start: str, stop: str):
        new_bounds = CharBounds(start, stop)
        self.char_bounds.append(new_bounds)
        self.lookup_by_start_char[start] = new_bounds
        return new_bounds

    def add_splitter(self, splitter: str, should_keep_if_split: bool):
        self.splitters[splitter] = should_keep_if_split

    def analyze_text(self, text, target_length):
        curr_section = TextWrapperSections(0, NoBounds(), None)
        curr_section.set_text(text)
        for i, c in enumerate(text):
            # check if is the start of a new section
            if c in self.lookup_by_start_char and not curr_section.wrapping_bounds.should_ignore_start(c):
                # create the new section
                curr_section = TextWrapperSections(i, self.lookup_by_start_char[c], curr_section)
                # break up any sub sections
                offset_index = i + 1
            elif curr_section.wrapping_bounds.is_closing_char(c):
                # close out the value, and move up to the parent section
                curr_section.set_text(text[curr_section.offset, i + 1])
                curr_section = curr_section.parent_section
            else:
                # check to see if is a breaking character
                pass

        # now, go through each section and try to split up into sections with the given length
