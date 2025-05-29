from __future__ import annotations

import warnings
from typing import Dict, List, Type, Set, Any, Tuple, Callable, Iterable
import inspect
import json
from enum import IntEnum
from abc import ABC, ABCMeta
from overall_helpers import TryGetValue, convert_to_iterable, ScoreValue

_pass_through_types = (str, int, float, bool)
_list_types = (list, tuple)
_reference_id_attr = "reference_id"
_reference_attr = "reference"
_class_name_helper = "__class_hint"
_lowest_valid_similarity_score = 0
_similarity_threshold = 0.5


class _JsonConversionPlaceholder:
    # holds a value (and a relation to the parent) until is ready to be converted
    def __init__(self, parent_json: Dict | List | None, index_or_key: str | int, value, reference_id):
        self.value = value
        self.parent_json = parent_json
        self.index_or_key = index_or_key
        self.json_result = {}
        self.reference_id = reference_id
        self.is_not_referenced = True

    def mark_as_referenced(self):
        if self.is_not_referenced:
            self.is_not_referenced = False
            self.json_result[_reference_id_attr] = self.reference_id


class _JsonSerializer:
    # serializes a given object
    # works through the keys of the object, and sets the value
    def __init__(self, converter_helper: _JsonConverter):
        self.next_id = 1
        self.converter_helper = converter_helper
        self.converted_items_by_hash: Dict[int, _JsonConversionPlaceholder] = {}
        self.converted_items_by_id: Dict[int, Dict] = {}
        self.conversion_stack: List[_JsonConversionPlaceholder] = []

    def serialize(self, item_to_serialize) -> Dict[str, Any]:
        json_output = self._add_item_to_convert(None, 0, item_to_serialize, hash(item_to_serialize))
        while len(self.conversion_stack) > 0:
            curr_placeholder = self.conversion_stack.pop()
            resulting_json = curr_placeholder.json_result
            cls_info = self.converter_helper.find_class_info(type(curr_placeholder.value))
            for k, v in self._get_keys_and_values_to_convert(curr_placeholder.value):
                # if can immediately create, then will do so
                # otherwise, will return a placeholder that is added to the stack
                resulting_json[k] = self._serialize_value(resulting_json, k, v, cls_info)

            # write to the parent json (if has one)
            if curr_placeholder.parent_json is not None:
                curr_placeholder.parent_json[curr_placeholder.index_or_key] = resulting_json
        return json_output.json_result

    def _get_keys_and_values_to_convert(self, item_to_convert) -> List[Tuple[str, Any]]:
        try:
            item_fields: Dict[str, Any] = item_to_convert.__dict__
        except Exception as e:
            raise e
        # figure out which keys to consider
        # try to find information about the item to convert
        if clsInfo := self.converter_helper.find_class_info(type(item_to_convert)):
            # determine which items should import
            values_to_use: Dict[str, Any] = {k: v for k, v in item_fields.items() if clsInfo.include_attribute(k)}
            # import the items to consider first, and then import all other items
            value_sets: List[Tuple[str, Any]] = [(n, values_to_use[n]) for n in clsInfo.attributes_to_prioritize
                                                 if n in values_to_use]
            for k, v in values_to_use.items():
                if k not in clsInfo.attribute_set_to_prioritize:
                    value_sets.append((k, v))
            # in addition, add the type of the class
            value_sets.append((_class_name_helper, clsInfo.class_name))
            return value_sets
        else:
            return [(k, v) for k, v in item_fields.items() if not k.startswith("_")]

    def _add_item_to_convert(self, parent_json, index_or_key: str | int, given_item, hash_val: int):
        assigned_id = self.next_id
        self.next_id += 1
        placeholder = _JsonConversionPlaceholder(parent_json, index_or_key, given_item, assigned_id)
        self.converted_items_by_hash[hash_val] = placeholder
        self.conversion_stack.append(placeholder)
        return placeholder

    def _serialize_value(self, parent_json, index_or_key: str | int, given_item,
                         parent_cls_info: _JsonClassInfo | None = None):
        # if can immediately create, then will return the created value
        # otherwise, will return a placeholder, with a link to the parent and the index
        # note that the parent should be either an object (dict in json) or iterable (list in json)

        # check to see if is none
        if given_item is None:
            return given_item

        # check if the item can be written as-is
        for t in _pass_through_types:
            if isinstance(given_item, t):
                return given_item

        # if is an iterable type, then convert over to a list
        # create records for all items first (in case one item references another item in the list)
        if isinstance(given_item, list):
            json_list = []
            for i, v in enumerate(given_item):
                json_list.append(self._serialize_value(json_list, i, v))
            return json_list
        elif isinstance(given_item, dict):
            json_obj = {}
            for k, v in given_item.items():
                json_obj[k] = self._serialize_value(json_obj, k, v)
            return json_obj

        # see if there is a registered transformation
        if parent_cls_info is not None:
            if transformed_value := parent_cls_info.apply_serialization_transformation(index_or_key, given_item):
                # rerun with the transformed value
                # don't include the parent class info, since shouldn't be able to transform again
                return self._serialize_value(parent_json, index_or_key, transformed_value, None)

        # check to see if has converted before
        # if have, then just return the id of the converted value
        try:
            hash_val = hash(given_item)
        except Exception as e:
            raise e
        if referenced_item := TryGetValue(self.converted_items_by_hash, hash_val):
            # mark that the item is used, and therefore should include a reference id]
            referenced_item.mark_as_referenced()
            return {_reference_attr: referenced_item.reference_id}
        else:
            # if here, then means that needs to convert, so add to the list to convert
            return self._add_item_to_convert(parent_json, index_or_key, given_item, hash_val)


class _JsonDeserializerWaitingInstance:
    def __init__(self, parent_item, index_or_key: str | int):
        self.parent_item = parent_item
        self.index_or_key = index_or_key

    def connect_to_reference(self, referenced_item):
        if self.parent_item is not None:
            if isinstance(self.parent_item, list):
                self.parent_item[self.index_or_key] = referenced_item
            elif isinstance(self.parent_item, dict):
                self.parent_item[self.index_or_key] = referenced_item
            else:
                # set the attribute within the dictionary
                self.parent_item.__dict__[self.index_or_key] = referenced_item


class _JsonDeserializer:
    def __init__(self, converter_helper: _JsonConverter, minimum_similarity=_similarity_threshold,
                 minimum_similarity_for_name_match=_lowest_valid_similarity_score):
        self.converter_helper = converter_helper
        self.referenced_items = {}
        self.created_items = []
        self.references_waiting: Dict[int, List[_JsonDeserializerWaitingInstance]] = {}
        self.minimum_similarity = minimum_similarity
        self.minimum_similarity_for_name_match = minimum_similarity_for_name_match
        self.warned_class_names: set[str] = set()

    def deserialize(self, given_data: Dict[str, Any]):
        # deserialize the item
        output_obj = self._deserialize(None, 0, given_data)
        # connect up any items still waiting
        # next, run any events after creation as needed
        for c in self.created_items:
            if isinstance(c, _JsonInterfaceBase):
                c.from_json_after_events()
            elif cls_info := self.converter_helper.find_class_info(type(c)):
                cls_info.after_deserialization(c)
        return output_obj

    def _cache_deserialized_item(self, deserialized_item, reference_id: int | None):
        self.created_items.append(deserialized_item)
        if reference_id is not None:
            # add to the referenced items list
            self.referenced_items[reference_id] = deserialized_item
            # check the references waiting
            if reference_id in self.references_waiting:
                itemsWaiting = self.references_waiting.pop(reference_id)
                for r in itemsWaiting:
                    r.connect_to_reference(deserialized_item)

    def _deserialize_as_dictionary(self, reference_id, given_item: Dict):
        # since doesn't match enough, then just return as a dictionary
        output_obj = {}
        self._cache_deserialized_item(output_obj, reference_id)
        for k, v in given_item.items():
            output_obj[k] = self._deserialize(output_obj, k, v)
        return output_obj

    def _deserialize_item_reference(self, parent_item, index_or_key: str | int,
                                    given_item: Dict) -> _JsonDeserializerWaitingInstance:
        # means that is a reference, so grab the id
        referenced_id = given_item[_reference_attr]
        if isinstance(referenced_id, int):
            # check to see if the given value is already found
            if referenced_item := TryGetValue(self.referenced_items, referenced_id):
                return referenced_item
            else:
                # add to the list
                waitingReference = _JsonDeserializerWaitingInstance(parent_item, index_or_key)
                if not (wait_list := TryGetValue(self.references_waiting, referenced_id)):
                    wait_list = []
                    self.references_waiting[referenced_id] = wait_list
                wait_list.append(waitingReference)
                return waitingReference

    def _find_best_deserialize_class_target(self, item_key_set: Set[str], class_name_hint: str | None = None) -> Tuple[float, _JsonClassInfo]:
        # if the class name hint is given, then check to find the best option with the given name
        if class_name_hint is not None:
            class_options = self.converter_helper.find_classes_with_name(class_name_hint)
            # if the name is given, but neither the item keys nor the class keys have values, then meet the minimum similarity
            best_score, best_class = self._find_best_deserialize_class_target_from_dictionary(class_options, item_key_set, self.minimum_similarity)
            # must meet the minimum similarity to be returned
            if best_score >= self.minimum_similarity_for_name_match:
                return best_score, best_class

        # otherwise, look through all options for best class
        return self._find_best_deserialize_class_target_from_dictionary(self.converter_helper.registered_classes, item_key_set)

    @staticmethod
    def _find_best_deserialize_class_target_from_dictionary(class_options: Iterable[_JsonClassInfo],
                                                            item_key_set: Set[str],
                                                            empty_key_score: float = _lowest_valid_similarity_score) -> Tuple[float, _JsonClassInfo]:
        # allow to have the minimum score of 0 (in the case that there are no matching values)
        best_similarity = ScoreValue(minimum_viable_score=_lowest_valid_similarity_score, minimum_score_inclusive=True)
        best_class_info = None
        # if the class name hint is given, then check to find the best option with the given name
        for c in class_options:
            class_similarity = c.calculate_similarity(item_key_set, empty_key_score)
            if best_similarity.update_score(class_similarity):
                best_class_info = c
                # if is a perfect match, then return
                if class_similarity == 1:
                    break
        return best_similarity.score, best_class_info

    def _build_deserialized_instance(self, best_class_info: _JsonClassInfo, given_item: Dict[str, Any]):
        # first, pull the arguments
        arg_list = []
        waiting_arguments: List[Tuple[str, _JsonDeserializerWaitingInstance]] = []
        for var_name in best_class_info.init_args:
            if var_name in given_item:
                stored_value = given_item[var_name]
            else:
                stored_value = best_class_info.default_arg_values[var_name]

            # build the argument
            # at first, set the parent to the argument list
            associated_value = self._deserialize(arg_list, len(arg_list), stored_value, best_class_info)
            if isinstance(associated_value, _JsonDeserializerWaitingInstance):
                waiting_arguments.append((var_name, associated_value))
            arg_list.append(associated_value)

        # if can go ahead and build, then do so
        # otherwise, will need to wait until is complete
        try:
            output_obj = best_class_info.given_type(*arg_list)
        except Exception as e:
            raise e

        # next, go through the waiting arguments and set the created item as what is waiting for
        # convert the waiting arguments over to the newly created object
        for v, w in waiting_arguments:
            w.parent_item = output_obj
            w.index_or_key = v
        return output_obj

    def _deserialize(self, parent_item, index_or_key: str | int, given_item,
                     parent_class_info: _JsonClassInfo | None = None):
        # check to see if is none
        if given_item is None:
            return given_item

        # check if the item can be written as-is
        for t in _pass_through_types:
            if isinstance(given_item, t):
                return given_item

        # if is an iterable type, then convert over to a list
        # create records for all items first (in case one item references another item in the list)
        if isinstance(given_item, list):
            result_list = []
            for i, v in enumerate(given_item):
                result_list.append(self._deserialize(result_list, i, v))
            return result_list
        elif isinstance(given_item, dict):
            # if is here, then means that is either a dictionary or a complex type
            # first, check to see if is a placeholder for a reference
            if _reference_attr in given_item and len(given_item) == 1:
                return self._deserialize_item_reference(parent_item, index_or_key, given_item)

            # iterate through the class options to find the best class
            # first, get a list of the keys
            # there may be a reference id. If there is, then read and remove
            item_key_set = set(given_item.keys())
            if reference_id := TryGetValue(given_item, _reference_id_attr):
                item_key_set.remove(_reference_id_attr)
            if class_name_hint := TryGetValue(given_item, _class_name_helper):
                item_key_set.remove(_class_name_helper)

            # next, try to find the best class of the options given
            best_similarity, best_class_info = self._find_best_deserialize_class_target(item_key_set, class_name_hint)
            if best_similarity >= self.minimum_similarity:
                # means that should be able to build
                output_obj = self._build_deserialized_instance(best_class_info, given_item)

                # next, log the created item (so that any items can use this item)
                self._cache_deserialized_item(output_obj, reference_id)

                # finally, load any non-argument values into the class
                # only load the arguments that were not provided to the constructor
                # and that are included within the key set
                for attr_name in best_class_info.non_init_attributes.intersection(item_key_set):
                    output_obj.__dict__[attr_name] = self._deserialize(output_obj, attr_name, given_item[attr_name],
                                                                       best_class_info)

                return output_obj
            else:
                # if hasn't been warned before, then warn about being unable to deserialize
                if class_name_hint is not None and class_name_hint not in self.warned_class_names:
                    warnings.warn(f"An instance of an object with class name '{class_name_hint}' was unable to be deserialized. "
                                  f"Unexpected behavior may occur.")
                    self.warned_class_names.add(class_name_hint)

                # since doesn't match enough, then just return as a dictionary
                return self._deserialize_as_dictionary(reference_id, given_item)
        else:
            # check to see if the given type is handled
            raise ValueError("Unknown json type passed in.")


# create as a subclass of abstract, so that subclasses can use the abstract methods
class _JsonInterfaceBase(ABC):
    @classmethod
    def include_json_names(cls, attributes_to_include: Set[str]):
        """
        Adds attribute names to include when serializing to json.
        By default, includes all instance attributes, except for protected attributes.
        :param attributes_to_include: Set[str]
        """
        # by default, don't include anything
        pass

    @classmethod
    def exclude_json_names(cls, attributes_to_exclude: Set[str]):
        """
        Adds attribute names to exclude when serializing to json.
        By default, includes all instance attributes, except for protected attributes.
        :param attributes_to_exclude: Set[str]
        """
        # by default, don't include anything
        pass

    @classmethod
    def prioritize_json_names(cls, attributes_to_prioritize: List[str]):
        """
        Adds attribute names to prioritize when serializing to json.
        :param attributes_to_prioritize: List[str]
        """
        pass

    def from_json_after_events(self):
        """
        Events to run after have created an item from a json value
        """
        pass

    def to_json(self) -> Dict[str, Any]:
        """
        Converts this instance to a JSON representation (represented by a dictionary)
        :return: Dict[str, Any]
        """
        return json_converter.to_json(self)

    def to_json_file(self, filename: str):
        """
        Writes the JSON representation of this instance to a file.
        :param filename: str
        """
        json_representation = self.to_json()
        with open(filename, "w") as fh:
            json.dump(json_representation, fh)

    @classmethod
    def from_json(cls, json_data: Dict[str, Any], minimum_similarity=0.5):
        """
        Creates an instance from the registered classes from a JSON representation.
        :param json_data: Dict[str, Any]
        :param minimum_similarity: Minimum Jaccard similarity between JSON keys and class instance attributes for class
        selection to occur.
        :return: A new instance of the this class.
        """
        return json_converter.from_json(json_data, minimum_similarity)

    @classmethod
    def from_json_file(cls, filename: str):
        """
        Reads the JSON representation of this class from a file, and creates the corresponding instance.
        :param filename:
        :return:
        """
        with open(filename, "r") as fh:
            json_representation = json.load(fh)
        given_value = cls.from_json(json_representation)
        assert isinstance(given_value, cls), "Type other than the expected type returned..."
        return given_value


class _JsonAttributeOptionApplicability(IntEnum):
    # ordered by
    DOES_NOT_APPLY = -1
    DEFAULT_MATCH = 0
    TYPE_MATCH = 1
    KEY_MATCH = 2
    TYPE_AND_KEY_MATCH = 3


# options for serializing / deserializing values
class _JsonAttributeOptions:
    def __init__(self):
        self.applies_to: Set[str] = set()
        self.serialize_transformation: Callable[[Any], Any] | None = None
        self.deserialize_transformation: Callable[[Any], Any] | None = None
        self.applies_to_types: Set[Type] = set()

    def get_applicability(self, attr_name: str, attr_type: Type) -> _JsonAttributeOptionApplicability:
        type_matches = attr_type in self.applies_to_types
        key_matches = attr_name in self.applies_to
        # if there are values within a set and the given criteria isn't within the set, then doesn't apply
        if len(self.applies_to) > 0 and not key_matches:
            return _JsonAttributeOptionApplicability.DOES_NOT_APPLY
        elif len(self.applies_to_types) > 0 and not type_matches:
            return _JsonAttributeOptionApplicability.DOES_NOT_APPLY
        elif type_matches and key_matches:
            return _JsonAttributeOptionApplicability.TYPE_AND_KEY_MATCH
        elif type_matches:
            return _JsonAttributeOptionApplicability.TYPE_MATCH
        elif key_matches:
            return _JsonAttributeOptionApplicability.KEY_MATCH
        else:
            return _JsonAttributeOptionApplicability.DEFAULT_MATCH


def get_class_initial_attributes(given_class: Type) -> list[str]:
    # first, get all of the variables used within the initialization function
    # note that this may include method names from other objects
    # because of this, will need to verify that at least one instance in the code occurs behind self.*
    if inspect.isfunction(given_class.__init__):
        # get the list of instance variables and / or functions used in the init method
        instance_vars = inspect.getclosurevars(given_class.__init__).unbound
        init_func_code = inspect.getsource(given_class.__init__)
        return [v for v in instance_vars if "self." + v in init_func_code]
    # if here, then return an empty list
    return []


class _JsonClassInfo:
    def __init__(self, given_class: Type):
        # set the type
        self.given_type = given_class
        self.class_name = given_class.__name__
        # get any attributes that have requested to import / exclude / prioritize
        self.attributes_to_import: Set[str] = set()
        self.attributes_to_ignore: Set[str] = set()
        self.attributes_to_prioritize: List[str] = []
        if issubclass(given_class, _JsonInterfaceBase):
            given_class.include_json_names(self.attributes_to_import)
            given_class.exclude_json_names(self.attributes_to_ignore)
            given_class.prioritize_json_names(self.attributes_to_prioritize)
        self.attribute_set_to_prioritize = set(self.attributes_to_prioritize)
        # figure out what items are needed for the initialization
        init_argspec = inspect.getfullargspec(given_class.__init__)
        default_args = [] if init_argspec.defaults is None else init_argspec.defaults
        numDefaults = len(default_args)
        numRequiredVars = len(init_argspec.args) - numDefaults
        required_args = [init_argspec.args[i] for i in range(numRequiredVars)]
        self.default_arg_values = {init_argspec.args[i + numRequiredVars]: default_args[i] for i in range(numDefaults)}
        self.init_args = [v for v in init_argspec.args if v != "self"]
        self.init_args_set = set(self.init_args)
        self.required_init_args = [v for v in required_args if v != "self"]
        self.required_init_arg_set = set(self.required_init_args)
        self.non_required_init_args = self.init_args_set.difference(self.required_init_arg_set)
        # figure out the methods within the class
        self.methods = set(dir(self.given_type))
        # get a list of the class static variables and methods
        class_data = inspect.getmembers(given_class)
        instance_func_or_class_vars = set([n[0] for n in class_data])
        self.static_vars = instance_func_or_class_vars - self.methods
        # figure out which instance attributes are set within the python class
        # note that the co_names will include super, and methods called, etc.
        # for now, these are set by the unbound property of the closure variables method
        instance_vars = get_class_initial_attributes(given_class)
        # remove any instance methods or class static variables that may be used
        self.instance_attributes = set([v for v in instance_vars if not (v.startswith("_") or v == "super"
                                                                         or v in instance_func_or_class_vars
                                                                         or v in self.methods)])
        # remove any items that are not passed in as init arguments
        # assumes that if a value is passed in from the initialization arguments and matches the name,
        # that will be set by the initialization function
        self.non_init_attributes = self.instance_attributes - self.init_args_set
        # set up the attribute options
        self.serialize_option_lookup: Dict[str, _JsonAttributeOptions] = {}
        self.deserialize_option_lookup: Dict[str, _JsonAttributeOptions] = {}
        # set up the classes that are underneath this class
        self.super_classes_info: List[_JsonClassInfo] = []
        self.sub_classes_info: List[_JsonClassInfo] = []

    def import_super_class_info(self, super_class_info: _JsonClassInfo):
        # import the methods, instance attributes, and non-init attributes
        self.methods.update(super_class_info.methods)
        self.instance_attributes.update(super_class_info.instance_attributes)
        self.non_init_attributes.update(super_class_info.non_init_attributes)
        self.super_classes_info.append(super_class_info)
        super_class_info.sub_classes_info.append(self)

    def __str__(self):
        return f"JSON Information for class '{self.class_name}'"

    def calculate_similarity(self, item_attrs: Set, empty_key_score: float = _lowest_valid_similarity_score):
        # uses the Jaccard similarity of the two sets
        # if there are init arguments, make sure that all are present
        # if not all present, then don't allow to create
        if len(self.required_init_arg_set) > 0:
            if not self.required_init_arg_set.issubset(item_attrs):
                return -1
        # calculate the number of shared elements
        numUnique = len(self.instance_attributes.union(item_attrs))
        if numUnique == 0:
            return empty_key_score
        numShared = len(self.instance_attributes.intersection(item_attrs))
        return numShared / numUnique

    def include_attribute(self, attr_name: str) -> bool:
        if attr_name in self.attributes_to_import:
            return True
        else:
            # don't include protected fields, as assume that will build within constructor
            return not (attr_name in self.attributes_to_ignore or attr_name.startswith("_"))

    def after_deserialization(self, deserialized_item):
        """
        Events to run after the deserialization of an item of an instance of a class that is NOT a sub-class of
        JsonBase. If the item is an instance of JsonBase, should use the from_json_after_events method override instead.
        :param deserialized_item:
        """
        pass

    def register_transformations(self, attr_name: List[str] | Tuple[str] | str,
                                 serialize_transformation: Callable[[Any], Any] | None = None,
                                 deserialize_transformation: Callable[[Any], Any] | None = None):
        if serialize_transformation is not None or deserialize_transformation is not None:
            attr_transformation = _JsonAttributeOptions()
            attr_transformation.deserialize_transformation = deserialize_transformation
            attr_transformation.serialize_transformation = serialize_transformation
            attr_transformation.applies_to.update(convert_to_iterable(attr_name, str))
            for n in convert_to_iterable(attr_name, str):
                if serialize_transformation is not None:
                    self.serialize_option_lookup[n] = attr_transformation
                    for c in self.sub_classes_info:
                        c.serialize_option_lookup[n] = attr_transformation
                if deserialize_transformation is not None:
                    self.deserialize_option_lookup[n] = attr_transformation
                    for c in self.sub_classes_info:
                        c.deserialize_option_lookup[n] = attr_transformation

    def apply_serialization_transformation(self, attr_name, attr_value):
        """
        Applies a registered function to the value to serialize to JSON.
        Returns None if not applicable.
        :param attr_name:
        :param attr_value:
        """
        if found_option := TryGetValue(self.serialize_option_lookup, attr_name):
            return found_option.serialize_transformation(attr_value)
        else:
            return None

    def apply_deserialization_transformation(self, attr_name, attr_value):
        """
        Applies a registered function to the value to deserialize from JSON.
        Returns None if not applicable.
        :param attr_name:
        :param attr_value:
        """
        if found_option := TryGetValue(self.deserialize_option_lookup, attr_name):
            return found_option.serialize_transformation(attr_value)
        else:
            return None


class _JsonConverter:
    def __init__(self):
        self.registered_classes: List[_JsonClassInfo] = []
        self.registered_classes_lookup: Dict[type, _JsonClassInfo] = {}
        self.registered_classes_lookup_by_name: Dict[str, Dict[type, _JsonClassInfo]] = {}

    def register(self, given_class: Type) -> _JsonClassInfo:
        # create the class information
        clsInfo = _JsonClassInfo(given_class)
        # for any superclasses, add the information for it
        for c in self.registered_classes:
            if issubclass(given_class, c.given_type):
                clsInfo.import_super_class_info(c)
            elif issubclass(c.given_type, given_class):
                c.import_super_class_info(clsInfo)
        self.registered_classes.append(clsInfo)
        self.registered_classes_lookup[given_class] = clsInfo
        # add by name
        # note that the same class name could be used within multiple modules, so create lookup to find exact name
        if clsInfo.class_name in self.registered_classes_lookup_by_name:
            lookup_by_name = self.registered_classes_lookup_by_name[clsInfo.class_name]
        else:
            lookup_by_name = {}
            self.registered_classes_lookup_by_name[clsInfo.class_name] = lookup_by_name
        lookup_by_name[given_class] = clsInfo
        return clsInfo

    def to_json(self, item):
        serializer_helper = _JsonSerializer(self)
        return serializer_helper.serialize(item)

    def from_json(self, json_item: Dict, minimum_similarity=0.5):
        # if is an simple type, then return that value
        jd = _JsonDeserializer(self, minimum_similarity)
        return jd.deserialize(json_item)

    def find_class_info(self, cls_type: Type) -> _JsonClassInfo | None:
        return TryGetValue(self.registered_classes_lookup, cls_type)

    def find_classes_with_name(self, cls_name: str) -> Iterable[_JsonClassInfo]:
        if cls_name in self.registered_classes_lookup_by_name:
            return self.registered_classes_lookup_by_name[cls_name].values()
        else:
            return []


json_converter = _JsonConverter()


class JsonBaseMeta(ABCMeta):
    # metaclass to keep track of any subclasses of the json base class
    def __init__(cls, name, bases, clsdict):
        super(JsonBaseMeta, cls).__init__(name, bases, clsdict)
        json_converter.register(cls)


# all classes that are convertible to / from JSON should use this as a base class
class JsonBase(_JsonInterfaceBase, metaclass=JsonBaseMeta):
    pass
