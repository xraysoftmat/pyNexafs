"""
The base parser classes for loading and parsing data files.

`parserMeta` is a metaclass that produces the `parserBase` (and inherited) parser class(es).
A metaclass allows for the dynamic creation of classes, and the modification of class attributes and methods.
This is used to redefine and share class properties and attributes relating to "name relabelling".
The `parserMeta` also collects `parse_<filetype>` methods from the parser classes and stores them for loading.
The following important attributes become re-defined as hidden properties:
- `ALLOWED_EXTENSIONS`, a list of strings that define the valid file extensions for the parser,
- `COLUMN_ASSIGNMENTS`, a dictionary that assigns data columns (i.e. 'x', 'y', 'y_err') to scan object parameters,
- `SUMMARY_PARAM_RAW_NAMES`, a list of parameter strings that provide a summary of the parser properties. Tuples of synonymous strings are also allowed in the list,
- `RELABELS`, a dictionary describing a unique one-to-one mapping between the original column/parameter name(s) and a more useful name.

`parserBase` is the base class for all parser classes, adding individual parser attributes and default methods for file loading.
The `parserBase` class is an abstract class, and cannot be instantiated. Classes that inherit from `parserBase` must implement the following attributes:
- `ALLOWED_EXTENSIONS`, a list of strings that define the valid file extensions for the parser,
- `COLUMN_ASSIGNMENTS`, a dictionary that assigns the labels 'x', 'y' (and optionally 'x_errs', 'y_errs') to the data column name(s) in the file. The
    'x' (and 'x_errs') must map to a single string, and 'y' (and 'y_errs') can map to a list of strings or a single string.
The following attributes are optional:
- `SUMMARY_PARAM_RAW_NAMES`, a list of parameter strings that provide a summary of the parser properties. Tuples of synonymous strings are also allowed in the list.
- `RELABELS`, a dictionary describing a unique mapping between the original column/parameter names and more useful name(s).
The following methods are also optional to implement, and will otherwise return a NotImplementedError:
- `summary_param_names_with_units`
"""

import abc
import io
import os
import sys
import tempfile
import types
import typing
import warnings
from _collections_abc import dict_items, dict_keys
from collections.abc import Callable, ItemsView, Iterable, KeysView
from typing import IO, Any, Self, TextIO, override

if typing.TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem
import datetime
import traceback

import numpy as np
import numpy.typing as npt

from pyNexafs.nexafs.scan import scanBase
from pyNexafs.types import assignments_type, dtype, parse_fn_ret_type, reduction_type

# Optional pandas import
try:
    import pandas as pd
except ImportError:
    pd = None


class parserMeta(abc.ABCMeta):
    """
    Metaclass to implement class properties for parser classes.

    Each parser class must implement the following properties:
    - `ALLOWED_EXTENSIONS`, a list of strings that define the valid file extensions for the parser,
    - `COLUMN_ASSIGNMENTS`, a dictionary that assigns data columns (i.e. 'x', 'y', 'y_err') to scan object parameters,

    The following properties are optional:
    - `SUMMARY_PARAM_RAW_NAMES`, a list of parameter strings that provide a summary of the parser properties. Tuples of
      synonymous strings are also allowed in the list.
    - `RELABELS`, a dictionary describing a unique mapping between the original column/parameter names and more useful name(s).

    The metaclass also checks for class methods beginning with `parser_` and adds them to the `parse_functions` list.

    Parameters
    ----------
    name : str
        The name of the class.
    bases : tuple[type, ...]
        The base classes of the class.
    namespace : dict[str, Any]
        The namespace of the class - it's attributes names and values.

    Raises
    ------
    NameError
        If `parse_<filetype>` methods do not have the correct signature.
    AttributeError
        If no parser methods are found in the class.

    See Also
    --------
    parserBase
        Implements loading and parsing of data files.
    """

    class relabels_dict(dict):
        """
        A dictionary subclass that deals with tuples of synonymous str keys.

        Allows for keys to be a synonymous tuple of strings, and for the dictionary to be searched
        for any of the synonymous keys. A `__getitem__` or `__contains__` requests will succeed for
        a key found in a tuple. The request will also succeed for a value, returning itself.

        When `__setitem__` is used, the following rules apply:
        - Setting a new value for a key will add the key and old value as a new tuple key.

        To remove an entry, use `del dict[key]` or `dict.pop(key)`, which will remove the key
        and any synonymous keys in a tuple.
        """

        @override
        def __getitem__(self, key: str | dtype | tuple[str, ...]) -> str | dtype:
            """
            Value matching for a given key. Overridden to allow for synonymous keys searching.

            The following mappings work:
            - key: str, matches a str key in the dictionary.
            - key: str, matches a str element in a tuple key in the dictionary.
            - key: tuple, matches a tuple key in the dictionary.
            - key: tuple, element matches a str key in a tuple in the dictionary.

            Intersections of tuples are not allowed, and will raise a KeyError.

            Parameters
            ----------
            key : str | tuple[str, ...]
                The key to search for in the dictionary.

            Returns
            -------
            str
                The value of the key in the dictionary.

            Raises
            ------
            KeyError
                If the key is not found in the dictionary.
            """
            if isinstance(key, tuple):
                if super().__contains__(key):  # Regular instance check
                    return super().__getitem__(key)
                # Check for each str.
                for key_sub in key:
                    if key_sub in self:  # Check each tuple as well
                        return self.__getitem__(
                            key_sub
                        )  # Recursive call on the contained key
            elif isinstance(key, (str, dtype)):
                for k, v in self.items():
                    if k == key or v == key:
                        # Direct match with key, return value.
                        return v
                    elif isinstance(k, tuple):
                        if key in k:
                            return v
            raise KeyError(f"Key {key} not found in RELABELS dictionary.")

        @override
        def __contains__(self, key: object) -> bool:
            """
            Check key membership, allowing for synonymous keys searching.

            Redefines the contains method to additionally search keys of
            `tuple` type for the key match, and the `dtype` for string matches.

            Parameters
            ----------
            key : object
                The key to search for in the dictionary.

            Returns
            -------
            bool
                True if the key is found in the dictionary or within a tuple element.
            """
            if isinstance(key, tuple):
                # Iterate over all keys and check for the key.
                for k in self.keys():
                    # Check a tuple match.
                    if k == key:
                        return True
                    # Check each tuple sub string.
                    for key_sub in key:
                        if k == key_sub:
                            return True
            elif isinstance(key, (str, dtype)):
                for k, v in self.items():
                    if k == key or v == key:
                        return True
                    elif isinstance(k, tuple):
                        if key in k:
                            return True
            return False

        @override
        def __setitem__(
            self, key: str | dtype | tuple[str | dtype, ...], value: str | dtype
        ) -> types.NoneType:
            """
            Set the value of a key in the relabels dictionary.

            Any modifications to the dictionary store the old key/value for equivalence.
            To remove old mappings, use `del dict[key]` or `dict.pop(key)` to remove the key
            before setting the new key/value pair.

            Parameters
            ----------
            key : str | dtype | tuple[str | dtype, ...]
                The key to set the value for.
            value : str | dtype
                The value to set for the key.
            """
            if not isinstance(key, (str, dtype, tuple)):
                raise ValueError(
                    f"Key {key} not a string, NEXAFS dtype, or tuple of either."
                )
            if not isinstance(value, (str, dtype)):
                raise ValueError(f"Value {value} not a string or NEXAFS dtype.")

            ## Use overridden contains method to check for key presence, which allows for tuple searching.
            # Check if the key is already in the dictionary, as a value or key:
            try:
                key_val = self[key]
                key_contained = True
            except KeyError:
                key_val = None
                key_contained = False

            # Check if the value is already in the dictionary:
            try:
                value_val = self[value]
                value_contained = True
            except KeyError:
                value_val = None
                value_contained = False

            if not key_contained and not value_contained:
                # Add the new mapping
                super().__setitem__(key, value)
            elif key_contained and value_contained:
                # Both key and value are already in the dictionary
                if key_val != value:
                    # The key and value are mapped to different values, so we need to merge the mappings.
                    # Get the existing keys for the values
                    key_keys = None
                    value_keys = None
                    for k, v in self.items():
                        if v == key_val:
                            key_keys = k
                        if v == value_val:
                            value_keys = k
                        if key_keys is not None and value_keys is not None:
                            break

                    new_key = (
                        {key}
                        | {key_val, value_val}
                        | set(key_keys if isinstance(key_keys, tuple) else (key_keys,))
                        | set(
                            value_keys
                            if isinstance(value_keys, tuple)
                            else (value_keys,)
                        )
                    )

                    if value in new_key:
                        new_key.remove(value)

                    new_key = tuple(new_key)

                    del self[key_keys]  # Remove the old key.
                    del self[value_keys]  # Remove the old value.
                    super().__setitem__(
                        new_key, value
                    )  # Add the new key with the value.
            elif key_contained:
                # Key is already in dict.
                if key_val == value:
                    # Value already set!
                    return
                else:
                    for k, v in self.items():
                        if key_val == v:
                            key_keys = k
                            new_key = {key_val} | (
                                set(k if isinstance(k, tuple) else (k,))
                            )
                            # Delete the existing key
                            del self[key_keys]
                            self[tuple(new_key)] = (
                                value  # Add the new key with the value.
                            )
                            return
            elif value_contained:
                # No key, but value exists. Find the mapping
                for k, v in self.items():
                    # Find the value is the same as relabels[value].
                    if v == value_val:
                        break
                assert v == value, "Value not found, despite contained."
                # The value is the same as
                if v == value:
                    # Add the new key to the existing value key.
                    new_key = set(k if isinstance(k, tuple) else (k,)) | {key}
                else:
                    # Value is found in the keys, move value_val to key
                    # and replace with value
                    new_key = (
                        {key} | {value_val} | set(k if isinstance(k, tuple) else (k,))
                    )
                    del self[k]  # Remove the old key.
                    self[tuple(new_key)] = value  # Add the new key with the value.
            return

    class summary_param_list(list):
        """
        A re-implemented list for the `SUMMARY_PARAM_RAW_NAMES` property.

        Allows tuples of multiple unique parameters in addition to `relabels` class functionality.
        This handles the case where multiple pieces of unique equipment have been interchanged
        for the same parameter.

        `parent` is a required keyword argument to access the `relabel` and `RELABELS` properties.

        A `__getitem__` or `__contains__` request will succeed for a key found in a tuple.
        If `parser.relabel` is `True`, the request will also succeed for any synonymous keys or the
        corresponding value.

        Modifies the `contains` and `index` method.

        Parameters
        ----------
        args : Iterable[str | tuple[str, ...]]
            The parameters to initialise the list with.
        parent : type[parserBase]
            The parent parser class, used to access the `relabel`/`RELABELS` properties.
        """

        @override
        def __init__(
            self, args: Iterable[str | tuple[str, ...]], *, parent: type["parserBase"]
        ) -> None:
            super().__init__(args)
            self._parent = parent
            self.__check_valid()

        def __check_valid(self) -> None:
            """
            Check the validity of the summary parameter list.

            Raises
            ------
            ValueError
                If there are overlapping `RELABELLED` parameters that are not part of the same tuple.
            """
            # Store the synonymous key/values in a dictionary - prefer to store the RELABEL value which is singular.
            collected_keys = dict()
            """Existing mappings of parameters to their relabelled values."""

            # Iterate over the list items
            for i in range(self.__len__()):
                item = self[i]

                # Check if the item already is collected
                if item in collected_keys:
                    raise ValueError(
                        f"Summary Parameter `{item}` already found in the summary parameter list with "
                        + f"value `{collected_keys[item] if collected_keys[item] else item}`."
                    )

                # Check if the item is a tuple
                if isinstance(item, tuple):
                    matches = []
                    for sub_item in item:
                        # Check if sub-item in relabels
                        if sub_item in self._parent.RELABELS:
                            relabel_val = self._parent.RELABELS[sub_item]
                            if relabel_val in collected_keys:
                                # Check if the collected key has a non-None value
                                raise ValueError(
                                    f"Summary Parameter `{item}` already found in the summary parameter list"
                                    + f" with value `{collected_keys[relabel_val] if collected_keys[relabel_val] else relabel_val}`."
                                )
                            else:
                                matches.append(relabel_val)
                        else:
                            # No relabel, see if in collected keys
                            if sub_item in collected_keys:
                                raise ValueError(
                                    f"Summary Parameter `{item}` already found in the summary parameter list"
                                    + f" with value `{collected_keys[sub_item] if collected_keys[sub_item] else sub_item}`."
                                )
                            else:
                                matches.append(sub_item)
                    # Add the set values to the collected keys
                    for match in set(matches):
                        collected_keys[match] = item
                else:
                    # Check if relabel
                    if item in self._parent.RELABELS:
                        relabel_val = self._parent.RELABELS[item]
                    else:
                        relabel_val = item
                    if relabel_val in collected_keys:
                        raise ValueError(
                            f"Summary Parameter `{item}` already found in the summary parameter list"
                            + f" with value `{collected_keys[relabel_val] if collected_keys[relabel_val] else relabel_val}`."
                        )
                    else:
                        collected_keys[relabel_val] = (
                            item if item != relabel_val else None
                        )

        @override
        def __contains__(self, key: object) -> bool:
            """
            Check if the key (string) is within the list or within a tuple element.

            If `parser.relabel` is `True`, then the key is also checked against the
            `parser.RELABELS` dictionary, for both synonymous string keys and values.
            Will also return true for a tuple match (but not search for individual
            components of the tuple).

            Parameters
            ----------
            key : object
                The key to search for in the list.

            Returns
            -------
            bool
                True if the key is found in the list or within a tuple element.

            Raises
            ------
            TypeError
                If the key is not a string.
            """
            # Check for an exact tuple match if querying a tuple.
            if isinstance(key, tuple):
                return super().__contains__(key)
            # Check for a string match
            elif not isinstance(key, str):
                raise TypeError(f"Key {key} is not a string.")

            # Check for non-relabel match
            for i in range(self.__len__()):
                # Check for direct match
                if self[i] == key:
                    return True
                # Check for tuple sub-item match
                elif isinstance(self[i], tuple):
                    for k in self[i]:
                        if k == key:
                            return True

            # Check for relabel match
            if self._parent.relabel:
                # Requires levels of iterations of relabel checking;
                # for the summary_param_list and the search key.

                # For every item in the list
                for i in range(self.__len__()):
                    # Iterate over all item/subitems in the summary_param_list
                    item = self[i]
                    if not isinstance(item, tuple):
                        item = (item,)
                    for sub_item in item:
                        # Check for relabel synonymous match
                        if (
                            key in self._parent.RELABELS_REVERSE
                        ):  # Key is a relabel value
                            rev_val: str | tuple[str, ...] = (
                                self._parent.RELABELS_REVERSE[key]
                            )
                            if (
                                rev_val == sub_item
                            ):  # one to one relabel:item match, covers strings.
                                return True
                            elif isinstance(rev_val, tuple):
                                for k in rev_val:
                                    if k == sub_item:
                                        return True

                        # Check for key in each relabel key/tuple
                        if key in self._parent.RELABELS:
                            value = self._parent.RELABELS[key]
                            if value == sub_item:
                                return True
            return False

        @override
        def index(
            self,
            value: str | tuple[str, ...],
            start: typing.SupportsIndex = 0,
            stop: typing.SupportsIndex = sys.maxsize,
        ) -> int:
            """
            Find the index of the value.

            Also supports synonymous tuple items,
            by searching tuple elements for the value.

            Parameters
            ----------
            value : str | tuple[str, ...]
                The value or equivalent tuple of values to find.
            start : typing.SupportsIndex, optional
                The starting index to search from, by default 0.
            stop : typing.SupportsIndex, optional
                The final index to search to, by default sys.maxsize.

            Returns
            -------
            int
                The index where the item is found.

            Raises
            ------
            ValueError
                If the value is not found in the list.
            """
            if stop == sys.maxsize:
                stop = self.__len__()
            for i in range(start, stop):
                element = self[i]
                # Direct match
                if element == value or isinstance(element, tuple) and value in element:
                    return i
            raise ValueError(f"{value} is not in the list, or within a tuple element.")

    _COLUMN_ASSIGNMENTS: assignments_type
    """Internal dictionary of x,y,x_err,y_err column assignments."""
    _ALLOWED_EXTENSIONS: list[str] = []
    """Internal list of allowed file extensions for the parser."""
    _SUMMARY_PARAM_RAW_NAMES: summary_param_list
    """Internal list of important parameter strings for displaying file summary information."""
    _RELABELS: relabels_dict = relabels_dict()
    """Internal dictionary of data label equivalence. Allowed to use lists for original labels, due to
    synonymous names (i.e. equipment replacement with a new channel name)."""
    _RELABELS_REVERSE: dict[str, str | tuple[str, ...]] = {}
    """Internal dictionary of label equivalence in reverse."""
    _dtype_channels: list[dtype] = []
    """A list of mapped datatype channels `dtype` found in RELABELS. These add dynamic accessor methods at runtime."""
    _relabel = True
    """Internal class boolean for relabels, by default True."""

    def __new__(
        mcls: type["parserMeta"],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "parserMeta":
        """
        Create a new parser classes with the necessary attributes for class methods/properties.

        Functionally checks and verifies the user defined properties of the parser class, including
        - `ALLOWED_EXTENSIONS`,
        - `COLUMN_ASSIGNMENTS`,
        - `SUMMARY_PARAM_RAW_NAMES`, and
        - `RELABELS`,
        and reassigns their values to hidden properties, i.e. `cls._ALLOWED_EXTENSIONS` that enables
        property dynamic functionality.

        Parameters
        ----------
        mcls : type[parserMeta]
            The metaclass type reference.
        name : str
            The name of the class.
        bases : tuple[type, ...]
            The base classes of the class.
        namespace : dict[str, Any]
            The namespace of the class - it's attributes names and values.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        parserMeta
            The new `parser class`, with the necessary attributes for class methods/properties.

        Raises
        ------
        ValueError
            If key properties (`ALLOWED_EXTENSIONS`, `COLUMN_ASSIGNMENTS`) are not defined, then a ValueError is raised.
        """

        # Perform checks on parsers that implement parserBase.
        if name != "parserBase":
            # If class does not define the important parameters, then set to empty list.
            if "SUMMARY_PARAM_RAW_NAMES" not in namespace:
                namespace["SUMMARY_PARAM_RAW_NAMES"] = []
            if "RELABELS" not in namespace:
                namespace["RELABELS"] = {}

            # Raise error if necessary variables are not defined.
            for prop in ["ALLOWED_EXTENSIONS", "COLUMN_ASSIGNMENTS"]:
                if prop not in namespace:
                    raise ValueError(f"Class {name} does not define {prop}.")

            # Rename attributes, avoid overriding property.
            for prop in [
                "ALLOWED_EXTENSIONS",
                "COLUMN_ASSIGNMENTS",
                "SUMMARY_PARAM_RAW_NAMES",
                "RELABELS",
            ]:
                namespace[f"_{prop}"] = namespace[
                    prop
                ]  # Adjust assignments to an internal variable i.e. _ALLOWED_EXTENSIONS
                del namespace[prop]  # Remove old assignment

            # Validate column assignments, and assign defaults if not provided.
            parserMeta.__validate_assignments(namespace["_COLUMN_ASSIGNMENTS"])

            if "_dtype_channels" in namespace:
                raise AttributeError(
                    f"`{name}._dtype_channels` is reserved for a list of `dtype` that are found in `{name}.RELABELS`."
                )
            _dtype_channels: list[dtype] = []
            """A list of `dtype` """

            # Check no overlapping duplicate values & create a copy of the RELABELS dictionary in reverse.
            reverse_dict = {}
            generic_msg = (
                r"\nPlease provide unique key,values for each key-value pair. Synonymous key names can be provided as a list."
                + r"\nI.e. {'old_name': 'new_name', ('old_name_2', 'old_name_3'): 'new_name_2'}"
            )
            for key, value in namespace["_RELABELS"].items():
                # Process the key/value for dtype channels.
                key_match = False
                if isinstance(key, dtype):
                    if key in _dtype_channels:
                        raise ValueError(
                            rf"Duplicate - dtype key '{key}' in class `{name}.RELABELS` dictionary with value '{namespace['_RELABELS'][key]}'."
                            + generic_msg
                        )
                    else:
                        _dtype_channels.append(key)
                        key_match = True
                elif isinstance(key, tuple):
                    for k in key:
                        if isinstance(k, dtype):
                            if k in _dtype_channels:
                                raise ValueError(
                                    rf"Duplicate - dtype key '{k}' in class `{name}.RELABELS` dictionary with value '{namespace['_RELABELS'][key]}'."
                                    + generic_msg
                                )
                            else:
                                _dtype_channels.append(k)
                                key_match = True

                if not key_match and isinstance(value, dtype):
                    if value in _dtype_channels:
                        raise ValueError(
                            rf"Duplicate - dtype value '{value}' in class `{name}.RELABELS` dictionary with key '{key}'."
                            + generic_msg
                        )
                    else:
                        _dtype_channels.append(value)

                # The value item has already been registered
                if value in reverse_dict:
                    raise ValueError(
                        rf"Duplicate - value '{value}' in class `{name}.RELABELS` dictionary with keys '{reverse_dict[value]}' and '{key}'."
                        + generic_msg
                    )
                # The key matches a value from another pair.
                elif isinstance(key, str) and (
                    key in reverse_dict
                ):  # Another value matches the considered key
                    raise ValueError(
                        rf"Overlap - key:value pair '{key}':'{value}' in class `{name}.RELABELS` "
                        + rf"dictionary overlaps with key:value pair '{reverse_dict[key]}':'{key}'."
                        + generic_msg
                    )
                # A key element matches the value from another pair.
                elif isinstance(
                    key, tuple
                ):  # Another value matches a tuple-key sub-element.
                    for k in key:
                        if k in reverse_dict:
                            raise ValueError(
                                rf"Overlap - key:value pair '{key}':'{value}' with subkey '{k}' as an overlapping entry in class'{name}.RELABELS' "
                                + rf"dictionary with key:value pair '{reverse_dict[k]}':'{k}'."
                                + generic_msg
                            )
                # Add the key:value pair to the reverse dictionary.
                reverse_dict[value] = key

            # Check no overlap in keys / tupled keys.
            key_set = set()
            for key in namespace["_RELABELS"].keys():
                if isinstance(key, str):
                    key2 = (key,)
                else:
                    key2 = key
                for k in key2:
                    if k in key_set:
                        raise ValueError(
                            f"Duplicate - key '{key}' is a duplicate key entry in `{name}.RELABELS` dictionary."
                            if key2 != key
                            else f"Duplicate - key '{k}' from tuple '{key}' is a duplicate key entry in `{name}.RELABELS` dictionary."
                        )
                    key_set.add(k)
            del key_set

            namespace["_RELABELS_REVERSE"] = reverse_dict

            # Convert _RELABELS to a relabels_dict
            namespace["_RELABELS"] = parserMeta.relabels_dict(namespace["_RELABELS"])

            # Add the _dtype_channels to the namespace:
            namespace["_dtype_channels"] = _dtype_channels

        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __init__(
        cls: type,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        # **kwds: Any,
    ):
        # Run super initialisation.. perhaps not necessary for ABCmeta / object?
        super().__init__(name, bases, namespace)

        # Gather internal parser methods at class creation for use in file loading.
        cls.parse_functions: list[Callable[..., parse_fn_ret_type]] = []
        """List of recognised parser methods for the class."""
        cls.parse_recent_success: dict[str, Callable[..., parse_fn_ret_type]] = {}
        """
        A mapping between the most recent successful parser method for each filetype for the class.

        Enables the faster loading of similar files that have already been successfully loaded.
        """

        cls._parse_kwargs: dict[Callable[..., parse_fn_ret_type], dict[str, Any]] = {}
        """For each parsing function, a dictionary of previously used keyword arguments to load the file."""

        cls._reduction_kwargs: dict[
            Callable[..., parse_fn_ret_type], dict[str, Any]
        ] = {}
        """A dictionary of previously used keyword arguments to reduce parser data and create the scan object."""

        # Convert the list to a `summary_param_list`, linked to the parser class.
        if cls.__name__ != "parserBase":
            cls.SUMMARY_PARAM_RAW_NAMES = parserMeta.summary_param_list(
                cls.SUMMARY_PARAM_RAW_NAMES, parent=cls
            )

        # Check for parser methods in the class and add to the parse_functions list.
        for fn_name in dir(cls):
            if fn_name.startswith("parse_"):
                fn = getattr(cls, fn_name)
                if callable(fn):
                    # Get the argument names of the function.
                    arg_names = fn.__code__.co_varnames[: fn.__code__.co_argcount]

                    # Check the parameters of each function match requirements.
                    if type(fn) is types.FunctionType:  # Static methods
                        # First parameter must always be file.
                        if arg_names[0] != "file":
                            raise NameError(
                                f"First argument of static parser method must be 'file'. Is {arg_names[0]}."
                            )
                        # Second optional parameter is header_only.
                        # if len(arg_names) == 2 and arg_names[1] != "header_only":
                        #     raise TypeError(
                        #         f"Second (optional) argument of static parser method must be 'header_only'. Is {arg_names[2]}."
                        #     )
                        cls.parse_functions.append(fn)
                    elif type(fn) is types.MethodType:  # Class methods
                        if len(arg_names) < 2:
                            raise NameError(
                                f"Parser method must have a minimum 2-3 arguments: 'cls', 'file' and (optional) 'header_only'. Has {arg_names}"
                            )
                        if arg_names[0] != "cls":
                            raise NameError(
                                f"First argument of parser method must be 'cls', i.e. the class. It is instead {arg_names[0]}."
                            )

                        if arg_names[1] != "file":
                            raise NameError(
                                f"Second argument of parser method must be 'file'. Is {arg_names[1]}."
                            )
                        # Second optional parameter is header_only.
                        # if len(arg_names) == 3 and arg_names[2] != "header_only":
                        #     raise TypeError(
                        #         f"Third (optional) argument of parser method must be 'header_only'. Is {arg_names[2]}."
                        #     )
                        cls.parse_functions.append(fn)

        # Check the number of parser methods is non-zero.
        if cls.__name__ != "parserBase" and len(cls.parse_functions) == 0:
            raise AttributeError(f"No parser methods found in `{cls.__name__}` class.")

    @property
    def ALLOWED_EXTENSIONS(cls) -> list[str]:
        """
        The allowed extensions for the parser.

        'parserBase.file_parser' will check validity of file extensions.
        Inheriting parser instance will need to deal with each filetype
        in the file_parser method.

        Returns
        -------
        list[str]
            List of allowed extensions. Read only.

        Examples
        --------
        synchrotron_parser.ALLOWED_EXTENSIONS = [".dat", ".txt", ".csv"]
        """
        return cls._ALLOWED_EXTENSIONS

    @staticmethod
    def __validate_assignments(assignments: assignments_type) -> None:
        """
        Validation method for column assignments for a parser and provides defaults.

        Assignments always correspond to 1D data channels; higher dimensional data ought to be reduced before assignments get used.

        Validates by checking the custom string label entries for assigning data columns to a scan object.
        Requires `x`, `y` keys, and can optionally contain `y_errs`, and `x_errs` keys.

        If `y_errs` and `x_errs` are defaulted to `None` if not provided.

        Parameters
        ----------
        assignments : dict[str, str | tuple[str, ...] | None]
            Dictionary of column assignments.

        Raises
        ------
        ValueError
            'x' assignment not found in assignments.
        ValueError
            'x' assignment is not a string.
        ValueError
            'y' assignment is not a tuple or string.
        ValueError
            'y' assignment is a tuple, but subelement is not a string.
        ValueError
            'y_errs' assignment is not a tuple, string, or None.
        ValueError
            'y_errs' assignment is a tuple, but subelement is not a string.
        ValueError
            'y_errs' assignment does not match length of 'y' assignment.
        ValueError
            'x_errs' assignment is not a string or None.
        """
        # X
        if "x" not in assignments:
            raise ValueError("'x' assignment not found in assignments.")
        x = assignments["x"]
        if not (isinstance(x, str) or isinstance(x, tuple)):
            raise ValueError(
                f"'x' assignment {x} is not a string or a tuple of (synonymous) strings."
            )

        # Y
        if "y" not in assignments:
            raise ValueError("'y' assignment not found in assignments.")
        y = assignments["y"]
        if not isinstance(y, (tuple, str, list)):
            raise ValueError(
                f"'y' assignment {y} is not a string, a tuple of (synonymous) strings, or a list of such strings/tuples."
            )
        if isinstance(y, list):
            for y_sub in y:
                if not isinstance(y_sub, (str, tuple)):
                    raise ValueError(
                        f"`y` list element {y_sub} is not a string or a tuple of (synonymous) strings."
                    )
                if isinstance(y, tuple):
                    for y_sub_sub in y_sub:
                        if not isinstance(y_sub_sub, str):
                            raise ValueError(
                                f"`y` tuple `{y_sub}` element `{y_sub_sub}` is not a string."
                            )
        if isinstance(y, tuple):
            for y_sub in y:
                if not isinstance(y_sub, str):
                    raise ValueError(
                        f"`y` tuple `{y}` element `{y_sub}` is not a string."
                    )

        # Yerrs
        if "y_errs" not in assignments:
            assignments["y_errs"] = None  # set a default value.
        y_errs = assignments["y_errs"]
        if not (isinstance(y_errs, (list, tuple, str)) or y_errs is None):
            raise ValueError(
                f"'y_errs' assignment {y_errs} is not a string, tuple of (synonymous) strings, None, or a list of such types."
            )
        if isinstance(y_errs, list):
            if not isinstance(y, list):
                raise ValueError(
                    f"'y_errs' list assignment {y_errs}\n is a list, but 'y' assignment {y} is not."
                )
            if len(y_errs) != len(y):
                raise ValueError(
                    f"'y_errs' list assignment {y_errs} does not match length ({len(y_errs)}) of 'y' list assignment {y} ({len(y)})."
                )
            for y_sub in y_errs:
                if not (isinstance(y_sub, (str, tuple)) or y_sub is None):
                    raise ValueError(
                        f"'y_errs' list element `{y_sub}` is not a string or a tuple of (synonymous) strings, or None."
                    )
                if isinstance(y_sub, tuple):
                    for y_sub_sub in y_sub:
                        if not isinstance(y_sub_sub, str) or y_sub_sub is None:
                            raise ValueError(
                                f"'y_errs' list, tuple element `{y_sub_sub}` is not a string or None."
                            )
        elif isinstance(y_errs, tuple):
            for y_sub in y_errs:
                if not isinstance(y_sub, str):
                    raise ValueError(
                        f"'y_errs' tuple element `{y_sub}` is not a string."
                    )

        # Xerrs
        if "x_errs" not in assignments:
            assignments["x_errs"] = None
        x_errs = assignments["x_errs"]
        if not (isinstance(x_errs, (str, tuple)) or x_errs is None):
            raise ValueError(
                f"'x_errs' assignment {x_errs} is not a string, tuple of (synonymous) strings, or None."
            )

    @property
    def COLUMN_ASSIGNMENTS(cls) -> assignments_type:
        """
        The assignments of parser data columns to intelligent scan columns.

        'parserBase.to_scan' will use this mapping to construct the scan object.
        Assignments can be a single column name, or a list of column names.
        y_errs and x_errs can be None if not present in the data.
        Use a list to include equivalent column names. Scan_base will use the first column name found.

        Returns
        -------
        dict[str, str | list[str] | tuple[str | list[str]] | None]
            A dictionary of column assignments.
            Requires 'x', 'y' keys, and can have optional 'y_errs' and 'x_errs' keys.

        Examples
        --------
        synchrotron_parser.COLUMN_ASSIGNMENTS ={
            "x": "Data_Column_1_Label",
            "y": [
                "Data_Column_2_Label",
                ("Data_Column_3_Label", "Alternative_Column_3_Label"),
                "Data_Column_4_Label",
            ],  # or "Data_Column_2_Label"
            "y_errs": [
                "Data_Column_5_Label",
                "Data_Column_6_Label",
                None, # If specifying errors, use None for data columns that don't have corresponding errors.
            ],
            "x_errs": None,
        }
        """
        return cls._COLUMN_ASSIGNMENTS

    @COLUMN_ASSIGNMENTS.setter
    def COLUMN_ASSIGNMENTS(cls, assignments: assignments_type) -> None:
        # Check validity
        cls.__validate_assignments(assignments)
        # Assign
        cls._COLUMN_ASSIGNMENTS = assignments

    @property
    def SUMMARY_PARAM_RAW_NAMES(cls) -> summary_param_list:
        """
        A list of important parameters, for displaying file summary information.

        Used by GUI methods for displaying summary file information.

        Parameters
        ----------
        summary_params : list[str | tuple[str, ...]]
            List of important parameter strings or tuple of synonymous strings,
            matching keys that exist in 'parserBase.params'.

        Returns
        -------
        list[str | tuple[str, ...]]
            List of important parameter strings or tuple of (synonymous) strings,
            matching keys in 'parserBase.params'.
        """
        return cls._SUMMARY_PARAM_RAW_NAMES

    @SUMMARY_PARAM_RAW_NAMES.setter
    def SUMMARY_PARAM_RAW_NAMES(
        cls,
        summary_params: (
            Iterable[str | tuple[str, ...]] | "parserMeta.summary_param_list"
        ),
    ) -> None:
        if isinstance(summary_params, cls.summary_param_list):
            cls._SUMMARY_PARAM_RAW_NAMES = summary_params
        else:
            cls._SUMMARY_PARAM_RAW_NAMES = parserMeta.summary_param_list(
                summary_params, parent=cls
            )

    @property
    def summary_param_names(cls) -> list[str]:
        """
        A list of important parameter names of the data file.

        Sources from `cls.SUMMARY_PARAM_RAW_NAMES`.
        Returns singular values if summary parameter is a tuple.

        If `cls.relabels` and names are defined in cls.RELABELS,
        then relabelled names are returned (and prioritised in tuples).

        Returns
        -------
        list[str]
            List of important parameter names.

        Examples
        --------
        >>> synchrotron_parser.relabel = False
        >>> synchrotron_parser.summary_param_names
        ['MCV01DA240EN', 'MCV01DA245TI', 'MCV01DA252IN']

        >>> synchrotron_parser.relabel = True
        >>> synchrotron_parser.summary_param_names
        ['Photon Energy', 'Time', 'Intensity']
        """
        if cls.relabel:
            names: list[str] = []
            for name in cls.SUMMARY_PARAM_RAW_NAMES:
                # If the summary name is a tuple, check each sub-name for a relabel and use the first one.
                if isinstance(name, tuple):
                    found = False
                    for sub_name in name:
                        sub_name: str
                        if sub_name in cls.RELABELS:
                            names.append(cls.RELABELS[sub_name])
                            found = True
                            break
                    if not found:
                        names.append(name[0])
                elif isinstance(name, str):  # string
                    if name in cls.RELABELS:
                        names.append(cls.RELABELS[name])
                    else:
                        names.append(name)
                # Ignore non-string/tuple[string] names.
            return names
        else:
            # Generate a list of names from the summary_param_list.
            names: list[str] = []
            for name in cls.SUMMARY_PARAM_RAW_NAMES:
                if isinstance(name, tuple):
                    names.append(name[0])
                elif isinstance(name, str):
                    names.append(name)
            return names

    @property
    def RELABELS(cls) -> relabels_dict:
        """
        A mapping to renames labels to more useful names (optional property).

        'parserBase.to_scan' will use this to relabel the scan params or column labels.
        Assignments should be dict entries in the form of 'old_label' : 'new_label',
        or `(old_label_1, old_label_2)` : 'new_label'.

        This property is a class attribute, not unique to each parser instance.

        Parameters
        ----------
        relabels : dict[str | tuple[str, ...], str]
            Dictionary of old labels to new labels, matching names in 'parserBase.labels' or 'parserBase.params'.

        Returns
        -------
        dict[str | tuple[str, ...], str]
            Dictionary of old labels to new labels, matching names in 'parserBase.labels' or 'parserBase.params'.

        Raises
        ------
        ValueError
            Raised if a duplicate value is found in the RELABELS dictionary.
        """
        return cls._RELABELS

    @RELABELS.setter
    def RELABELS(
        cls, relabels: dict[str | tuple[str, ...], str] | relabels_dict
    ) -> None:
        # Check no duplicate values & create a copy of the RELABELS dictionary in reverse.
        reverse_dict = {}
        for key, value in relabels.items():
            if value in reverse_dict:
                raise ValueError(
                    f"Duplicate value '{value}' in RELABELS dictionary with keys `{reverse_dict[value]}` and {key}."
                    + "\nPlease provide unique values for each key-value pair. Synonymous key names can be provided as a tuple."
                    + "\nI.e. {'old_name': 'new_name', ('old_name_2', 'old_name_3'): 'new_name_2'}"
                )
            reverse_dict[value] = key

        # Update the class attributes:
        cls._RELABELS_REVERSE = reverse_dict
        # Create a relabels dictionary, mapping tuples of synonymous keys to the same value.
        if not isinstance(relabels, cls.relabels_dict):
            relabels = cls.relabels_dict(relabels)
        cls._RELABELS = relabels

    @property
    def RELABELS_REVERSE(cls) -> dict[str, str | tuple[str, ...]]:
        """
        The reverse of the RELABELS dictionary.

        Possible because the RELABELS dictionary requires a one-to-one mapping.

        Returns
        -------
        dict[str, str]
            The reverse of the RELABELS dictionary.
        """
        return cls._RELABELS_REVERSE

    @property
    def relabel(cls) -> bool:
        """
        A property to enable relabelling `params` and `labels` defined in cls.RELABELS.

        If True, by the provided params-names and column-headers are relabelled.
        If False, then the original params-names and column-headers are used by the class.

        Parameters
        ----------
        value : bool
            True if the parser object returns relabelled params/column-headers.

        Returns
        -------
        bool
            True if the parser object returns relabelled params/column-headers.
        """
        return cls._relabel

    @relabel.setter
    def relabel(cls, value) -> None:
        cls._relabel = value


class parserBase(abc.ABC, metaclass=parserMeta):
    """
    Abstract general class that parses raw files to acquire data/meta information.

    Requires implementation of methods `parser_<filetype>` methods that can be used for
    the parserBase.`file_parser` method, as well as the property attributes `ALLOWED_EXTENSIONS`,
    `COLUMN_ASSIGNMENTS` and (optional) `RELABELS`. These properties modify the behaviour
    of the parser object.

    Parameters
    ----------
    file : str | FileIO | None, optional
        An FileIO object or str path of the data file, by default None.
    header_only : bool, optional
        If True, then only the header of the file loaded, by default False.
    relabel : bool, optional
        If True, then column and parameter labels are returned as more useful
        names defined in 'parserBase.RELABELS', by default False.
    **kwargs
        Additional keyword arguments that will be passed to the `file_parser` method.

    Attributes
    ----------
    filepath : str | None
        The filepath of the data file.
    data : npt.NDArray | None
        The data array of the file.
    units : list[str] | None
        The units of the data columns.
    labels : list[str] | None
        The labels of the data columns. Affected by relabel.
    params : dict[str, Any]
        Additional parameters of the data file. Affected by relabel.
    ALLOWED_EXTENSIONS : list[str]
        (class attribute) Allowable extensions for the parser.
    COLUMN_ASSIGNMENTS : dict[str, str | list[str] | None]
        (class attribute) Assignments of scan input variables to column names.
    SUMMARY_PARAM_RAW_NAMES : list[str]
        (class attribute) A list of important parameters, for displaying file summary information.
    RELABELS : dict[str, str]
        (class attribute) Renames labels to more useful names (optional property) when calling to_scan().
    summary_params : dict[str, Any]
        (class attribute) Returns a dictionary of important parameters of the data file. Affected by relabel.

    Raises
    ------
    TypeError
        If `parserBase` is instantiated directly, a TypeError is raised, as the class is abstract.
    """

    # Set attributes to class properties. Necessary for calling class properties on the instance.
    ALLOWED_EXTENSIONS = parserMeta.ALLOWED_EXTENSIONS
    COLUMN_ASSIGNMENTS = parserMeta.COLUMN_ASSIGNMENTS
    SUMMARY_PARAM_RAW_NAMES = parserMeta.SUMMARY_PARAM_RAW_NAMES
    RELABELS = parserMeta.RELABELS
    RELABELS_REVERSE = parserMeta.RELABELS_REVERSE
    # These methods are not class methods copies, but rather return names existing on the object instance.
    # summary_param_names_with_units = parserMeta.summary_param_names_with_units
    # summary_param_names = parserMeta.summary_param_names

    class param_dict(dict):
        """
        A dictionary extension that deals with relabelling of parameters.

        Requires a `parent` keyword argument to access the `parserBase.RELABELS` dictionary.

        When `parent.relabel` is active, `param_dict` remaps `parent.RELABELS` values
        back to the keys when getting (`__getitem__`), setting (`__setitem__`)
        and contains (`__contains__`) methods are called, allowing seamless access to
        the parameter dictionary.

        If `parent.relabel` is False, then the dictionary behaves as a standard dictionary, but
        doesn't allow relabelled keys to be set that already have equivalents in the original dictionary.

        Parameters
        ----------
        map : dict[str, Any] | Iterable[tuple[str, Any]]
            The dictionary of parameters.
        parent : parserBase
            The parent parser object. Required keyword argument.
        **kwargs : Any
            Additional keyword arguments.
        """

        __slots__ = ("_parent",)  # Do not allow attributes other than the parent.

        def __init__(
            self: Self,
            map: "SupportsKeysAndGetItem[str, Any]" | Iterable[tuple[str, Any]],
            *,
            parent: "parserBase",
            **kwargs: Any,
        ) -> None:
            # Initialise the dictionary
            super().__init__(map, **kwargs)
            # Store a reference to the parser object, and the "RELABELS" dictionary
            self._parent: parserBase = parent

        @override
        def keys(self) -> KeysView | dict_keys:  # type: ignore - cannot instantiate "dict_keys" though this is the C return type.
            """
            The `keys` method for the parameter dictionary.

            If `parent.relabel` is True, then the keys are returned with relabelled keys.
            Otherwise defaults to `dict.keys` behaviour.

            Returns
            -------
            dict_keys
                The keys of the parameter dictionary.
            """
            if self._parent.relabel:
                keys = super().keys()
                return KeysView(
                    [
                        (
                            key
                            if key not in self._parent.RELABELS and isinstance(key, str)
                            else self._parent.RELABELS[key]
                        )
                        for key in keys
                    ]  # type: ignore - handles a list of strings fine for the mapping.
                )
            return super().keys()

        @override
        def items(self) -> dict_items | ItemsView:  # type: ignore - cannot instantiate "dict_items" though this is the C return type.
            """
            The `items` method for the parameter dictionary.

            If `parser.relabel` is True, then the items are returned with relabelled keys.
            Otherwise the default dict items method is used.

            Returns
            -------
            dict_items | ItemsView
                The items of the parameter dictionary.
            """
            if self._parent.relabel:
                items = super().items()
                new_items: list[tuple[str, str]] = []
                for key, value in items:
                    if key in self._parent.RELABELS:
                        new_items.append((self._parent.RELABELS[key], value))
                    else:
                        new_items.append((key, value))
                return ItemsView(new_items)  # type: ignore - handles a list of tuples fine for the mapping.
            else:
                return super().items()

        @override
        def __getitem__(self, key: str) -> str:
            """
            Get item method for the parameter dictionary.

            The equivalent `parser.RELABELS` parameter name is also checked
            (and used if existing) for a lookup.

            Parameters
            ----------
            key : str | tuple[str, ...]
                The key of the parameter.

            Returns
            -------
            str
                The value of the parameter.
            """
            result = None
            try:
                result = super().__getitem__(key)
            except KeyError as e:
                if not self._parent.relabel:
                    raise e
                else:
                    # If the key is not found, check if the key is in RELABELS
                    # and use the corresponding value.
                    if (
                        key in self._parent.RELABELS
                    ):  # `relabels_dict.__contains__` handles tuples and strings.
                        # If tuple, __get__ isolates a single key from the tuple.
                        relabel_key = self._parent.RELABELS[key]
                        relabel_reverse_key = self._parent.RELABELS_REVERSE[relabel_key]

                        # Attempt to access the param using all known matches (RELABEL value first!)
                        if isinstance(relabel_reverse_key, tuple):
                            k_possible = (relabel_key, key, *relabel_reverse_key)
                        else:
                            k_possible = (relabel_key, relabel_reverse_key)
                        for k in k_possible:
                            try:
                                result = super().__getitem__(k)
                                break
                            except KeyError:
                                pass
                        if result is None:
                            raise KeyError(
                                f"Key {key} not found in the `{self._parent.__class__.__name__}.params` after searching: '[{k_possible}'."
                            )
                    else:
                        raise KeyError(
                            f"Key {key} not found in the `{self._parent.__class__.__name__}params` or `{self._parent.__class__.__name__}.RELABELS` dictionary."
                        )
            # # Check if the result is a RELABELS key, and if so, return the value.
            # if result in self._parent.RELABELS:
            #     return self._parent.RELABELS[result]
            # # Otherwise, return the result.
            return result

        @override
        def __setitem__(self, key: str, value: str):  # numpydoc ignore=RT01
            """
            Set item method for the parameter dictionary.

            The equivalent `parser.RELABELS` parameter name is also checked
            (and used if existing) prior to creating a new key.
            Operates whether `parser.relabel` is True or False.

            Parameters
            ----------
            key : _KT
                The key of the parameter.
            value : _VT
                The value of the parameter.

            Raises
            ------
            KeyError
                If the key is in the `parser.RELABELS` dictionary and `parser.relabel` is False.
            """
            # Check if the key is directly in the dictionary:
            if super().__contains__(key):  # use old __contains__ method
                return super().__setitem__(key, value)

            # Check if the key is in the RELABELS dictionary and use the value
            if key in self._parent.RELABELS:
                # Use the relabel key
                relabel_key = self._parent.RELABELS[key]
                # Check if the key is the raw key in the dictionary.
                if super().__contains__(relabel_key):
                    if self._parent.relabel:
                        return super().__setitem__(relabel_key, value)
                    else:
                        raise KeyError(
                            f"Key `{key}` already exists in the `{self._parent.__class__.__name__}.params` dictionary as"
                            + f" the RELABELS value `{relabel_key}`. Because `parser.relabel` is False, the key cannot be set."
                        )
                # Check if any of the synonymous keys are already in the dictionary.
                contained = False
                keyset = self._parent.RELABELS_REVERSE[relabel_key]
                if isinstance(keyset, str):
                    # As the relabelled value didn't match, neither does it's key.
                    if self._parent.relabel:
                        # Use the relabel key to set the value.
                        return super().__setitem__(relabel_key, value)
                    else:
                        # No relabelling yet, add new key as is.
                        return super().__setitem__(key, value)

                # keyset is a tuple
                assert isinstance(keyset, tuple)
                for k in keyset:
                    if super().__contains__(k):
                        contained = True
                        break
                # k is the key in the dictionary.
                # Check if relabelling or not...
                if not contained:
                    if self._parent.relabel:
                        # Use the relabel key to set the value.
                        return super().__setitem__(relabel_key, value)
                    else:
                        # No relabelling yet, add new key as is.
                        return super().__setitem__(key, value)
                else:
                    # k is contained in the dictionary.
                    if self._parent.relabel:
                        # Use the existing key to set the value.
                        return super().__setitem__(k, value)  # type: ignore - k is a string if `contained` == True.
                    else:
                        raise KeyError(
                            f"Key `{key}` already exists in the `{self._parent.__class__.__name__}.params` dictionary as"
                            + f" the RELABELS key `{relabel_key}`. Because `parser.relabel` is False, the key cannot be set."
                        )

            # Key doesn't exist and not found in RELABELS, so create a new key.
            return super().__setitem__(key, value)

        @override
        def __contains__(self, key: object) -> bool:
            """
            The `contains` / `in` method for the parameter dictionary.

            The equivalent `parser.RELABELS` parameter name is also checked
            (and used if existing) for a key match.
            Operates whether `parser.relabel` is True or False.
            Does not accept non-string keys.

            Parameters
            ----------
            key : object
                The parameter key.

            Returns
            -------
            bool
                True if the parameter is in the dictionary.
            """
            # Reject non-string keys - should only be looking for a string value.
            if not isinstance(key, str):
                return False
            # Check if the key is in the dictionary.
            if super().__contains__(key):
                return True

            # Check if the key is in the RELABELS dictionary, and if the value is in the dictionary.
            elif self._parent.relabel:
                for k, v in self._parent.RELABELS.items():
                    # Check if RELABEL value matches lookup key
                    if (
                        key
                        == v  # Check key-v first to prevent recursive expensive call.
                        and self.__contains__(k)
                    ):  # Recursive call to handle tuple keys defined in RELABELS.
                        return True

                    # Tuple of strings key
                    if isinstance(k, tuple):
                        # Do not need to check tuple match to key, as key only str.
                        # Check if tuple element matches key.
                        is_contained = False
                        # Also check the relabel value 'v' for a match,
                        # despite checked for contains above.
                        for k_sub in (v, *k):
                            if k_sub == key:
                                is_contained = True
                                break
                        # Check all keys if tuple equivalence condition met.
                        if is_contained:
                            for k_sub in (v, *k):
                                if super().__contains__(k_sub):
                                    return True
                    elif isinstance(k, str):
                        # Already checked
                        pass
                    else:
                        raise ValueError(
                            f"`parser.RELABELS` key `{key}` not str or tuple of str."
                        )
            # Key not found
            return False

        @override
        def __str__(self) -> str:
            """
            String representation of the parameter dictionary.

            Returns as a string of key-value pairs, with relabelling if enabled.

            Returns
            -------
            str
                String representation of the parameter dictionary, with key values
                relabelled if `parser.relabel` is True.
            """
            if not self._parent.relabel:
                return super().__str__()
            else:
                retstr: str = "{"
                for key, value in self.items():
                    if key in self._parent.RELABELS:
                        retstr += f"{self._parent.RELABELS[key]}: {value}, "
                    else:
                        retstr += f"{key}: {value}, "
                # Appropriately terminate the string and return.
                return retstr[:-2] + "}"

    def __init__(
        self,
        file: str | typing.IO | None,
        header_only: bool = False,
        relabel: bool | None = None,
        **kwargs,
    ) -> None:
        if type(self) is parserBase:
            raise TypeError("Cannot instantiate abstract class `parserBase`.")

        # parserMeta super
        super().__init__()

        # Initialise variables
        self._filepath: str | None
        if isinstance(file, str):
            if os.path.isfile(file):
                self._filepath = file
            else:
                raise FileNotFoundError(f"Filepath '{file}' does not exist.")
        elif isinstance(file, (typing.IO, io.FileIO)):
            fname = file.name
            if os.path.isfile(fname):
                self._filepath = fname
            else:
                raise FileNotFoundError(f"Filepath '{fname}' does not exist.")
        else:
            self._filepath = None

        """The file path of the data"""
        self.data: npt.NDArray | tuple[npt.NDArray | None, ...] | None = None
        """
        Data loaded through the `parser.load()` function. Can be a single (2D) array,
        with indexes (NEXAFS energy, channel #), or a tuple of arrays for
        multi-dimensional/multi-scan files where the indexes match
        (NEXAFS energy, ..., <other setting / indepedent variables>, ..., channel #).
        """

        self._labels: (
            list[str | dtype | None]
            | tuple[list[str | dtype | None] | str | dtype, ...]
            | None
        ) = None
        """
        Channel Labels loaded through the `parser.load()` function.
        If a tuple should match the length of data
        """

        self.units: list[str | None] | tuple[list[str | None] | None, ...] | None = None
        """
        Units loaded through the `parser.load()` function.
        Should match the shape of the data array.
        """
        self._params: parserBase.param_dict = self.param_dict({}, parent=self)
        """Parameters loaded through the `parser.load()` function."""

        self._parsing_kwargs: dict[str, Any] = kwargs
        """Keyword arguments passed upon instantiation of the object, for re-use later to reload."""

        self._parser_fn: Callable | None = None
        """The most-recent parser function used on the object to load file data, labels, units."""

        # Copy class value at initialisation if None
        self._relabel: bool = relabel
        """
        Whether to use relables or not.
        If None, then the class value is used by default when `obj.relabel` is called.
        """
        # Create a variable to track loading
        self._parsed_data = False
        """Tracks wether the data has been loaded (True), or only the header (False)"""
        self._parser_fn: Callable | None = None
        """The most-recent parser function used on the object to load file data, labels,
            units or parameters. Use `obj._parser_fn.__name__` to get the method name."""

        # Parse data?
        if file is None:
            return
        elif isinstance(file, str):
            if os.path.isfile(file):
                self.load(header_only=header_only, **kwargs)  # Load data
            else:
                raise FileNotFoundError(f"Filepath '{file}' does not exist.")
        else:
            if isinstance(
                file, (IO, typing.IO, io.FileIO, tempfile._TemporaryFileWrapper)
            ):
                self.load(file=file, header_only=header_only, **kwargs)  # Load data
            else:
                raise TypeError(f"Filepath is {type(file)}, not str.")

    @property
    def is_loaded(self) -> bool:
        """
        Whether the parser object has loaded data.

        Not related to whether the header has been loaded.

        Returns
        -------
        bool
            True if the parser object has loaded data.
        """
        return self._parsed_data

    @property
    def relabel(self) -> bool:
        """
        Object property for relabel behaviour to be applied to the parser object, defined in cls.RELABELS.

        If True, by the provided params-names and column-headers (labels) are relabelled.
        If False, then the original params-names and column-headers are used by the class.

        Using `del obj.relabel` resets to class property default behaviour (`cls.relabel` sets a default behaviour.)

        Parameters
        ----------
        value : bool
            Enable (True) or disable (False) the relabel functionality on the object.

        Returns
        -------
        bool
            True if the parser object returns relabelled params/column-headers.
        """
        if self._relabel is None:
            return type(self)._relabel
        else:
            return self._relabel

    @relabel.setter
    def relabel(self, value: bool) -> None:
        self._relabel = value

    @relabel.deleter
    def relabel(self) -> None:
        self._relabel = None

    @property
    def parser_fn(self) -> Callable | None:
        """
        The most-recent parser function used on the object to load file data, labels, units.

        Use `obj.parser_fn.__name__` to get the method name.

        Returns
        -------
        Callable | None
            The most-recent parser function used on the object to load file data, labels, units.
            None if no parser function has been used yet.
        """
        return self._parser_fn

    @property
    def parser_fn_name(self) -> str | None:
        """
        The name of the most-recent parser function used on the object to load file data, labels, units.

        Returns
        -------
        str | None
            The name of the most-recent parser function used on the object to load file data, labels, units.
            None if no parser function has been used yet.
        """
        if self._parser_fn is None:
            return None
        else:
            return self._parser_fn.__name__

    @property
    def filepath(self) -> str:
        """
        The filepath of the data file.

        There is no setter for the filepath attribute, a new instance needs to be created.

        Returns
        -------
        str
            Filepath of the data file or an empty string if not set.
        """
        fp = self._filepath
        return fp if fp is not None else ""

    @property
    def filename(self) -> str:
        """
        The filename of the data file.

        There is no setter for the filename attribute, a new instance needs to be created.

        Returns
        -------
        str
            The filename of the data file, or an empty string if not set.
        """
        fp = self._filepath
        return os.path.basename(fp) if fp is not None else ""

    # TODO: Implement a generic method for relabelling. Should be testable.
    # @staticmethod
    # def _apply_relabels(
    #     labels: list[str | dtype | None] | tuple[list[str | dtype | None] | str | dtype] | str | dtype,
    #     relabels: dict[str | dtype | tuple[str | dtype, ...], str | dtype],
    # ) -> list[str | dtype | None] | tuple[list[str | dtype | None] | str | dtype]:
    #     """
    #     Apply relabelling to a list of labels.

    #     Parameters
    #     ----------
    #     labels : list[str | dtype | None] | tuple[list[str | dtype | None] | str | dtype] | str | dtype
    #         The labels to relabel. Can be a single label, a list of labels, or a tuple of lists of labels.
    #     relabels : dict[str | dtype | tuple[str | dtype, ...], str | dtype]
    #         The relabels dictionary to apply.

    #     Returns
    #     -------
    #     list[str | dtype | None] | tuple[list[str | dtype | None] | str | dtype]
    #         The relabelled labels, in the same format as the input.
    #     """

    @property
    def labels(
        self,
    ) -> (
        list[str | dtype | None]
        | tuple[list[str | dtype | None] | str | dtype | None, ...]
        | None
    ):
        """
        The labels of the data channels.

        If multiple dimensions of data,
        If relabel is True, then the labels are returned as more useful names defined in 'parserBase.RELABELS'.

        Returns
        -------
        list[str | dtype | None] | tuple[list[str | dtype | None] | str | dtype | None, ...] | None
            Labels of the data columns. Can be None if not provided.
            If multiple dimensions of data, returns a tuple of lists.
            If None, no label data exists or has been loaded.
        """
        labels_orig = self._labels
        if not self.relabel or labels_orig is None:
            return labels_orig
        else:
            if not isinstance(labels_orig, tuple):
                labels = (labels_orig,)
            else:
                labels = labels_orig
            # Collect labels if they exist.
            all_relabels = []
            for labi in labels:
                if labi is None:
                    continue
                relabels = []
                for label in labi:
                    if label is None:
                        continue
                    elif isinstance(label, list):
                        relabels.append(
                            [
                                (
                                    self.RELABELS[sub_label]
                                    if sub_label in self.RELABELS
                                    else sub_label
                                )
                                for sub_label in label
                            ]
                        )
                    else:
                        relabels.append(
                            self.RELABELS[label] if label in self.RELABELS else label
                        )
                all_relabels.append(relabels)
            # Restore labels to a single list if singular
            if len(all_relabels) == 1:
                all_relabels = all_relabels[0]
            else:
                # Restore to tuple if multiple dimensions of data.
                all_relabels = tuple(all_relabels)
            return all_relabels

    # No property for units required, as units are not relabelled.

    @property
    def params(self) -> "parserBase.param_dict":
        """
        The parameters of the parser object.

        Parameters are bundled in the dictionary extension class `parserBase.param_dict`.
        If relabel is True, then the dictionary functionality changes significantly.
        - `__contains__`, `__getitem__` and `__setitem__` all recognise equivalent `RELABEL` key/values.
        - When using `__setitem__`, if the key is in `RELABELS` and `relabel` is False, then a KeyError is raised (to avoid duplicate keys).

        Returns
        -------
        parserBase.param_dict
            The parameters of the parser object.
        """
        return self._params

    @property
    def reduction_kwargs(self) -> dict[str, Any] | None:
        """
        The keyword arguments used for reduction of higher dimensional data.

        Stored as a class attribute dictionary, with keys matching the parser function name.
        This allows different reduction parameters to be stored for different parser functions.

        Note that modifications to this dictionary will affect the storage on the class.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Set the keyword arguments used for reduction of higher dimensional data.

        Returns
        -------
        dict[str, Any]
            The keyword arguments used for reduction of higher dimensional data.
        """
        parser_fn = self._parser_fn
        reduction_dict = self.__class__._reduction_kwargs
        if parser_fn is None:
            raise ValueError(
                "Cannot get reduction kwargs before successfully loading data."
            )
        if parser_fn in reduction_dict:
            return reduction_dict[parser_fn]
        else:
            return None

    @reduction_kwargs.setter
    def reduction_kwargs(self, kwargs: dict[str, Any]) -> None:
        parser_fn = self._parser_fn
        if parser_fn is None:
            raise ValueError(
                "Cannot set reduction kwargs before successfully loading data."
            )
        reduction_dict = self.__class__._reduction_kwargs
        reduction_dict[parser_fn] = kwargs

    def reduce(
        self,
        use_prior_params: bool = True,
    ) -> reduction_type:
        """
        Perform a reduction of higher dimensional data.

        Override this function to provide parser-specific reduction of higher dimensional data.
        Recommended using a `match` and `case` statement, and override `self._parser_fn`.

        Parameters
        ----------
        use_prior_params : bool, optional
            Previously used parameters are stored on the class, matching the parser fn.
            If True, then these kwargs are used to perform the reduction. By default True.
        kwargs : dict[str, Any]
            Additional keyword arguments to pass to the reduction function.

        Returns
        -------
        reduced_data : npt.NDArray | None
            The reduced data array, or None if no reduction is performed.
        reduced_labels : list[str | None] | None
            The reduced labels, or None if no reduction is performed.
        reduced_units : list[str | None] | None
            The reduced units, or None if no reduction is performed.
        """
        # Store the kwargs for future use.
        parser_fn = self._parser_fn

        if use_prior_params:
            # Add Logic to load prior parameters for reduction
            # ...
            pass

        # Get existing kwargs
        # reduction_kwargs = self.reduction_kwargs

        # Reduction defaults
        match parser_fn:
            # Use reduction kwargs
            case _:  # Default case for unmatched parser functions.
                # Use any reduction keyword arguments...

                # Save the reduction keyword arguments...
                # self.reduction_kwargs = dict(kw_arg1 = )

                # Return data, labels and units.
                return None, None, None

    def to_scan(
        self,
        use_prior_params: bool = True,
        *,  # Indicate keyword arguments. Allows overriding with more positional arguments (i.e. if reduction parameters are necessary).
        load_all_columns: bool = False,
        warn_missing_labels: bool = True,
        only_labels: bool = False,
        scan_obj: scanBase | None = None,
        **kwargs,
    ) -> scanBase:
        """
        Convert the parser object to a scan_base object.

        By default, only assigns 1D data to the scan object.
        Override this function if you want to perform reduction on 2D or higher order data.
        I.e. Use `match` and `case` statements to check `parser._parse_function` and perform
        appropriate reduction.

        Parameters
        ----------
        use_prior_params : bool, optional
            If True, then previously used parameters for reduction are used.
        load_all_columns : bool, optional
            If True, then columns unreferenced by `parser.COLUMN_ASSIGNMENTS` are also
            loaded into the y attribute of the scan object. By default False.
        warn_missing_labels : bool, optional
            If True, then triggers a warning for each label in `parser.COLUMN_ASSIGNMENTS`
            that is not found in the data. By default True.
        only_labels : bool, optional
            Only load labels and units from the parser object onto the scan object, by default False.
        scan_obj : scanBase, optional
            The scan object in which to add the parser data.
            Also utilized by the scanBase constructor.
        **kwargs
            Additional keyword arguments to pass to `parser.reduce` function.
            Check `reduce` docstring for possible arguments.

        Returns
        -------
        scan_base
            A scan object that allows easy interfacing with the NEXAFS data.

        Raises
        ------
        ValueError
            Raised if no data is loaded into the parser object.

        Notes
        -----
        Abstracts extra parameters used in `_to_scan` to convert reduced data and apply to an empty scan object.
        """

        # Use 1D data if provided
        data = self.data
        data = data[0] if isinstance(data, tuple) else data
        if data is None:
            raise ValueError(
                "No data loaded into the parser object, only header information. Use parser.load() to load data."
            )
        assert len(data.shape) == 2, "Data must be 2D for to_scan conversion."
        dshape = data.shape if data is not None else ()
        labels = self.labels
        labels = labels[0] if isinstance(labels, tuple) else labels
        units = self.units
        units = units[0] if isinstance(units, tuple) else units

        assert len(dshape) > 0

        # No point making a scan object if there's no data.
        reduction = self.reduce(use_prior_params=use_prior_params, **kwargs)
        if any(x is not None for x in reduction):
            rdata, rlabels, runits = reduction

            # Add reduced data / units / labels to the parser object.
            if rdata is not None:
                assert len(rdata) == len(data), (
                    "Reduced data length does not match original data length."
                )
                if len(rdata.shape) > 1 and rdata.shape[:-1] == dshape[:-1]:
                    # Concatenante the last index, i.e. the channel index.
                    data = np.concat((data, rdata), axis=-1)

                    # If labels are also defined and match add them.
                    if rlabels is not None:
                        if labels is None:
                            labels = [None] * dshape[-1]
                        labels.extend(rlabels)
                    else:
                        if labels is not None:
                            labels.extend([None] * rdata.shape[-1])
                    if runits is not None:
                        if units is None:
                            units = [None] * dshape[-1]
                        units.extend(runits)
                    else:
                        if units is not None:
                            units.extend([None] * rdata.shape[-1])

                else:
                    raise ValueError(
                        f"Excluding the final channel index, the reduced data shape ({rdata.shape}) \
                        does not match original data shape ({dshape}) for file {self.filename} loaded with {self._parser_fn}."
                    )

        # Create a new scan instance.
        if scan_obj is None:
            scan = scanBase(None, load_all_columns=load_all_columns)
        else:
            scan = scan_obj

        # Add the parser map back to the scan object.
        scan._parser = self
        scan._parser_class = self.__class__

        # Get assignments from parser object.
        assignments = self.COLUMN_ASSIGNMENTS  # Validated column assignments.

        ### Load data from parser
        # X data - locate column
        x_index = self.label_index(assignments["x"], reduced_labels=labels)
        # Y data - locate columns
        y_labels = assignments["y"]
        if isinstance(y_labels, list):
            y_indices = []
            for label in y_labels:
                try:
                    # label could be multiple labels, or a tuple of labels.
                    index = self.label_index(label, reduced_labels=labels)
                    y_indices.append(index)
                except (AttributeError, ValueError):
                    if warn_missing_labels:
                        warnings.warn(
                            f"Label {label} not found in parser object {self}."
                        )
        else:  # Singular index.
            y_indices = [self.label_index(y_labels, reduced_labels=labels)]
        # Y errors - locate columns
        if "y_errs" in assignments:
            y_errs_labels = assignments["y_errs"]
        else:
            y_errs_labels = None
        if isinstance(y_errs_labels, list):
            y_errs_indices = [
                (
                    self.label_index(label, reduced_labels=labels)
                    if label in y_errs_labels and label is not None
                    else None
                )
                for label in y_errs_labels
            ]
        elif isinstance(y_errs_labels, str):
            y_errs_indices = [self.label_index(y_errs_labels, reduced_labels=labels)]
        else:  # y_errs_labels is None:
            y_errs_indices = None
        # X errors - locate column
        if "x_errs" in assignments:
            x_errs_label = assignments["x_errs"]
        else:
            x_errs_label = None
        x_errs_index = (
            self.label_index(x_errs_label, reduced_labels=labels)
            if x_errs_label is not None
            else None
        )
        ### add conditional for load_all_columns.
        if load_all_columns:
            ## iterate over columns, and add to y_indices if not existing indices.
            # get the number of channels either from data shape
            dchannels: int
            if isinstance(data, tuple):
                dchannels = data[0].shape[1]
            else:
                dchannels = data.shape[1]
            # Then iterate for each of the extra data channels
            for i in range(dchannels):
                if (
                    i not in y_indices
                    and (y_errs_indices is None or i not in y_errs_indices)
                    and i != x_index
                    and (x_errs_index is None or i != x_errs_index)
                ):
                    y_indices.append(i)

        ### Generate data clones.
        # Data-points
        if not only_labels:
            scan._x = data[:, x_index].copy()
            scan._y = data[:, y_indices].copy()

            y_errs: npt.NDArray | None
            if y_errs_indices is not None:
                y_errs = np.zeros_like(scan._y)
                for i in y_errs_indices:
                    if i is None:
                        y_errs[:, i] = np.nan
                    else:
                        y_errs[:, i] = data[:, i].copy()
            else:
                y_errs = None
            scan._y_errs = y_errs

            scan._x_errs = (
                data[:, x_errs_index].copy() if x_errs_index is not None else None
            )
        # Labels and Units
        scan._x_label = labels[x_index] if labels is not None else None
        scan._x_unit = units[x_index] if units is not None else None
        y_labels = []
        for i in y_indices:
            if labels is not None and i < len(labels):
                y_labels.append(labels[i])
            else:
                y_labels.append(None)
        scan._y_labels = y_labels if len(y_labels) > 0 else None
        y_units = []
        for i in y_indices:
            if units is not None and i < len(units):
                y_units.append(units[i])
            else:
                y_units.append(None)
        scan._y_units = y_units if len(y_units) > 0 else None

        # Add attributes to scan for pre-defined channel names.
        self._add_channel_attr_to_scan(scan)
        return scan

    @classmethod
    def file_parser(
        cls: type[Self],
        file: TextIO,
        header_only: bool = False,
        **kwargs,
    ) -> tuple[
        Callable,
        npt.NDArray | tuple[npt.NDArray | None, ...] | None,
        list[str | None] | tuple[list[str | None] | None, ...] | None,
        list[str | None] | tuple[list[str | None] | None, ...] | None,
        dict[str, Any],
    ]:
        """
        A class method that tries to call `cls.parse_functions` methods to load file data.

        Successful parses are added to `cls.parse_recent_success`, with the extension/function as a key/value pair.

        Parameters
        ----------
        file : TextIO
            TextIO of the datafile (i.e. file=open('file.csv', 'r')).
        header_only : bool, optional
            If True, then only the header of the file is read and npt.NDArray is returned as None, by default False.
        **kwargs
            Additional keyword arguments that will be passed to the attempted parser method.
            Method will be skipped if the keyword arguments are not in the method signature.

        Returns
        -------
        parse_fn : Callable
            The reference to the successful class function.
        data : npt.NDArray | tuple[npt.NDArray, ...]
            Either a 2D array indexed by beam energy and channel number, or a tuple of increasing dimensional
            data (i.e. 2D, 3D... etc) where the first dimension index is the energy, the following indexes
            other independent variables, finally the channel number.
        labels : list[str | None] | tuple[list[str | None], ...] | None
            Labels corresponding to the data channel index for each dimension.
            Either None, a list of strings or None, or a tuple of the latter lists.
        units : list[str | None] | tuple[list[str | None], ...] | None
            Units corresponding to the data channel index for each dimension.
            Either None, a list of strings or None, or a tuple of the latter lists.
        params : dict[str, Any]
            Parameters of the datafile.
            While not enforced, recommended to be either a map to a direct value,
            or a tuple of (value, unit).

        Raises
        ------
        ImportError
            If no function collected in `cls.parse_functions` successfully loads the data in the datafile.
        """
        # Check if allowed extension
        if not file.name.endswith(tuple(cls.ALLOWED_EXTENSIONS)):
            raise ValueError(
                f"File {file.name} is not a valid file type for {cls.__name__}."
            )

        extension = file.name.split(".")[-1]

        if len(cls.parse_functions) > 0:
            # Check if any parse functions match the file type.
            if (
                extension in cls.parse_recent_success
                and cls.parse_functions[0] != cls.parse_recent_success[extension]
            ):
                # Reorder parse functions to put the most recent successful parse function first.
                i = cls.parse_functions.index(cls.parse_recent_success[extension])
                cls.parse_functions.insert(0, cls.parse_functions.pop(i))

            # Collect the errors of each parse function, to display if all fail.
            parse_errs: dict[Callable, Exception] = {}

            # Attempt to load the file using the parse functions.
            for parse_fn in cls.parse_functions:
                # Attempt to use parse functions that contain a string that matches the extension
                if extension in parse_fn.__name__:
                    try:
                        # Get the argument names of the parse function.
                        arg_names = list(
                            parse_fn.__code__.co_varnames[
                                : parse_fn.__code__.co_argcount
                            ]
                        )
                        def_len = (
                            len(parse_fn.__defaults__)
                            if parse_fn.__defaults__ is not None
                            else 0
                        )
                        default_args = arg_names.copy()[-def_len:]

                        # Remove the file_parser arguments from the list.
                        for name in ["cls", "file"]:  # Non-optionals
                            if name in arg_names:
                                arg_names.remove(name)

                        # Check all keyword args are in the arg_names, otherwise skip the method.
                        if not all([kw in kwargs for kw in arg_names]):
                            missing_args = [kw for kw in arg_names if kw not in kwargs]

                            # Check if the argument has a default value.
                            for arg in missing_args:
                                if arg not in default_args:
                                    # Skip method if argument is NOT covered by a default value.
                                    warnings.warn(
                                        f"Skipping {parse_fn.__name__} due to missing argument {arg}."
                                    )
                                    continue

                        # Copy kwargs and only use the arguments that are in the method signature.
                        fn_kwargs = {kw: kwargs[kw] for kw in arg_names if kw in kwargs}
                        if type(parse_fn) is types.FunctionType:  # staticmethod
                            obj = (
                                parse_fn(file, header_only, **fn_kwargs)
                                if "header_only" in arg_names
                                else parse_fn(file)
                            )
                            # Close the file upon successful load
                            file.close()
                            # Add the successful parse function to the recent success dictionary.
                            cls.parse_recent_success[extension] = parse_fn
                            # Put the successful parse function first in the list.
                            i = cls.parse_functions.index(
                                cls.parse_recent_success[extension]
                            )
                            cls.parse_functions.insert(0, cls.parse_functions.pop(i))
                            return (parse_fn, *obj)
                        else:  # classmethod
                            # type(parse_fn == types.MethodType)
                            # cls is the first argument of the method, already incorporated into the function call.
                            obj = (
                                parse_fn(file, header_only=header_only, **fn_kwargs)
                                if "header_only" in arg_names
                                else parse_fn(file)
                            )

                            # Close the file upon successful load
                            file.close()
                            cls.parse_recent_success[extension] = parse_fn
                            # Put the successful parse function first in the list.
                            i = cls.parse_functions.index(
                                cls.parse_recent_success[extension]
                            )
                            cls.parse_functions.insert(0, cls.parse_functions.pop(i))
                            return (parse_fn, *obj)
                    except Exception as e:
                        parse_errs[parse_fn] = e
                        # Method failed, continue to next method.
                        # warnings.warn(
                        #     f"Attempted method '{parse_fn.__name__}' failed to load '{os.path.basename(file.name)}' from '{cls.__name__}' with {repr(e)}.",
                        #     ImportWarning,
                        # )
                        # Uncomment this to see your import errors.
                        # print("Traceback: ", file.name)
                        # traceback.print_exception(e)

            msg = f"------------------------- All {cls.__name__} loaders failed -------------------------"
            for pfn, err in parse_errs.items():
                msg += f"\nMethod '{pfn.__name__}' failed with {err!r}."
                with io.StringIO() as buf:
                    traceback.print_exception(
                        type(err), err, err.__traceback__, file=buf
                    )
                    msg += "\nTraceback:\n" + buf.getvalue()
            msg += "\n-------------------------------------------------------------------------------------"
            warnings.warn(msg)

            # If no parse functions successfully import the file type,
            file.close()
            raise ImportError(
                f"No parser method in {cls.__name__} succeeded on {file.name}."
            )

        # If no parse functions match the file type, raise an error.
        file.close()
        raise ImportError(f"No parser method in {cls.__name__} found for {file.name}.")

    def load(
        self,
        file: str | IO | None = None,
        header_only: bool = False,
        **kwargs,
    ) -> None:
        """
        Load all data from the specified file, and attach it to the object.

        Additionally rewrites filepath attribute if a new file is loaded.
        Will also save additional "created" and "modified" entries in param list
        using `os` library if not already generated by parser methods.

        Parameters
        ----------
        file : str | IO | None, optional
            File information can be passed as a string or an IO object (i.e. TextIO or BinaryIO implementations).
            If None, then the object filepath attribute is used to load the data.
            If filepath is also None, then a ValueError is raised.
        header_only : bool, optional
            If True, then only the header of the file is read, by default False
            Consequently loads labels, units and params. Pre-existing object data persists.
        **kwargs
            Additional keyword arguments that will be passed to the `file_parser` method.

        Raises
        ------
        ValueError
            Raised if no file is provided and the object filepath attribute is None.
        """
        # Use the instantiation kwargs, and update with any additional kwargs for the load call.
        # Does not override the initialisation kwargs.
        parsing_kwargs = self._parsing_kwargs.copy()
        parsing_kwargs.update(kwargs)

        data: npt.NDArray | tuple[npt.NDArray | None, ...] | None
        labels: list[str | None] | tuple[list[str | None] | None, ...] | None
        units: list[str | None] | tuple[list[str | None] | None, ...] | None
        params: dict[str, Any]

        # Load object filepath
        load_filepath = self._filepath  # Might be None
        try:
            # Check if file parameter is provided:
            if isinstance(
                file, (IO, typing.IO, io.FileIO, tempfile._TemporaryFileWrapper)
            ):
                if isinstance(file.name, str) and os.path.exists(file.name):
                    load_filepath = file.name
                    if file.closed:
                        file = open(load_filepath, "r")  # Re-open file if closed.
                # File already loaded
                fn, data, labels, units, params = self.file_parser(
                    file, header_only=header_only, **parsing_kwargs
                )
                file.close()  # Close file after reading
                # If a file is provided override filepath.
            elif isinstance(file, str) and os.path.isfile(file):
                # Update filepath
                load_filepath = file
                with open(load_filepath, "r") as load_file:
                    fn, data, labels, units, params = self.file_parser(
                        load_file, header_only=header_only, **parsing_kwargs
                    )  # makes sure to close the file.
            elif file is None and load_filepath is not None:
                with open(load_filepath, "r") as load_file:
                    fn, data, labels, units, params = self.file_parser(
                        load_file, header_only=header_only, **parsing_kwargs
                    )  # makes sure to close the file.
            elif load_filepath is None:
                raise ValueError("No file/filepath provided to load data.")
            else:
                raise ValueError(
                    f"File parameter of type {type(file)} not supported for loading."
                )

            # Prepare d to check data lengths
            if data is not None or self.data is not None:
                # Use existing data to check consistency if only (re-)loading header.
                if data is None and self.data is not None:
                    data = self.data
                if not isinstance(data, tuple):
                    datum = (data,)
                else:
                    datum = data
            else:
                datum = None
            # Prepare l to check label lengths, u to check unit lengths.
            if labels is not None:
                if not isinstance(labels, tuple):
                    lab = (labels,)
                else:
                    lab = labels
            else:
                lab = None
            if units is not None:
                if not isinstance(units, tuple):
                    unit = (units,)
                else:
                    unit = units
            else:
                unit = None

            # Check each dimension
            if datum is not None and (lab is not None or unit is not None):
                for i, di in enumerate(datum):
                    if di is None:
                        # No data, skip.
                        continue
                    # Get corresponding labels and units
                    if lab is not None:
                        li = lab[i]
                    else:
                        li = None
                    if unit is not None:
                        ui = unit[i]
                    else:
                        ui = None

                    # Pull column length of data to compare to units and labels.
                    col_len = di.shape[-1]  # last index is the number of channels
                    if li is not None and len(li) != col_len:
                        raise ValueError(
                            f"Labels length {len(li)} does not match data channel length {col_len} for the {len(di.shape)}D data."
                        )
                    if ui is not None and len(ui) != col_len:
                        raise ValueError(
                            f"Units length {len(ui)} does not match data channel length {col_len} for the {len(di.shape)}D data."
                        )
            elif lab is not None and unit is not None:
                # Check the labels and unit lengths are the same.
                for i, li in enumerate(lab):
                    ui = unit[i]
                    if ui is not None and li is not None:
                        if len(ui) != len(li):
                            raise ValueError(
                                f"Labels length ({len(li)}) does not match the units length ({len(ui)}) for the index #{i}."
                            )

        except AssertionError:
            # Assume file cannot be loaded using the parser methods. Raise loading error.
            raise ImportError(
                f"'{load_filepath}' encountered an assertion error while loading using the parse methods of {self.__class__}."
            )

        # Add modified and created parameter entries from system OS, if not existing in params already.
        # If aleady provided, then convert to datetime object using `convert_to_datetime` method below.
        if "created" not in params or params["created"] is None:
            if load_filepath is not None and os.path.exists(load_filepath):
                params["created"] = datetime.datetime.fromtimestamp(
                    os.path.getctime(load_filepath)
                )
        elif not isinstance(params["created"], datetime.datetime):
            try:
                params["created"] = self.convert_to_datetime(params["created"])
            except ValueError as e:
                warnings.warn(
                    f"Could not convert 'created' parameter `{params['created']}` to datetime object; {e}"
                )
        if "modified" not in params:
            if load_filepath is not None and os.path.exists(load_filepath):
                params["modified"] = datetime.datetime.fromtimestamp(
                    os.path.getmtime(load_filepath)
                )
        elif not isinstance(params["modified"], datetime.datetime):
            try:
                params["modified"] = self.convert_to_datetime(params["modified"])
            except ValueError as e:
                warnings.warn(
                    f"Could not convert 'modified' parameter `{params['modified']}` to datetime object; {e}"
                )
        self._filepath = load_filepath
        self._parser_fn = fn  # store the used parsing function.

        # Add a size entry for the file
        if "filesize" not in params or params["filesize"] is None:
            params["filesize"] = self.filesize

        if isinstance(data, tuple):
            # Check that each tuple index matches the consecutive rank
            for i, datum in enumerate(data):
                if datum is None:
                    continue
                assert len(datum.shape) == i + 2, (
                    f"Data at index {i} has shape {datum.shape}, expected rank {i + 2}."
                )
            # Reduce single array data to a single array (not a tuple).
            if len(data) == 1:
                data = data[0]

        # Assign data, labels, units, and params to object.
        self.data, self._labels, self.units = data, labels, units
        self.params.update(params)

        # Update filepath after data load if successful!
        if not header_only:
            self._parsed_data = True

        # Add a size entry for pyNexafs
        if "memory_size" not in params or params["memory_size"] is None:
            params["memory_size"] = self.memorysize

    @staticmethod
    def convert_to_datetime(time_input: Any) -> datetime.datetime:
        """
        Convert a generic time input to a datetime object.

        Attempts to convert the input to a datetime object using various methods.

        Parameters
        ----------
        time_input : Any
            Time input to convert to a datetime object.

        Returns
        -------
        datetime.datetime
            Datetime object of the time input.

        Raises
        ------
        ValueError
            Raised if the time input cannot be converted to a datetime object.
        """
        if isinstance(time_input, datetime.datetime):
            return time_input
        elif isinstance(time_input, float):
            return datetime.datetime.fromtimestamp(time_input)
        elif isinstance(time_input, int):
            return datetime.datetime.fromordinal(time_input)
        elif isinstance(time_input, str):
            # First try ISO 8601 time format.
            try:
                return datetime.datetime.fromisoformat(time_input)
            except ValueError:
                pass
            # Try to convert to float or int
            try:
                return parserBase.convert_to_datetime(float(time_input))
            except ValueError:
                pass
            try:
                return parserBase.convert_to_datetime(int(time_input))
            except ValueError:
                pass
        # If none of these work, raise an error.
        raise ValueError(
            f"Time input {time_input} could not be converted to a datetime object."
        )

    @property
    def ctime(self) -> datetime.datetime:
        """
        The creation time of the file as a datetime object.

        Data is pulled from the 'created' parameter in the params dictionary.
        This is generated by the 'load' method using the 'os' library,
        if not already created in the params.

        Returns
        -------
        datetime.datetime
            Creation time of the file.
        """
        return self.convert_to_datetime(self.params["created"])

    @property
    def mtime(self) -> datetime.datetime:
        """
        The modification time of the file as a datetime object.

        Data is pulled from the 'modified' parameter in the params dictionary.
        This is generated by the 'load' method using the 'os' library,
        if not already created in the params.

        Returns
        -------
        datetime.datetime
            Modification time of the file.
        """
        return self.convert_to_datetime(self.params["modified"])

    def copy(self, clone: Self | None = None) -> Self:
        """
        Generate a copy of the parser object.

        Parameters
        ----------
        clone : Self | None, optional
            If a clone instance is provide, then properties are copied to the clone.
            If None, then a new copy is made from the current object.

        Returns
        -------
        Self
            A copy of the parser object.
        """
        # Use the provided clone, otherwise create newobj.
        newobj = (
            self.__class__(
                filepath=None,
                header_only=False,
                relabel=None,
            )
            if not clone
            else clone
        )
        ## Perform deep copy of data, labels, and params.
        newobj._filepath = self._filepath  # str copy
        # Data
        data = self.data
        if isinstance(data, tuple):
            newobj.data = tuple(
                d.copy() if d is not None else None for d in data
            )  # numpy copy
        else:
            newobj.data = data.copy() if data is not None else None  # numpy copy
        # Labels
        labels = self._labels
        if isinstance(labels, tuple):
            newobj._labels = tuple(
                lab.copy() if lab is not None else None for lab in labels
            )  # str list copy
        else:
            newobj._labels = (
                labels.copy() if labels is not None else None
            )  # str list copy
        # Units
        units = self.units
        if isinstance(units, tuple):
            newobj.units = tuple(
                unit.copy() if unit is not None else None for unit in units
            )  # str list copy
        else:
            newobj.units = units.copy() if units is not None else None  # str list copy
        # Params
        for key in self.params:  # dict key str - value Any copy
            value = self.params[key]
            if isinstance(value, (int, str, float, bool, datetime.datetime, tuple)):
                newobj.params[key] = value
            elif hasattr(value, "copy"):
                newobj.params[key] = value.copy()
            else:
                # Shallow
                newobj.params[key] = value
                warnings.warn(
                    f"Shallow copy of parameter {key} with type {type(value)} in {type(self)}."
                )
        newobj.relabel = self.relabel  # bool copy
        return newobj

    @property
    def filesize(self) -> int:
        """
        The size of the file in bytes.

        Returns
        -------
        int
            Size of the file in bytes.
        """
        if self._filepath is None:
            raise ValueError("No file loaded.")
        elif os.path.isfile(self._filepath):
            return os.path.getsize(self._filepath)
        else:
            raise FileNotFoundError(f"File {self._filepath} not found.")

    @property
    def memorysize(self) -> int:
        """
        The size of the object (and it's attributes) in memory bytes.

        Returns
        -------
        int
            Size of the object in memory in bytes.
        """
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.data)
            + sys.getsizeof(self.labels)
            + sys.getsizeof(self.params)
            + sum([sys.getsizeof(v) for v in self.params.values()])
            + sys.getsizeof(self.units)
        )

    def to_DataFrame(self) -> "pd.DataFrame":
        """
        Generate a pandas DataFrame from the data.

        Returns
        -------
        DataFrame
            A pandas DataFrame of the data and labels.
        """
        if pd is None:
            raise ImportError("Pandas is required for this method.")
        return pd.DataFrame(
            self.data, columns=self._labels if len(self._labels) > 0 else None
        )

    @classmethod
    def _label_index(
        cls,
        labels: list[str | None] | tuple[list[str | None] | None, ...] | None,
        search: str | tuple[str, ...] | list[str | tuple[str, ...]],
        search_relabels: bool = True,
    ) -> int:
        """
        A class method to find the index of the label in the labels list.

        Searches for raw labels first, then checks and searches for any `RELABELS` match
        (if `search_relabels`, but default `True`).

        Only returns indexes for 1D data channels. If the label is found in a higher dimensionality list,
        then a ValueError is raised, as this method is only intended for searching for 1D channel labels.

        Parameters
        ----------
        labels : list[str | None] | tuple[list[str | None] | None, ...] | None,
            Labels to search for in the labels list. Either None, a list of strings or None, or a tuple of the latter lists.
        search : str | tuple[str, ...] | list[str | tuple[str, ...]]
            String labels to search for in the labels list. Also allows tuples of synonymous strings, and lists of the former.
            Returns the index of the first search query that is successful.
        search_relabels : bool, optional
            To additionally search check the RELABELS dictionary for the label and subsequently search. By default True.

        Returns
        -------
        int
            The index of the label in the labels list.

        Raises
        ------
        ValueError
            If `search` is not a string, tuple of strings, or list of the previous two types.
        ValueError
            If the label is not found in the labels list
        ValueError
            If the label is found in higher dimensionality lists rather than the 1D list.
        """
        if isinstance(search, list) and isinstance(search[0], (str, tuple)):
            queries = []
            for query in search:
                if isinstance(query, tuple):
                    queries.extend(list(query))
                else:
                    queries.append(query)
        elif isinstance(search, tuple):
            queries = list(search)
        elif isinstance(search, str):
            queries = [search]
        else:
            raise ValueError(
                f"Search parameter {search} is not a string, tuple of strings or list of the previous."
            )
        # Collect 1D labels first.
        if isinstance(labels, tuple):
            labels_1D = labels[0]
            if labels_1D is None:
                raise AttributeError(
                    f"The list of labels for the 1D channels is None. Cannot search for {search}."
                )
        else:
            labels_1D = labels

        for query in queries:
            # Check default queries
            try:
                return labels_1D.index(query)  # throws value error if not found.
            except ValueError:
                # Also check relabel if search_relabel is True.
                if search_relabels:
                    # Search keys
                    if query in cls.RELABELS:
                        try:
                            return labels_1D.index(
                                cls.RELABELS[query]
                            )  # throws value error if not found
                        except ValueError:
                            pass

                        # Search values instead to reverse for key
                        k = None
                        v = None
                        for k, v in cls.RELABELS.items():
                            if v == cls.RELABELS[query]:
                                break

                        if isinstance(k, str):
                            try:
                                return labels_1D.index(
                                    k
                                )  # throws value error if not found
                            except ValueError:
                                pass
                        if isinstance(k, tuple):
                            for kk in k:
                                try:
                                    return labels_1D.index(
                                        kk
                                    )  # throws value error if not found
                                except ValueError:
                                    pass

        # Check higher order dimensions in case the value is not in the 1D list.
        if isinstance(labels, tuple) and len(labels) > 1:
            for i, labels_ND in enumerate(labels[1:]):
                # Get the channel labels
                if labels_ND is None:
                    continue
                elif isinstance(labels_ND, str):
                    labels_ND = [labels_ND]
                # Check if the query exists in the higher dimension labels.
                for query in queries:
                    # Check default queries
                    try:
                        if query in labels_ND:
                            raise ValueError(
                                f"The label `{query}` was found in the index #{i + 1} dimensional data, \
                                but only 1D channels (0th index) are considered. Perhaps use `reduced_labels` \
                                parameter after reduction?"
                            )
                    except ValueError:
                        # Also check relabel if search_relabel is True.
                        if search_relabels:
                            # Search keys
                            if query in cls.RELABELS:
                                raise ValueError(
                                    f"The label `{query}` matched to `{cls.RELABELS[query]}` was found \
                                    in the index #{i + 1} dimensional data, but only 1D channels (0th index) are considered. \
                                    Perhaps use `reduce_labels` parameter after reduction?"
                                )

                            # Search values instead to reverse for key
                            relabel_vals = list(cls.RELABELS.values())
                            if query in relabel_vals:
                                relabel_keys = list(cls.RELABELS.keys())
                                # Check for multiple matching values (opposed to unique keys).
                                warn = True if relabel_vals.count(query) > 1 else False
                                # Find value in RELABELS
                                i = relabel_vals.index(query)
                                if warn:
                                    warnings.warn(
                                        f"Multiple labels matched `{query}` in {cls.__name__}.RELABELS. Using key '{relabel_keys[i]}' as the label."
                                    )
                                # If labels found, return index of key.
                                key = relabel_keys[i]
                                if key in labels_ND:
                                    raise ValueError(
                                        f"The label `{query}` matched to `{key}` was found \
                                    in the index #{i + 1} dimensional data, but only 1D channels (0th index) are considered. \
                                    Perhaps use `reduce_labels` parameter after reduction?"
                                    )
        # Finally
        raise ValueError(f"Label(s) '{search}' not found in labels {labels_1D}.")

    def label_index(
        self,
        search: str | tuple[str, ...] | list[str | tuple[str, ...]],
        search_relabels: bool = True,
        reduced_labels: list[str | None] | None = None,
    ) -> int:
        """
        Find the index of the label in the labels list.

        Searches for raw labels first, then checks and searches for any RELABELS match (if `search_relabels`, but default `True`).

        Parameters
        ----------
        search : str | tuple[str, ...] | list[str | tuple[str, ...]]
            String labels to search for in the labels list. Also allows tuples of synonymous strings, and lists of the former.
            Returns the index of the first search query that is successful.
        search_relabels : bool, optional
            To additionally search check the RELABELS dictionary for the label and subsequently search. By default True.
        reduced_labels : list[str | None] | None = None
            Replaces use of default `parser.labels`, particularly useful when using reduced data.

        Returns
        -------
        int
            Index of the label in the labels list.

        Raises
        ------
        ValueError
            If the (1D channels) labels list is None.
        """
        # Check if labels is a tuple (i.e. multiple channel dimensions)
        labels = self._labels if reduced_labels is None else reduced_labels
        if labels is None:
            raise AttributeError(
                f"The labels attribute of {self} is `None`. Cannot search for {search}."
            )
        return self._label_index(labels, search, search_relabels)

    def __getitem__(self, key: str | dtype) -> npt.NDArray:
        """
        Get the data for the specified key.

        Parameters
        ----------
        key : str | dtype
            The key to get the data for. Can be a string label or a dtype.

        Returns
        -------
        npt.NDArray
            The data for the specified key.

        Raises
        ------
        KeyError
            If the key is not found in the labels or channel map.
        ValueError
            If the data or labels are not loaded.
        """
        data = self.data
        labels = self._labels
        if data is None or labels is None:
            raise ValueError("Data or labels are not loaded.")
        if isinstance(key, (str, dtype)):
            try:
                index = self.label_index(key)
                if isinstance(data, tuple):
                    for d in data:
                        if d is not None and len(d.shape) == 2:
                            data = d
                            break
                if isinstance(data, tuple):
                    raise ValueError("No set of 1D data channels found.")
                return data[:, index]
            except ValueError as e:
                # Check the value error text matches the following:
                if str(e).startswith("Label(s) ") and " not found in labels " in str(e):
                    raise KeyError(f"Key {key} not found in labels.")
                else:
                    raise e
        else:
            raise KeyError(f"Key {key} must be a string label or a dtype.")

    @classmethod
    def _get_dtype_relabel_channels(cls) -> list[dtype]:
        """
        Collect a list dtype enumerate that are available in the RELABELS dictionary.

        Returns
        -------
        list[dtype]
            A list of dtype that are present in the RELABELS dictionary, either as keys or
            values.
        """
        if (
            hasattr(cls, "_dtype_channels")
            and cls._dtype_channels is not None
            and len(cls._dtype_channels) > 0
        ):
            return cls._dtype_channels
        else:
            # Collect the dtype channels present.
            dtypes: list[dtype] = []
            for key, val in cls.RELABELS.items():
                if isinstance(val, dtype):
                    dtypes.append(val)
                elif isinstance(key, tuple):
                    for k in key:
                        if isinstance(k, dtype):
                            dtypes.append(k)
                elif isinstance(key, dtype):
                    dtypes.append(key)
            return dtypes

    @classmethod
    def _add_channel_attr_to_scan(cls, scan: scanBase) -> None:
        """
        Add attribute accessors for the channels defined in the RELABELS dictionary.

        For example, if a channel has a RELABELS key or value with dtype.PHD, then an attribute `scan.PHD`
        is added to the scan object that returns the data for that channel.

        Parameters
        ----------
        scan : scanBase
            The scan object to add the channel attributes to.
        """
        channels = cls._get_dtype_relabel_channels()
        available_channels = []
        for channel in channels:
            # Check if the channel is indexable in the labels:
            try:
                _ = cls._label_index([scan.x_label] + scan.y_labels, channel)

                @property
                def get_dtype_data(self):
                    if (channel == self.x_label) or (
                        (channel in cls.RELABELS and self.x_label in cls.RELABELS)
                        and (cls.RELABELS[channel] == cls.RELABELS[self.x_label])
                    ):
                        return self.x
                    else:
                        idx = cls._label_index(self.y_labels, channel)
                        return self.y[:, idx]

                setattr(scan, channel.name, get_dtype_data)

                available_channels.append(channel)
            except ValueError:
                pass

        # Add the available channels as an attribute of the scan class.
        scan._available_channels = available_channels

    @property
    def summary_params(self) -> dict[str, Any]:
        """
        A dictionary of important parameters of the data file.

        Immediately searches
        Utilises the `parser.RELEBELS` dictionary to identify important parameters.

        Returns
        -------
        dict
            Dictionary of important parameters.
        """
        # TODO: Implement tests
        summary_params = {}
        for key in self.SUMMARY_PARAM_RAW_NAMES:
            if isinstance(key, tuple):
                keys = key
            elif isinstance(key, str):
                keys = [key]
            else:
                raise ValueError(
                    f"Summary parameter key {key} is not a string or list of strings."
                )
            key: str | tuple[str]
            # Take first name if multiple names are provided.
            summary_key = keys[0]
            # Use the key string(s) to search for the parameter in the params dictionary:
            found = False
            for key in keys:
                if key in self.params:
                    summary_params[summary_key] = self.params[key]
                    found = True
                    break  # Only use first value found.

            # If not found see if the key can be found in reverse:
            if not found and key in self.RELABELS_REVERSE:
                # Check if the key is a string and exists in the params dictionary.
                rev_key: str | tuple[str, ...] = self.RELABELS_REVERSE[key]
                if isinstance(rev_key, str) and rev_key in self.params:
                    summary_params[summary_key] = self.params[rev_key]
                elif isinstance(rev_key, tuple):
                    for k in rev_key:
                        if k in self.params:
                            summary_params[summary_key] = self.params[k]
                            break
        return summary_params

    @property
    def summary_param_names(self) -> list[str]:
        """
        A list of important parameter names of the data file.

        Sources names from cls.SUMMARY_PARAM_RAW_NAMES list.
        Returns summary names that are found in params.
        If names are defined in cls.RELABELS, then the relabelled names are returned.

        Returns
        -------
        list[str]
            List of important parameter names.
        """
        # TODO: Implement Tests
        names: list[str] = []
        for keystr in self.__class__.SUMMARY_PARAM_RAW_NAMES:
            keys = keystr if isinstance(keystr, tuple) else [keystr]
            key1 = keys[0]  # Take first name if multiple names are provided.
            for key in keys:
                # Collect summary names that exist in the parser params.
                if (
                    key in self.params
                    or (key in self.RELABELS and self.RELABELS[key] in self.params)
                    or (
                        key in self.RELABELS_REVERSE
                        and self.RELABELS_REVERSE[key] in self.params
                    )
                ):
                    names.append(key1)
                    break
        return names

    @property
    def summary_param_values(self) -> list[Any]:
        """
        A list of important parameter values of the data file.

        Returns
        -------
        list
            List of important parameter values.
        """
        # TODO: Implement Tests
        return list(self.summary_params.values())  # TODO: modify to dict_values?

    @property
    def summary_param_names_with_units(self) -> list[str]:
        """
        A list of important parameter names with units.

        Requires a loaded dataset to return the units of the parameters.

        Returns
        -------
        list
            List of important parameter names with units.

        Raises
        ------
        NotImplementedError
            Raised if the method is not implemented for the parser object.
        """
        raise NotImplementedError(
            f"Method 'summary_param_names_with_units' not implemented for {type(self)}."
        )
