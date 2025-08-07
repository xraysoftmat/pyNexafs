""" "Tests the base parser classes and their methods"""

import pytest

# Test the creation of a new parser class.
from pyNexafs.parsers._base import parser_meta, parser_base

##############################################################################
################### Test the inner classes of parser_meta ####################
##############################################################################


###
### The RELABELS dict
###
class TestRelabelsDict:
    """Tests the synonymous tuple functionality of the relabels dictionary."""

    class DummyParserRelabel:
        RELABELS = {"a": "b", "c": "d", ("e", "f"): "g"}

    @pytest.mark.parametrize(
        "key, value",
        [("a", "b"), ("c", "d"), ("e", "g"), ("f", "g"), (("e", "f"), "g")],
    )
    def test_get(self, key, value):
        """Tests the relabels dictionary can be set and returns the correct value,
        including the use of tuples."""

        # Create the relabel dictionary.
        relabel_dict = parser_meta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test its access
        assert relabel_dict[key] == value

    @pytest.mark.parametrize(
        "key, value",
        [("a", "1"), ("c", "2"), ("e", "3"), ("f", "4"), (("e", "f"), "5")],
    )
    def test_set(self, key, value):
        # Create the relabel dictionary.
        relabel_dict = parser_meta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test setting
        relabel_dict[key] = value
        # Check it was set
        assert relabel_dict[key] == value

    @pytest.mark.parametrize(
        "key, contained",
        [
            ("a", True),
            ("b", True),
            ("c", True),
            ("d", True),
            ("e", True),
            ("f", True),
            (("e", "f"), True),
            ("g", True),
            ("h", False),
            (("e", "g"), False),
        ],
    )
    def test_contains(self, key, contained):
        """Tests the relabels dictionary can be set and returns the correct value,
        including the use of tuples."""

        # Create the relabel dictionary.
        relabel_dict = parser_meta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test its access
        assert (key in relabel_dict) == contained

    def test_tuples(self):
        """Tests the synonymous tuple functionality of the relabels dictionary."""
        relabel_dict = parser_meta.relabels_dict(self.DummyParserRelabel.RELABELS)
        assert relabel_dict[("e", "f")] == "g"  # type: ignore ##pylance doesnt like this.
        assert relabel_dict["e"] == "g"
        # Test propogation
        relabel_dict["f"] = "h"  # Should also update `e` and `("e", "f")`
        assert relabel_dict["e"] == "h"
        assert relabel_dict[("e", "f")] == "h"  # type: ignore


###
### The Summay Param List
###
class TestSummaryParamList:
    """Tests the summary parameters list functionality."""

    class ParserNoRelabels:
        RELABELS = {}
        RELABELS_REVERSE = {}
        relabel = False

    class ParserRelabels:
        RELABELS = {"a": "b", "c": "d", ("e", "f"): "g"}
        RELABELS_REVERSE = {"b": "a", "d": "c", "g": ("e", "f")}
        relabel = True

    @pytest.mark.parametrize(
        "elements, err, overlapping_param",
        [
            (["a", "c"], None, None),
            (["a", "c", "c"], ValueError, "c"),
            (["a", "b", "c"], None, None),
            (["1", "2", "3"], None, None),
            (["a", ("b", "c"), "d"], None, None),
            (["a", ("b", "c"), "b"], ValueError, "b"),
        ],
    )
    def test_init_valid(self, elements, err, overlapping_param):
        """Tests the summary parameters list can be created and initialised, particularly the `__check_valid` method."""
        # Create the summary parameters list.
        if err:
            with pytest.raises(err) as e:
                summary_params = parser_meta.summary_param_list(
                    elements, parent=self.ParserNoRelabels
                )  # type: ignore
            if overlapping_param:
                assert f"Summary Parameter `{overlapping_param}`" in str(e.value)
        else:
            summary_params = parser_meta.summary_param_list(
                elements, parent=self.ParserNoRelabels
            )  # type: ignore

    @pytest.mark.parametrize(
        "elements, err, overlapping_param",
        [
            (["a", "c"], None, None),
            (["a", "c", "c"], ValueError, "c"),
            (["a", "b", "c"], ValueError, "b"),
            (["1", "2", "3"], None, None),
            (["a", ("i", "c"), "d"], ValueError, "d"),
            (["a", ("b", "c"), "b"], ValueError, "('b', 'c')"),
            (["0", ("b", "c"), "b"], ValueError, "b"),
            (["a", "d", ("b", "c")], ValueError, "('b', 'c')"),
        ],
    )
    def test_init_valid_relabelled(self, elements, err, overlapping_param):
        """Tests the summary parameters list can be created and initialised, particularly the `__check_valid` method when relabelling."""
        # Create the summary parameters list.
        if err:
            with pytest.raises(err) as e:
                summary_params = parser_meta.summary_param_list(
                    elements, parent=self.ParserRelabels
                )  # type: ignore
            if overlapping_param:
                assert f"Summary Parameter `{overlapping_param}`" in str(e.value)
        else:
            summary_params = parser_meta.summary_param_list(
                elements, parent=self.ParserRelabels
            )  # type: ignore

    @pytest.mark.parametrize(
        "elements, query, contains",
        [
            (["a", "b", "c"], "a", True),
            (["1", "2", "3"], "1", True),
            (["a", ("b", "c"), "d"], "b", True),
            (["a", ("b", "c"), "d"], ("b", "c"), True),
            (["a", "b", "c"], "d", False),
        ],
    )
    def test_contains(self, elements, query, contains):
        """Tests the summary parameters list can be set and returns the correct value."""

        # Create the summary parameters list.
        summary_params = parser_meta.summary_param_list(
            elements, parent=self.ParserNoRelabels
        )  # type: ignore
        # Test its access
        assert (query in summary_params) == contains

    @pytest.mark.parametrize(
        "elements, query, relabelled_contains",
        [
            (["a", "c"], "a", True),
            (["a", "c"], "b", True),
            (["a", "c"], "e", False),
            (["1", "2", "3"], "1", True),
            ([("b", "c"), "e"], "b", True),
            ([("b", "c"), "e"], "d", True),
            (["f", ("b", "c"), "o"], ("b", "c"), True),
            (["f", ("b", "c"), "o"], ("b", "d"), False),
            (["f", ("b", "c"), "o"], ("b", "e"), False),
        ],
    )
    def test_contains_relabelled(self, elements, query, relabelled_contains):
        """Tests the summary parameters list can be set and returns the correct value."""
        # Create the summary parameters list.
        summary_params = parser_meta.summary_param_list(
            elements, parent=self.ParserRelabels
        )  # type: ignore
        # Test its access
        assert (query in summary_params) == relabelled_contains

    @pytest.mark.parametrize(
        "elements, query, index_or_exception",
        [
            (["a", "b", "c"], "a", 0),
            (["1", "2", "3"], "1", 0),
            (["a", ("b", "c"), "d"], "b", 1),
            (["a", ("b", "c"), "d"], "d", 2),
            (["a", ("b", "c"), "d"], ("b", "c"), 1),
            (["a", "b", "c"], "e", ValueError),
        ],
    )
    def test_index_tuples(self, elements, query, index_or_exception):
        """Tests the summary parameters list can be indexed, and tuples behave as individual elements also."""
        # Create the summary parameters list.
        summary_params = parser_meta.summary_param_list(
            elements, parent=self.ParserNoRelabels
        )  # type: ignore
        # Test its access
        if isinstance(index_or_exception, type) and issubclass(
            index_or_exception, Exception
        ):
            with pytest.raises(
                index_or_exception,
                match=f"{query} is not in the list, or within a tuple element.",
            ) as e:
                summary_params.index(query)
        elif isinstance(index_or_exception, int):
            assert summary_params.index(query) == index_or_exception
        else:
            raise ValueError("Invalid test case.")

    @pytest.mark.parametrize(
        "elements, query, index_or_exception",
        [
            (["a", "c"], "a", 0),
            (["a", "c"], "b", 0),
            (["1", "2", "3"], "3", 2),
            (["0", ("b", "c"), "e"], "b", 1),
            (["0", ("b", "c"), "e"], "d", 1),
            (["0", ("b", "c"), "e"], ("b", "c"), 1),
            (["0", ("b", "c"), "e"], ("b", "d"), 1),
            (["a", "0", "c"], "e", ValueError),
        ],
    )
    def test_index_relabelled(self, elements, query, index_or_exception):
        """Tests the summary parameters list can be indexed, tuples behave as individual elements and relabels apply."""
        # Create the summary parameters list.
        summary_params = parser_meta.summary_param_list(
            elements, parent=self.ParserRelabels
        )  # type: ignore
        # Test its access
        if isinstance(index_or_exception, type) and issubclass(
            index_or_exception, Exception
        ):
            with pytest.raises(
                index_or_exception,
                match=f"{query} is not in the list, or within a tuple element.",
            ) as e:
                summary_params.index(query)


##############################################################################
############################ Test parser_meta ################################
##############################################################################


class TestParserMeta:
    """Tests the metaclass functionality on creating new Parser classes."""

    def test_missing_ALLOWED_EXTENSIONS(self):
        """Tests ValueError is raised when ALLOWED_EXTENSIONS is not defined."""
        with pytest.raises(ValueError) as e:

            class Test_Parser(parser_base):
                pass

        assert "Class Test_Parser does not define ALLOWED_EXTENSIONS." in str(e)

    def test_missing_COLUMN_ASSIGNMENTS(self):
        """Tests ValueError is raised when COLUMN_ASSIGNMENTS is not defined."""
        with pytest.raises(ValueError) as e:

            class Test_Parser(parser_base):
                ALLOWED_EXTENSIONS = [".txt"]
                pass

        assert "Class Test_Parser does not define COLUMN_ASSIGNMENTS." in str(e)


##############################################################################
#################### Test the inner classes of parser_base ###################
##############################################################################
class TestParamDict:
    """Tests the parameter dictionary functionality, which stores a set of parameters,
    but allows relabelled indexing and contains."""

    class DummyLoadedParser:
        RELABELS = {"a": "b", "c": "d", ("e", "f"): "g"}
        RELABELS_REVERSE = {"b": "a", "d": "c", "g": ("e", "f")}
        params = {"a": 1, "c": 3, "f": 6, "g": 8}

    class DummyLoadedParserNoRelabel(DummyLoadedParser):
        relabel = False

    class DummyLoadedParserRelabel(DummyLoadedParser):
        relabel = True

    @pytest.mark.parametrize(
        "key, contains, contains_relabelled",
        [
            ("a", True, True),
            ("b", False, True),
            ("c", True, True),
            ("d", False, True),
            ("e", False, True),
            (("e", "f"), False, False),
            ("h", False, False),
        ],
    )
    def test_contains(self, key, contains, contains_relabelled):
        """Tests the parameter dictionary can be set and returns the correct value,
        including the use of tuples.

        Note, currently `in` keyword does not use the `__contains__` method. https://stackoverflow.com/a/38542777/1717003
        This may be confusing to users, but is not a bug in the code.
        """

        # Create the parameter dictionary.
        param_dict = parser_base.param_dict(
            map=self.DummyLoadedParserNoRelabel.params,
            parent=self.DummyLoadedParserNoRelabel,
        )  # type: ignore
        # Test its access
        assert (param_dict.__contains__(key)) == contains, "Contains failed."

        # Create the parameter dictionary.
        param_dict = parser_base.param_dict(
            map=self.DummyLoadedParserRelabel.params,
            parent=self.DummyLoadedParserRelabel,
        )  # type: ignore
        # Test its access
        assert (
            param_dict.__contains__(key)
        ) == contains_relabelled, "Contains relabelled failed."

    @pytest.mark.parametrize(
        "key, value_or_exception, value_or_exception_relabelled",
        [
            ("a", 1, 1),
            ("b", KeyError, 1),
            ("c", 3, 3),
            ("d", KeyError, 3),
            (
                "e",
                KeyError,
                8,
            ),  # TODO: Fix this test... pytest catches a try/except block.
            (("e", "f"), KeyError, 8),
            ("h", KeyError, KeyError),
        ],
    )
    def test_get(self, key, value_or_exception, value_or_exception_relabelled):
        """Tests the values returned by the parameter dictionary."""

        # Create the parameter dictionary.
        param_dict = parser_base.param_dict(
            parent=self.DummyLoadedParserNoRelabel,
            map=self.DummyLoadedParserNoRelabel.params,
        )  # type: ignore
        # Test its access
        if isinstance(value_or_exception, type) and issubclass(
            value_or_exception, Exception
        ):
            with pytest.raises(value_or_exception) as e:
                param_dict[key]
        else:
            assert param_dict[key] == value_or_exception

        # Test the relabelled value
        param_dict = parser_base.param_dict(
            parent=self.DummyLoadedParserRelabel,
            map=self.DummyLoadedParserRelabel.params,
        )  # type: ignore
        if isinstance(value_or_exception_relabelled, type) and issubclass(
            value_or_exception_relabelled, Exception
        ):
            with pytest.raises(value_or_exception_relabelled) as e:
                param_dict[key]


##############################################################################
############################ Test parser_base ################################
##############################################################################


class TestParserBase:
    """Tests the base parser class and its methods."""

    class TestInitialisation:
        """Tests the initialisation of the parser_base class."""

        def test_abstract(self):
            """Tests the parser base class is abstract"""
            with pytest.raises(TypeError) as e:
                parser = parser_base(None)
            assert "Cannot instantiate abstract class `parser_base`." in str(e)

        def test_no_parse_fn(self):
            """Tests an invalid implementations of the parser_base class with no parse method."""

            # No parser methods
            with pytest.raises(AttributeError) as e:

                class NoParserMethods(parser_base):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}

            assert "No parser methods found in `NoParserMethods` class." in str(e)

        def test_parse_fn_params(self):
            # One parameter
            with pytest.raises(NameError) as e:

                class ParserRelabelOverlapMono(parser_base):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                    RELABELS = {}
                    parse_test = lambda x: x

            assert (
                "First argument of static parser method must be 'file'. Is x." in str(e)
            )
            e = None

            # Two parameters
            class ParserRelabelOverlapDuo(parser_base):
                ALLOWED_EXTENSIONS = [".txt"]
                COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                RELABELS = {}
                parse_test = lambda file, y: y

            # With Header only
            class ParserRelabelOverlapHeader(parser_base):
                ALLOWED_EXTENSIONS = [".txt"]
                COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                RELABELS = {}
                parse_test = lambda file, y, header_only: header_only

        # def test_parser_fn_reuse_params(self):
        #     # Test that settings are used in the next parser function call.

        @pytest.mark.parametrize(
            "relabels, assertion",
            [
                (
                    {"a": "b", "c": "b"},
                    r"Duplicate - value 'b' in class `ParserRelabelOverlap.RELABELS` dictionary with keys 'a' and 'c'.",
                ),
                (
                    {"a": "b", "b": "c"},
                    r"Overlap - key:value pair 'b':'c' "
                    + r"in class `ParserRelabelOverlap.RELABELS` dictionary overlaps with key:value pair 'a':'b'.",
                ),
                (
                    {"a": "b", ("b", "c"): "d"},
                    r"Overlap - key:value pair '('b', 'c')':'d' with subkey 'b' as an overlapping entry "
                    + r"in class'ParserRelabelOverlap.RELABELS' dictionary with key:value pair 'a':'b'",
                ),
                (
                    {("a", "b"): "c", ("e", "c"): "d"},
                    r"Overlap - key:value pair '('e', 'c')':'d' with subkey 'c' as an overlapping entry "
                    + r"in class'ParserRelabelOverlap.RELABELS' dictionary with key:value pair '('a', 'b')':'c'.",
                ),
                (
                    {("a", "b"): "c", "b": "f"},
                    r"Duplicate - key 'b' is a duplicate key entry in `ParserRelabelOverlap.RELABELS` dictionary.",
                ),
                (
                    {("a", "b"): "c", ("b", "e"): "f"},
                    r"Duplicate - key 'b' from tuple '('b', 'e')' is a duplicate key entry in `ParserRelabelOverlap.RELABELS` dictionary.",
                ),
            ],
        )
        def test_RELABEL_overlap(self, relabels, assertion):
            """Tests the handling of overlapping values in the RELABELS dictionary"""
            # Test string string overlap
            with pytest.raises(ValueError) as e:

                class ParserRelabelOverlap(parser_base):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                    RELABELS = relabels
                    parse_test = lambda file: file

            # Trim the type and generic messages.
            substr = (
                repr(e.value)
                .split("\n")[0]
                .replace("ValueError(", "")
                .replace("'", "'")
            )
            print(substr)
            assert assertion in substr

    # Generate some basic implementation of the parser_base class.
