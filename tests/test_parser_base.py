""" "Tests the base parser classes and their methods"""

import tempfile

import numpy as np
import pytest

# Test the creation of a new parser class.
from pyNexafs.parsers._base import parserBase, parserMeta
from pyNexafs.types import dtype

##############################################################################
################### Test the inner classes of parserMeta ####################
##############################################################################


###
### The RELABELS dict
###
class TestRelabelsDict:
    """Tests the synonymous tuple functionality of the relabels dictionary."""

    class DummyParserRelabel:
        RELABELS = {"a": "b", "c": "d", ("e", "f"): "g", "i": dtype.PEY}

    @pytest.mark.parametrize(
        "key, value",
        [
            ("a", "b"),
            ("c", "d"),
            ("e", "g"),  # Individual keys from a tuple
            ("f", "g"),
            (("e", "f"), "g"),  # The tuple itself
            ("g", "g"),  # The value itself (no relabeling)
            (dtype.PEY, dtype.PEY),
            (dtype.PEY.value, "Partial Electron Yield"),
            ("i", "Partial Electron Yield"),
            ("Partial Electron Yield", "Partial Electron Yield"),
        ],
    )
    def test_get(self, key, value):
        """Tests the relabels dictionary can be set and returns the correct value,
        including the use of tuples."""

        # Create the relabel dictionary.
        relabel_dict = parserMeta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test its access
        assert relabel_dict[key] == value

    @pytest.mark.parametrize(
        "key, value",
        [
            ("a", "1"),
            ("c", "2"),
            ("e", "3"),
            ("f", "4"),
            (("e", "f"), "5"),
            ("g", "6"),
            (dtype.PEY, "Partial Electron Yield"),
            ("Partial Electron Yield", "Partial Electron Yield"),
            ("i", "Partial Electron Yield"),
        ],
    )
    def test_set(self, key, value):
        # Create the relabel dictionary.
        print("Type:", type(value))
        relabel_dict = parserMeta.relabels_dict(self.DummyParserRelabel.RELABELS)
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
        relabel_dict = parserMeta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test its access
        assert (key in relabel_dict) == contained

    def test_tuples(self):
        """Tests the synonymous tuple functionality of the relabels dictionary."""
        relabel_dict = parserMeta.relabels_dict(self.DummyParserRelabel.RELABELS)
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
                parserMeta.summary_param_list(elements, parent=self.ParserNoRelabels)  # type: ignore
            if overlapping_param:
                assert f"Summary Parameter `{overlapping_param}`" in str(e.value)
        else:
            parserMeta.summary_param_list(elements, parent=self.ParserNoRelabels)  # type: ignore

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
                parserMeta.summary_param_list(elements, parent=self.ParserRelabels)  # type: ignore
            if overlapping_param:
                assert f"Summary Parameter `{overlapping_param}`" in str(e.value)
        else:
            parserMeta.summary_param_list(elements, parent=self.ParserRelabels)  # type: ignore

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
        summary_params = parserMeta.summary_param_list(
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
        summary_params = parserMeta.summary_param_list(
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
        summary_params = parserMeta.summary_param_list(
            elements, parent=self.ParserNoRelabels
        )  # type: ignore
        # Test its access
        if isinstance(index_or_exception, type) and issubclass(
            index_or_exception, Exception
        ):
            with pytest.raises(
                index_or_exception,
                match=f"{query} is not in the list, or within a tuple element.",
            ):
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
        summary_params = parserMeta.summary_param_list(
            elements, parent=self.ParserRelabels
        )  # type: ignore
        # Test its access
        if isinstance(index_or_exception, type) and issubclass(
            index_or_exception, Exception
        ):
            with pytest.raises(
                index_or_exception,
                match=f"{query} is not in the list, or within a tuple element.",
            ):
                summary_params.index(query)


##############################################################################
############################ Test parserMeta ################################
##############################################################################


class TestParserMeta:
    """Tests the metaclass functionality on creating new Parser classes."""

    def test_missing_ALLOWED_EXTENSIONS(self):
        """Tests ValueError is raised when ALLOWED_EXTENSIONS is not defined."""
        with pytest.raises(ValueError) as e:

            class Test_Parser(parserBase):
                pass

        assert "Class Test_Parser does not define ALLOWED_EXTENSIONS." in str(e)

    def test_missing_COLUMN_ASSIGNMENTS(self):
        """Tests ValueError is raised when COLUMN_ASSIGNMENTS is not defined."""
        with pytest.raises(ValueError) as e:

            class Test_Parser(parserBase):
                ALLOWED_EXTENSIONS = [".txt"]

        assert "Class Test_Parser does not define COLUMN_ASSIGNMENTS." in str(e)


##############################################################################
#################### Test the inner classes of parserBase ###################
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
        param_dict = parserBase.param_dict(
            map=self.DummyLoadedParserNoRelabel.params,
            parent=self.DummyLoadedParserNoRelabel,
        )  # type: ignore
        # Test its access
        assert (param_dict.__contains__(key)) == contains, "Contains failed."

        # Create the parameter dictionary.
        param_dict = parserBase.param_dict(
            map=self.DummyLoadedParserRelabel.params,
            parent=self.DummyLoadedParserRelabel,
        )  # type: ignore
        # Test its access
        assert (param_dict.__contains__(key)) == contains_relabelled, (
            "Contains relabelled failed."
        )

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
        param_dict = parserBase.param_dict(
            parent=self.DummyLoadedParserNoRelabel,
            map=self.DummyLoadedParserNoRelabel.params,
        )  # type: ignore
        # Test its access
        if isinstance(value_or_exception, type) and issubclass(
            value_or_exception, Exception
        ):
            with pytest.raises(value_or_exception):
                param_dict[key]
        else:
            assert param_dict[key] == value_or_exception

        # Test the relabelled value
        param_dict = parserBase.param_dict(
            parent=self.DummyLoadedParserRelabel,
            map=self.DummyLoadedParserRelabel.params,
        )  # type: ignore
        if isinstance(value_or_exception_relabelled, type) and issubclass(
            value_or_exception_relabelled, Exception
        ):
            with pytest.raises(value_or_exception_relabelled):
                param_dict[key]


##############################################################################
############################ Test parserBase ################################
##############################################################################


class TestParserBase:
    """Tests the base parser class and its methods."""

    class TestInitialisation:
        """Tests the initialisation of the parserBase class."""

        def test_abstract(self):
            """Tests the parser base class is abstract"""
            with pytest.raises(TypeError) as e:
                parserBase(None)
            assert "Cannot instantiate abstract class `parserBase`." in str(e)

        def test_no_parse_fn(self):
            """Tests an invalid implementations of the parserBase class with no parse method."""

            # No parser methods
            with pytest.raises(AttributeError) as e:

                class NoParserMethods(parserBase):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}

            assert "No parser methods found in `NoParserMethods` class." in str(e)

        def test_parse_fn_params(self):
            # One parameter
            with pytest.raises(NameError) as e:

                def parse_fn(x):
                    return x

                class ParserRelabelOverlapMono(parserBase):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                    RELABELS = {}
                    parse_test = parse_fn

            assert (
                "First argument of static parser method must be 'file'. Is x." in str(e)
            )
            e = None

            # Two parameters
            def parse_fn_duo(file, y):
                return y

            class ParserRelabelOverlapDuo(parserBase):
                ALLOWED_EXTENSIONS = [".txt"]
                COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                RELABELS = {}
                parse_test = parse_fn_duo

            # With Header only
            def parser_fn_header(file, y, header_only):
                return header_only

            class ParserRelabelOverlapHeader(parserBase):
                ALLOWED_EXTENSIONS = [".txt"]
                COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                RELABELS = {}
                parse_test = parser_fn_header

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
                (
                    {("a", dtype.PEY): "c", dtype.PEY: "f"},
                    r"Duplicate - dtype key 'Partial Electron Yield' in class `ParserRelabelOverlap.RELABELS` dictionary with value 'f'.",
                ),
            ],
        )
        def test_RELABEL_overlap(self, relabels, assertion):
            """Tests the handling of overlapping values in the RELABELS dictionary"""
            # Test string string overlap
            with pytest.raises(ValueError) as e:

                def parse_fn(file):
                    return file

                class ParserRelabelOverlap(parserBase):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                    RELABELS = relabels
                    parse_test = parse_fn

            # Trim the type and generic messages.
            substr = (
                repr(e.value)
                .split("\n")[0]
                .replace("ValueError(", "")
                .replace("'", "'")
            )
            print(substr)
            assert assertion in substr

    class TestGetItem:
        """Tests the `__getitem__` method of the parserBase class."""

        class ParserGetitem(parserBase):
            ALLOWED_EXTENSIONS = [".txt"]
            COLUMN_ASSIGNMENTS = {"x": "a", "y": ["b", dtype.TEY]}
            RELABELS = {
                "a": "b",
                "c": dtype.E,
                "d": "e",
                "DrainCurrent": dtype.TEY,
            }

            @classmethod
            def parse_txt(cls, file):
                fake_data = np.random.rand(10, 4)  # 10 rows, 4 channels
                labels = [
                    "a",
                    "c",
                    "f",
                    "DrainCurrent",
                ]  # Original labels for the channels
                units = ["", "", "", ""]  # Units for the channels
                params = {"filename": file}  # Example parameters
                return fake_data, labels, units, params

        @pytest.mark.parametrize(
            "key,err",
            [
                ("a", None),
                ("b", None),
                ("g", KeyError),
                ("c", None),
                (dtype.E, None),
                (dtype.PFY, KeyError),
            ],
        )
        def test_getitem(self, key, err):
            """Tests the `__getitem__` method of the parserBase class,
            which should return a 1D channel, using relabels if necessary."""

            # Create a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("Fake data for testing.")
                obj = self.ParserGetitem(f)

                # Test the getitem method
                if err:
                    with pytest.raises(err):
                        _ = obj[key]
                else:
                    channel = obj[key]
                    assert isinstance(channel, np.ndarray), (
                        "Returned channel is not a numpy array."
                    )
                    assert channel.ndim == 1, "Returned channel is not 1D."
