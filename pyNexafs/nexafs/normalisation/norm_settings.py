"""
Module to handle default normalisation settings to be used across datasets.

In
"""

from enum import Enum, EnumType
from abc import ABC, abstractmethod, ABCMeta
from io import TextIOWrapper
import yaml
import os


class configMeta(ABCMeta):
    """
    Metaclass for all normalisation settings classes.

    Ensures every subclass has default arg values, so that configurations can
    be initialised and set.

    Raises
    ------
    NotImplementedError
        If the subclass does not have default arguments for __init__ variables.
    """

    def __new__(cls: type, name: str, bases: tuple[type, ...], dct: dict) -> type:
        """
        The metaclass' class creator.

        Check that each constructor argument has a default value.

        Parameters
        ----------
        cls : type
            The metaclass type.
        name : str
            The name of the class being created.
        bases : tuple[type]
            The base classes of the class being created.
        dct : dict
            The class attributes and methods dictionary.

        Returns
        -------
        type
            The new class type with the metaclass applied.
        """
        # Check the arguments for the class constructor and subclasses
        init_fns = []
        # The the class init
        if "__init__" in dct:
            init_fns.append(dct["__init__"])
        # Get subclass inits
        for key, val in dct.items():
            if hasattr(val, "__init__") and key != "__init__" and isinstance(val, type):
                init_fns.append(val.__init__)

        # Check all inits for default arguments to allow instantiation
        for init_fn in init_fns:
            args: tuple = init_fn.__code__.co_varnames[1:]
            defs: list = (
                init_fn.__defaults__ if init_fn.__defaults__ is not None else []
            )
            if "args" in args:
                args = args[: args.index("args")]
            if len(args) != len(defs):
                raise NotImplementedError(
                    f"Class {name} must have default arguments for __init__ variables {init_fn.__code__.co_varnames[: -(len(init_fn.__code__.co_varnames - 1 - len(init_fn.__defaults__)))]}"
                )
        return super().__new__(cls, name, bases, dct)


class configBase(ABC, metaclass=configMeta):
    """
    Base abstract class that represents the normalisation/background settings for a scan.

    Parameters
    ----------
    scan : scan_base
        The scan number to apply the normalisation settings to.
    apply_to : list[int] | list[str] | None, optional
        The list of scan channel indices or scan channel names to apply the normalisation to.
        If None, apply to all channels (default).
    """

    def __init__(self, apply_to: list[str] | None = None) -> None:
        # Internal variables
        self._apply_to: list[str] | None = apply_to
        """The list of scan scan channel names to apply the normalisation to.
        If None, apply to all channels."""

    @property
    @abstractmethod
    def is_valid(self) -> bool:
        """
        Check if the normalisation settings are valid, i.e. data paths exist.

        Requires implementation in subclasses.

        Returns
        -------
        bool
            True if the settings are valid, False otherwise.
        """
        pass

    @property
    def apply_to(self) -> list[str] | None:
        """
        The list of scan channel names to apply the normalisation to.

        None implies apply normalisation to all channels.

        Parameters
        ----------
        apply_to : list[str] | None
            The list of scan channel names to apply the normalisation to.

        Returns
        -------
        list[str] | None
            The list of scan channel names to apply the normalisation to.
        """
        return self._apply_to

    @apply_to.setter
    def apply_to(self, apply_to: list[str] | None):
        self._apply_to = apply_to


class configSeries(metaclass=configMeta):
    """
    Base class for a series of sequential normalisation configurations.

    Parameters
    ----------
    configs : list[normConfig] | None, optional
        The sequential list of normalisation configurations.
    """

    def __init__(self, configs: list[configBase] | None = None) -> None:
        if configs is None:
            self._configs: list[configBase] = []
        else:
            self._configs = configs

    def __len__(self):
        """
        Get the number of normalisation settings in the series.

        Returns
        -------
        int
            The number of normalisation settings in the series.
        """
        return self._configs.__len__()

    def __getitem__(self, index: int) -> configBase:
        """
        Get the normalisation settings at the specified index.

        Parameters
        ----------
        index : int
            The index of the normalisation settings to get.

        Returns
        -------
        configBase
            The normalisation settings at the specified index.
        """
        return self._configs[index]

    def __setitem__(self, index: int, value: configBase) -> None:
        """
        Set the normalisation settings at the specified index.

        Parameters
        ----------
        index : int
            The index of the normalisation settings to set.
        value : configBase
            The normalisation settings to set at the specified index.
        """
        self._configs[index] = value

    def yaml(self) -> str:
        """
        Convert the normalisation settings to yaml format.

        Returns
        -------
        str
            The yaml formatted string.
        """
        return yaml.dump(self._configs)

    def save(self, path: str, override: bool = False) -> None:
        """
        Save the normalisation settings to a yaml file.

        Parameters
        ----------
        path : str
            The path to save the settings file.
        override : bool, optional
            If True, override the file if it already exists, by default False.

        Raises
        ------
        FileExistsError
            If the file already exists and override is False.
        """
        if not path.endswith(".yaml") or not path.endswith(".yml"):
            path += ".yml"

        if not override and os.path.exists(path):
            raise FileExistsError(f"File {path} already exists.")

        with open(path, "w") as file:
            file.write(self.yaml())

    @staticmethod
    def loadstream(stream: str | TextIOWrapper) -> "configSeries":
        """
        Import a list of sequential normalisation settings from a yaml stream.

        Parameters
        ----------
        stream : str
            The yaml formatted string.

        Returns
        -------
        configSeries
            The loaded series of normalisation settings.
        """
        # Get list of normConfig class + subclasses
        config_clss: list[type[configBase]] = [configBase] + configBase.__subclasses__()
        # Get list of class attributes for settings
        attr_clss: list[type] = [
            val
            for cls in config_clss
            for val in cls.__dict__.values()
            if isinstance(val, type)
        ]
        # Use the union to define new objects
        clss = config_clss + attr_clss

        # Define the extra context for constructors
        def default_ctor(
            loader: yaml.SafeLoader, tag_suffix: str, node: yaml.MappingNode
        ):
            """
            Default constructor for unknown objects.

            Parameters
            ----------
            loader : yaml.SafeLoader
                The yaml loader to use.
            tag_suffix : str
                The suffix of the tag to match the class name.
            node : yaml.MappingNode
                The yaml node to construct the object from.

            Returns
            -------
            configBase
                The constructed object of the matching class.
            """
            # Get the class name from the tag
            class_name = tag_suffix.split(".")[-1]
            # Find the class that matches the tag
            for cls in clss:
                if cls.__name__ == class_name:
                    # Check if an enumerate type
                    if isinstance(cls, EnumType) and len(node.value) == 1:
                        n = node.value[0]
                        if "int" in n.tag and hasattr(n, "value"):
                            return cls(int(n.value))
                    else:
                        new_obj = cls()
                        for key, value in loader.construct_mapping(node).items():
                            setattr(new_obj, key, value)
                        return new_obj

        # Add the constructor to the yaml loader
        yaml.SafeLoader.add_multi_constructor("", default_ctor)
        data: list[configBase] = yaml.load(stream, Loader=yaml.SafeLoader)
        return configSeries(data)

    @staticmethod
    def load(path: str) -> "configSeries":
        """
        Load the sequential list of normalisation settings from a yaml file.

        Parameters
        ----------
        path : str
            The path to load the settings file.

        Returns
        -------
        configSeries
            The loaded series of normalisation settings.
        """
        with open(path, "r") as file:
            series = configSeries.loadstream(file)
        return series


class normMethod(Enum):
    """
    Different methods of performing channel normalisation.

    Attributes
    ----------
    NONE : 0
        No normalisation of the scan channel by the background channel.
    BACKGROUND : 1
        Subtract the background channel from the scan channel.
    FLUX : 2
        Divide the scan channel by the normalized (maximum value=1) background channel amplitude.
    """

    NONE = 0
    BACKGROUND = 1
    FLUX = 2


class configExternal(configBase, ABC):
    """
    Abstract class for external data normalisation settings.

    Allows user to specify temporal proximity and keyword matching for external data.

    Parameters
    ----------
    channel_selection : extSelection, optional
        The method of selecting the external data, by default extSelection.NONE.
    keyword : str | None, optional
        The keyword to use for selecting the external data, by default None.
    path : str | None, optional
        The path to the external data, by default None.
    apply_to : list[str] | None, optional
        The list of scan channel names to apply the normalisation to, by default None.
    """

    class extSelection(Enum):
        """
        Definitions for choosing external data selection.

        Allow the user to describe how the data selection should be accessed.

        Attributes
        ----------
        NONE : 0
            No background normalisation of the scan.
        MOST_RECENT_PRIOR_KEYWORD : 1
            Use the most recent scan containing a keyword prior to the existing scan.
        CLOSEST_TIME_KEYWORD : 2
            Use the closest scan in time containing a keyword to the existing scan.
        MOST_RECENT_POST_KEYWORD : 3
            Use the most recent scan containing a keyword after the existing scan.
        FIXED_SCAN : 4
            Use a fixed scan object for background normalisation.
        """

        NONE = 0
        MOST_RECENT_PRIOR_KEYWORD = 1
        CLOSEST_TIME_KEYWORD = 2
        MOST_RECENT_POST_KEYWORD = 3
        FIXED_SCAN = 4

    def __init__(
        self,
        channel_selection: extSelection = extSelection.NONE,
        keyword: str | None = None,
        path: str | None = None,
        apply_to: list[str] | None = None,
    ):
        # Initialise the normConfig class
        super().__init__(apply_to)
        # Initialise the internal variables
        self._channel_selection: configExternal.extSelection = channel_selection
        """The external data selection method."""
        self._keyword: str | None = keyword
        """The keyword selection used for various CHANNEL_SELECTION enumerate options."""
        self._path: str | None = path
        """The path selection corresponding to a file for the CHANNEL_SELECTION.FIXED_SCAN enumerate,
        or a directory for the CHANNEL_SELECTION.CLOSEST_TIME_KEYWORD enumerate."""

    @property
    def selection(self) -> extSelection:
        return self._channel_selection

    @selection.setter
    def selection(self, selection: extSelection):
        self._channel_selection = selection
        if (
            selection is self.extSelection.NONE
            or selection is self.extSelection.FIXED_SCAN
        ):
            self._keyword = None

    @property
    def keyword(self) -> str | None:
        return self._keyword

    @keyword.setter
    def keyword(self, keyword: str | None):
        self._keyword = keyword

    @property
    def path(self) -> str | None:
        return self._path

    @path.setter
    def path(self, path: str | None):
        if os.path.exists(path):
            self._path = path
        else:
            raise FileNotFoundError(f"Path `{path}` does not exist.")


class configChannel(configBase):
    """
    Configurations where a particular channel is required to perform the normalisation.

    Parameters
    ----------
    norm_method : normMethod, optional
        The method of normalisation, by default normMethod.NONE.
    channel_name : str | None, optional
        The name of the channel to apply the normalisation to, by default None.
    apply_to : list[str] | None, optional
        The list of scan channel names to apply the normalisation to, by default None.
    """

    # Add the normMethod Enum as a class attribute
    normMethod = normMethod

    def __init__(
        self,
        norm_method: normMethod = normMethod.NONE,
        channel_name: str | None = None,
        apply_to: list[str] | None = None,
    ) -> None:
        # Initialise the normConfig class
        super().__init__(apply_to)

        # Initialise the internal variables
        self._norm_method: configChannel.normMethod = norm_method
        """The normalisation method choice."""
        self._channel_name: str | None = channel_name
        """The normalisation channel name choice."""

    @property
    def method(self) -> normMethod:
        """
        Property to get/set  the normalisation method.

        Can be one of the following:
        - norm_method.NONE
            No normalisation of the scan channel by the background channel.
        - norm_method.BACKGROUND
            Subtract the background channel from the scan channel.
        - norm_method.FLUX
            Divide the scan channel by the normalized (maximum value=1) background channel amplitude.

        Parameters
        ----------
        method : normMethod
            The normalisation method.

        Returns
        -------
        normMethod
            The normalisation method.
        """
        return self._norm_method

    @method.setter
    def method(self, method: normMethod):
        self._norm_method = method
        if method is self.normMethod.NONE:
            self._keyword = None
            self._path = None

    @property
    def channel(self) -> str:
        """
        The normalisation channel name.

        Parameters
        ----------
        channel_name : str
            The normalisation channel name.

        Returns
        -------
        str
            The normalisation channel name.
        """
        if self._channel_name is None:
            raise ValueError("Channel name not defined.")
        return self._channel_name

    @channel.setter
    def channel(self, channel_name: str | None):
        self._channel_name = channel_name

    @configBase.is_valid.getter
    def is_valid(self) -> bool:
        """
        Check if the normalisation settings are valid, i.e. data paths exist.

        Returns
        -------
        bool
            True if the settings are valid, False otherwise.
        """
        return self._channel_name is not None or self.method is self.normMethod.NONE


class configExternalChannel(configChannel, configExternal):
    """
    Configurations for a single channel normalisation using external data.

    Parameters
    ----------
    channel_selection : extSelection, optional
        The method of selecting the external data, by default extSelection.NONE.
    keyword : str | None, optional
        The keyword to use for selecting the external data, by default None.
    path : str | None, optional
        The path to the external data, by default None.
    norm_method : normMethod, optional
        The method of normalisation, by default normMethod.NONE.
    channel_name : str | None, optional
        The name of the channel to apply the normalisation to, by default None.
    apply_to : list[str] | None, optional
        The list of scan channel names to apply the normalisation to, by default None.
    """

    # Add enumerates as class attributes
    extSelection = configExternal.extSelection
    normMethod = normMethod

    def __init__(
        self,
        channel_selection: extSelection = extSelection.NONE,
        keyword: str | None = None,
        path: str | None = None,
        norm_method: normMethod = normMethod.NONE,
        channel_name: str | None = None,
        apply_to: list[str] | None = None,
    ) -> None:
        # Initialise the normConfigChannel class
        configChannel.__init__(self, norm_method, channel_name, apply_to=apply_to)
        configExternal.__init__(
            self, channel_selection, keyword, path, apply_to=apply_to
        )

    @configBase.is_valid.getter
    def is_valid(self) -> bool:
        """
        Check if the normalisation settings are valid, i.e. data paths exist.

        Returns
        -------
        bool
            True if the settings are valid, False otherwise.
        """
        match self.selection:
            case (
                self.extSelection.MOST_RECENT_PRIOR_KEYWORD
                | self.extSelection.CLOSEST_TIME_KEYWORD
                | self.extSelection.MOST_RECENT_POST_KEYWORD
            ):
                return (
                    self._keyword is not None
                    and self._path is not None
                    and os.path.exists(self._path)
                )
            case self.extSelection.FIXED_SCAN:
                return self._path is not None and os.path.exists(self._path)
            case self.extSelection.NONE:
                return True
            case _:
                return False


class configExternalData(configBase):
    """
    Configurations for external data normalisation.

    Includes the method of normalisation and the source of the external data.
    Does not include the data itself, when saving settings.

    Parameters
    ----------
    norm_method : normMethod, optional
        The method of normalisation, by default normMethod.NONE.
    norm_source : normSource, optional
        The source of the normalisation data, by default normSource.NONE.
    handling_method : handlingMethod, optional
        The method of handling the normalisation data, by default handlingMethod.NONE.
    data_path : str, optional
        The path to the normalisation data, by default None.
    apply_to : list[str] | None, optional
        The list of scan channel names to apply the normalisation to, by default None.
    """

    # Add the normMethod Enum as a class attribute to emulate data.
    normMethod = normMethod

    class normSource(Enum):
        """
        Enumerate flags to define the source of the external data.

        Attributes
        ----------
        NONE : int
            No external data source.
        FILE : int
            External data source is a file.
        ARRAY : int
            External data source is an array.
        """

        NONE = 0
        FILE = 1
        ARRAY = 2

    class handlingMethod(Enum):
        """
        Enumerate flags to define how to apply external data to the scan.

        Attributes
        ----------
        NONE : int
            No data handling. Data must match the scan shape.
        INTERPOLATE : int
            Linearly interpolate the external data to match the scan shape.
        SPLINE : int
            Use a spline interpolation to match the external data to the scan shape.
        """

        NONE = 0
        INTERPOLATE = 1
        SPLINE = 2

    def __init__(
        self,
        norm_method: normMethod = normMethod.NONE,
        norm_source: normSource = normSource.NONE,
        handling_method: handlingMethod = handlingMethod.NONE,
        data_path: str | None = None,
        apply_to: list[str] | None = None,
    ) -> None:
        self._norm_method: configExternalData.normMethod = norm_method
        """Internal variable to track the normalisation method."""
        self._norm_source: configExternalData.normSource = norm_source
        """Internal variable to track the normalisation data source."""
        self._handling_method: configExternalData.handlingMethod = handling_method
        """Internal variable to track the normalisation data handling method."""
        self._data_path: str | None = data_path
        """Internal variable to track the normalisation data path."""
        # Initialise the configBase class
        super().__init__(apply_to)

    @property
    def method(self) -> normMethod:
        """
        A property defining the normalisation method.

        Can be one of the following:
        - DATA_METHOD.NONE
            No normalisation performed.
        - DATA_METHOD.BACKGROUND
            Subtract the background data from the scan data.
        - DATA_METHOD.FLUX
            Divide the scan data by the normalized (maximum value=1) background data amplitude.

        Returns
        -------
        normMethod
            The normalisation method.
        """
        return self._norm_method

    @method.setter
    def method(self, method: normMethod) -> None:
        self._norm_method = method

    @property
    def handling(self) -> handlingMethod:
        """
        A property defining the normalisation data handling method.

        Parameters
        ----------
        handling_method : handlingMethod
            The normalisation data handling method. Can be one of the following:
            - HANDLING_METHOD.NONE
                No data handling. Data must match the scan shape.
            - HANDLING_METHOD.INTERPOLATE
                Linearly interpolate the external data to match the scan shape.
            - HANDLING_METHOD.SPLINE
                Use a spline interpolation to match the external data to the scan shape.

        Returns
        -------
        handlingMethod
            The normalisation data handling method.
        """
        return self._handling_method

    @handling.setter
    def handling(self, handling_method: handlingMethod) -> None:
        self._handling_method = handling_method

    @property
    def source(self) -> normSource:
        """
        A property defining the normalisation data source.

        Parameters
        ----------
        source : normSource
            The normalisation data source. Can be one of the following:
            - SOURCE.NONE
                No external data source.
            - SOURCE.FILE
                External data source is a file.
            - SOURCE.ARRAY
                External data source is an array.

        Returns
        -------
        normSource
            The normalisation data source.
        """
        return self._norm_source


class configBackground(configBase):
    """
    Configurations for using a background scan for normalisation.
    """

    # TODO: Implement this class.


class normConfigEdges(configBase):
    """
    A class to handle settings for edge normalisation.

    Parameters
    ----------
    pre_edge_norm_method : scan_normalised_edges.PREEDGE_NORM_TYPE, optional
        Method to normalise the pre-edge region, by default scan_normalised_edges.PREEDGE_NORM_TYPE.NONE.
    post_edge_norm_method : scan_normalised_edges.POSTEDGE_NORM_TYPE, optional
        Method to normalise the post-edge region, by default scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE.
    pre_edge_domain : tuple[float, float] | None, optional
        Domain of the pre-edge region to normalise, by default None.
    post_edge_domain : tuple[float, float] | None, optional
        Domain of the post-edge region to normalise, by default None.
    pre_edge_level : float | None, optional
        Baseline level of the pre-edge region, by default None.
    post_edge_level : float | None, optional
        Baseline level of the post-edge region, by default None.
    apply_to : list[str] | None, optional
        The list of scan channel names to apply the normalisation to, by default None.

    Examples
    --------
    An example to normalise at the pre-edge and post-edge of sulfur K-edge.
    >>> edge_config = norm_config_edges(pre_edge_norm_method=scan_normalised_edges.PREEDGE_NORM_TYPE.LINEAR,
    ...                                 post_edge_norm_method=scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE,
    ...                                 pre_edge_domain=(2460, 2465),
    ...                                 post_edge_domain=(2500, 2510),
    ...                                 pre_edge_level=0.0,
    ...                                 post_edge_level=1.0)
    """

    class PREEDGE_NORM_TYPE(Enum):
        """
        Enumerated definitions for pre-edge normalisation.

        Attributes
        ----------
        NONE : int
            No pre-edge normalisation.
        CONSTANT : int
            Constant normalisation over the edge domain.
        LINEAR : int
            Linear normalisation over the edge domain.
        EXPONENTIAL : int
            EXPONENTIAL normalisation over the edge domain.
        """

        NONE = 0
        CONSTANT = 1
        LINEAR = 2
        EXPONENTIAL = 3

    class POSTEDGE_NORM_TYPE(Enum):
        """
        Enumerated definitions for post-edge normalisation.

        Attributes
        ----------
        NONE : int
            No post-edge normalisation.
        CONSTANT : int
            Constant normalisation over the edge domain.
        """

        NONE = 0
        CONSTANT = 1

    DEFAULT_PRE_EDGE_LEVEL_CONSTANT = 0.0
    """The constant offset for the pre-edge normalisation. I.e. Pre-edge at 0.0"""
    DEFAULT_PRE_EDGE_LEVEL_LINEAR = 0.0
    """The constant gradient for the pre-edge normalisation. I.e. Linear gradient at 0.0"""
    DEFAULT_PRE_EDGE_LEVEL_EXP = 0.0
    """The constant exponential level for the pre-edge normalisation. I.e. Exponential level at 0.0"""
    DEFAULT_POST_EDGE_LEVEL_CONSTANT = 1.0
    """The constant offset for the post-edge normalisation. I.e. Post-edge at 1.0"""

    def __init__(
        self,
        pre_edge_norm_method: PREEDGE_NORM_TYPE = PREEDGE_NORM_TYPE.NONE,
        post_edge_norm_method: POSTEDGE_NORM_TYPE = POSTEDGE_NORM_TYPE.NONE,
        pre_edge_domain: tuple[float, float] | list[int] | None = None,
        post_edge_domain: tuple[float, float] | list[int] | None = None,
        pre_edge_level: float | None = None,
        post_edge_level: float | None = None,
        apply_to: list[str] | None = None,
    ):
        super().__init__(apply_to)
        # Initialise some internal variables
        self._pre_edge_level = None
        self._post_edge_level = None
        self.pre_edge_norm_method = pre_edge_norm_method
        self.post_edge_norm_method = post_edge_norm_method
        self.pre_edge_domain = pre_edge_domain
        self.post_edge_domain = post_edge_domain

        # Set default values if not defined
        if pre_edge_level is None:
            self.pre_edge_level = self.DEFAULT_PRE_EDGE_LEVEL_CONSTANT
        else:
            self.pre_edge_level = pre_edge_level
        if post_edge_level is None:
            self.post_edge_level = self.DEFAULT_PRE_EDGE_LEVEL_CONSTANT
        else:
            self.post_edge_level = post_edge_level

    # Validation
    @configBase.is_valid.getter
    def is_valid(self) -> bool:
        """
        Check if the normalisation settings are valid.

        The minimum requirements for an edge normalisation are the pre-edge and post-edge domains.

        Returns
        -------
        bool
            True if the settings are valid, False otherwise.
        """
        # TODO more intelligent validation.
        return (
            self._pre_edge_domain is not None  # A domain exists
            and self._post_edge_domain is not None  # A domain exists
            # Check the domains are valid
            and (
                (
                    isinstance(self._pre_edge_domain, tuple)
                    and self._pre_edge_domain[0] < self._pre_edge_domain[1]
                )
                or not isinstance(self._pre_edge_domain, tuple)
            )
            and (
                (
                    isinstance(self._post_edge_domain, tuple)
                    and self._post_edge_domain[0] < self._post_edge_domain[1]
                )
                or not isinstance(self._post_edge_domain, tuple)
            )
        )

    # Edge Normalisation Method
    @property
    def pre_edge_norm_method(self) -> PREEDGE_NORM_TYPE:
        return self._pre_edge_norm_method

    @pre_edge_norm_method.setter
    def pre_edge_norm_method(self, method: PREEDGE_NORM_TYPE):
        # Check if method is valid
        if method in normConfigEdges.PREEDGE_NORM_TYPE:
            self._pre_edge_norm_method = method
        else:
            raise ValueError(f"{method} not in {normConfigEdges.PREEDGE_NORM_TYPE}.")
        # If pre-edge not defined, set to default level
        if self.pre_edge_level is None:
            match method:
                case normConfigEdges.PREEDGE_NORM_TYPE.CONSTANT:
                    self.pre_edge_level = self.DEFAULT_PRE_EDGE_LEVEL_CONSTANT
                case normConfigEdges.PREEDGE_NORM_TYPE.LINEAR:
                    self.pre_edge_level = self.DEFAULT_PRE_EDGE_LEVEL_LINEAR
                case normConfigEdges.PREEDGE_NORM_TYPE.EXPONENTIAL:
                    self.pre_edge_level = self.DEFAULT_PRE_EDGE_LEVEL_EXP
                case _:
                    # Do nothing if not defined / NONE.
                    pass

        # Change pre-edge level by default to a reasonable value if setting exponential.
        if (
            method is normConfigEdges.PREEDGE_NORM_TYPE.EXPONENTIAL
            and self.pre_edge_level <= 0
        ):
            self._pre_edge_level = self.DEFAULT_PRE_EDGE_LEVEL_EXP

        self._pre_edge_norm_method = method

    @pre_edge_norm_method.deleter
    def pre_edge_norm_method(self) -> None:
        self._pre_edge_norm_method = normConfigEdges.PREEDGE_NORM_TYPE.NONE

    @property
    def post_edge_norm_method(self) -> POSTEDGE_NORM_TYPE:
        """
        Property to define the type of normalisation performed on the post-edge.

        Parameters
        ----------
        vals : normConfigEdges.EDGE_NORM_TYPE
            LINEAR, EXPONENTIAL or NONE.

        Returns
        -------
        normConfigEdges.EDGE_NORM_TYPE
            The current normalisation type.
        """
        return self._post_edge_norm_method

    @post_edge_norm_method.setter
    def post_edge_norm_method(self, method: POSTEDGE_NORM_TYPE):
        if method in normConfigEdges.POSTEDGE_NORM_TYPE:
            self._post_edge_norm_method = method
        else:
            raise ValueError(f"{method} not in {normConfigEdges.POSTEDGE_NORM_TYPE}.")

        # Set default if None
        if self._post_edge_level is None:
            match method:
                case normConfigEdges.POSTEDGE_NORM_TYPE.CONSTANT:
                    self._post_edge_level = (
                        normConfigEdges.DEFAULT_POST_EDGE_LEVEL_CONSTANT
                    )
                case _:
                    # Do nothing if not defined / NONE.
                    pass

    @post_edge_norm_method.deleter
    def post_edge_norm_method(self) -> None:
        self._post_edge_norm_method = normConfigEdges.POSTEDGE_NORM_TYPE.NONE

    # Edge Normalisation Domain
    @property
    def pre_edge_domain(self) -> tuple[float, float] | list[int] | None:
        """
        A property defining the pre-edge domain of normalisation.

        If setting and `pre_edge_norm_method` is `NONE`, will set new normalisation enumerate to `LINEAR`.

        Parameters
        ----------
        vals : list[int] | tuple[float, float] | None
            Can be defined using a list of included indices matching x datapoints.
            Alternatively a tuple of two floats defining the inclusive endpoints.
            Alternatively None stops normalisation being performed on the pre-edge.

        Returns
        -------
        list[int] | tuple[float, float] | None
            Same as vals parameter.
        """
        return self._pre_edge_domain

    @pre_edge_domain.setter
    def pre_edge_domain(self, domain: tuple[float, float] | list[int] | None):
        if isinstance(domain, list):
            self._pre_edge_domain = domain.copy()  # non-immutable
        elif isinstance(domain, tuple) and len(domain) == 2:
            if domain[0] > domain[1]:
                raise ValueError(
                    f"Pre-edge domain contains lower-bound `{domain[0]}` which is greater than upperbound `{domain[1]}`."
                )
            self._pre_edge_domain = domain  # immutable, can't modify.
        elif domain is None:
            # Remove vals
            del self.pre_edge_domain
        else:
            raise ValueError(
                "The pre-edge domain needs to be defined by a list of integer indices, a tuple of inclusive endpoints or None."
            )
        # Default normalisation to linear if not already defined.
        if (
            self._pre_edge_norm_method is normConfigEdges.PREEDGE_NORM_TYPE.NONE
            and domain is not None
        ):
            self._pre_edge_norm_method = normConfigEdges.PREEDGE_NORM_TYPE.LINEAR

    @pre_edge_domain.deleter
    def pre_edge_domain(self):
        self._post_edge_domain = None

    @property
    def post_edge_domain(self) -> list[int] | tuple[float, float] | None:
        """
        A property defining the post-edge domain of normalisation.

        If setting and `post__apply_normalisation` is `NONE`, will set new normalisation enumerate to `LINEAR`.

        Parameters
        ----------
        vals : list[int] | tuple[float, float] | None
            Can be defined using a list of included indices matching x datapoints.
            Alternatively a tuple of two floats defining the inclusive endpoints.
            Alternatively None stops normalisation being performed on the post-edge.

        Returns
        -------
        list[int] | tuple[float, float] | None
            Same as vals parameter.
        """

        return self._post_edge_domain

    @post_edge_domain.setter
    def post_edge_domain(self, vals: list[int] | tuple[float, float] | None) -> None:
        if isinstance(vals, list):
            self._post_edge_domain = vals.copy()  # non-immutable
        elif isinstance(vals, tuple) and len(vals) == 2:
            if vals[0] > vals[1]:
                raise ValueError(
                    f"Post-edge domain contains lower-bound `{vals[0]}` which is greater than upperbound `{vals[1]}`."
                )
            self._post_edge_domain = vals  # immutable, can't modify.
        elif vals is None:
            # Remove vals and perform any other
            del self.post_edge_domain
        else:
            raise ValueError(
                "The post-edge domain needs to be defined by a list of integer indices, a tuple of inclusive endpoints or None."
            )
        # Default normalisation to constant if not already defined.
        if (
            self._post_edge_norm_method is normConfigEdges.POSTEDGE_NORM_TYPE.NONE
            and vals is not None
        ):
            self._post_edge_norm_method = normConfigEdges.POSTEDGE_NORM_TYPE.CONSTANT

    @post_edge_domain.deleter
    def post_edge_domain(self):
        self._post_edge_domain = None

    # Edge Normalisation Level
    @property
    def pre_edge_level(self) -> float:
        """
        Property to define the normalisation level (constant) for the pre-edge.

        Parameters
        ----------
        vals : float
            The normalisation average.

        Returns
        -------
        float
            The normalisation level.
        """
        return self._pre_edge_level

    @pre_edge_level.setter
    def pre_edge_level(self, level: float):
        if self.pre_edge_norm_method is normConfigEdges.PREEDGE_NORM_TYPE.EXPONENTIAL:
            if level <= 0:
                raise ValueError(
                    "Exponential normalisation requires a positive, non-zero level."
                )
        self._pre_edge_level = level

    @pre_edge_level.deleter
    def pre_edge_level(self):
        if self.pre_edge_norm_method is normConfigEdges.PREEDGE_NORM_TYPE.EXPONENTIAL:
            self._pre_edge_level = normConfigEdges.DEFAULT_PRE_EDGE_LEVEL_EXP
        # elif self.pre_edge_norm_method is normConfigEdges.EDGE_NORM_TYPE.LINEAR:
        # self._pre_edge_level = normConfigEdges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
        else:
            self._pre_edge_level = normConfigEdges.DEFAULT_PRE_EDGE_LEVEL_LINEAR

    @property
    def post_edge_level(self) -> float:
        return self._post_edge_level

    @post_edge_level.setter
    def post_edge_level(self, level: float):
        self._post_edge_level = level

    @post_edge_level.deleter
    def post_edge_level(self):
        self._post_edge_level = normConfigEdges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
