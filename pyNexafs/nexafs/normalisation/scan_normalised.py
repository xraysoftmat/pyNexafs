from __future__ import annotations  # For type hinting within class definitions
import numpy.typing as npt
import numpy as np
import abc
import warnings
import overrides
import datetime
from scipy import optimize as sopt
from types import NoneType
from typing import Self, overload, Literal
from pyNexafs.nexafs.scan import scanBase, scanAbstract
from pyNexafs.nexafs.normalisation.norm_settings import (
    normMethod,
    configBase,
    configChannel,
    normConfigEdges,
)


class scanAbstractNorm(scanAbstract, metaclass=abc.ABCMeta):
    """
    Abstract class to define the common methods used in all normalized scans.

    Parameters
    ----------
    scan : scan_abstract
        The original scan object to apply the normalisation to.
    apply_to : list[str] | list[int] | None, optional
        The list of y-channel labels to apply the normalisation to.
        Can also be a list of integers, representing the index of the y-channel, which
        will be converted to a list of strings to reduce ambiguity.
        By default None (applies to all channels).
    """

    def __init__(
        self, scan: scanAbstract, apply_to: list[int] | list[str] | None = None
    ) -> None:
        self._origin = scan
        """The original scan object."""
        self._config = None
        """The normalisation configuration object, used to save settings."""
        self._apply_to: list[str] | None
        """The list of y-channel labels to apply the normalisation to."""

        # Set the apply_to attribute
        if apply_to is not None:
            if all(isinstance(i, int) for i in apply_to):
                # Convert indexes to labels
                labels: list[str] = scan.y_labels
                self._apply_to = [labels[i] for i in apply_to]  # type: ignore - all i are int.
                return
            elif all(isinstance(i, str) for i in apply_to):
                # Create a copy
                self._apply_to = apply_to.copy()  # type: ignore - all i are str.
                return
            else:
                raise ValueError(
                    "`apply_to` must be a list of integers or a list of strings."
                )
        else:
            # Assign the apply_to attribute
            self._apply_to = None

    @overrides.overrides
    @abc.abstractmethod
    def copy(self) -> scanAbstractNorm:
        """
        Copy the scan object.

        Returns
        -------
        scan_abstract
            A copy of the scan object.
        """
        copy_obj = type(self)(scan=self._origin, apply_to=self._apply_to)
        return copy_obj

    @property
    def origin(self) -> scanAbstract:
        """
        Property for the original scan object.

        Returns
        -------
        scan_base
            The original scan object.
        """
        return self._origin

    @property
    def apply_to(self) -> list[str] | None:
        """
        The list of y-channel labels to apply the normalisation to.

        Parameters
        ----------
        apply_to : list[str] | None
            The new list of y-channel labels to apply the normalisation to.
            If None, applies to all channels.

        Returns
        -------
        list[str] | None
            The list of y-channel labels to apply the normalisation to.
            If None, applies to all channels.

        See Also
        --------
        scan_normalised.apply_to_indexes
            Method to return the indexes of the y-channels to apply the normalisation to.
        """
        return self._apply_to

    @apply_to.setter
    def apply_to(self, apply_to: list[str] | None) -> None:
        self._apply_to = apply_to

    @overload
    def apply_to_indexes(
        self, with_labels: Literal[True]
    ) -> tuple[list[int], list[str]]: ...

    @overload
    def apply_to_indexes(self, with_labels: Literal[False]) -> list[int]: ...

    @overload
    def apply_to_indexes(self) -> list[int]: ...

    def apply_to_indexes(
        self, with_labels: bool = False
    ) -> list[int] | tuple[list[int], list[str]]:
        """
        The list of y-channel indexes to apply the normalisation to.

        Any labels not found in the y_labels are ignored, and a warning is raised.

        Parameters
        ----------
        with_labels : bool, optional
            If True, returns a tuple of a list of indexes and a list of labels, by default False.

        Returns
        -------
        list[int] | tuple[list[int], list[str]]
            If `with_labels` is False, returns a list of indexes.
            If `with_labels` is True, returns a tuple of indexes and labels.
            If no `apply_to` is defined, returns a list of all indexes.
        """
        at = self._apply_to
        if at is not None:
            indexes: list[int] = []
            labels: list[str] = []
            ylabels = self.y_labels
            for label in at:
                try:
                    indexes.append(ylabels.index(label))
                    labels.append(label)
                except ValueError:
                    warnings.warn(f"Label {label} not found in y_labels. ")
            if with_labels:
                return indexes, labels
            return indexes
        else:
            return list(range(len(self.y_labels)))

    @overrides.overrides
    def reload_labels_from_parser(self) -> None:
        return self._origin.reload_labels_from_parser()

    @property
    def ctime(self) -> datetime.datetime:
        """
        The creation time of the file as a datetime object.

        Returns
        -------
        datetime.datetime
            The creation time of the linked origin scan object.
        """
        return self._origin.ctime

    @property
    def mtime(self) -> datetime.datetime:
        """
        The modification time of the file as a datetime object.

        Returns
        -------
        datetime.datetime
            The modification time of the linked origin scan object.
        """
        return self._origin.mtime

    @scanAbstract.filename.getter
    def filename(self) -> str:
        """
        Property for the filename of the scan object.

        Returns
        -------
        str
            The filename of the linked origin scan object.
        """
        return self._origin.filename

    @abc.abstractmethod
    def _apply_normalisation(self) -> None:
        """
        Abstract method to apply the normalisation to the y data.
        """
        pass

    def _load_from_origin(self) -> None:
        """
        Method to reload data from the origin scan object.

        Uses instantiated scan reference.
        """
        # Copy data
        self._x = self._origin.x.copy()
        self._x_errs = (
            self._origin.x_errs.copy() if self._origin.x_errs is not None else None
        )
        self._y = self._origin.y.copy()
        self._y_errs = (
            self._origin.y_errs.copy() if self._origin.y_errs is not None else None
        )

    def load_and_normalise(self) -> None:
        """
        (Re)Load data from the origin and apply normalisation.

        Calls the `self._load_from_origin` method, which copies data from an original `scan` object,
        and then calls the `self._apply_normalisation` method, required to be implemented in subclasses.
        """
        self._load_from_origin()
        self._apply_normalisation()
        return

    ## X property already covered by scan_abstract

    @scanAbstract.x_label.getter
    def x_label(self) -> str:
        """
        Property for the x-axis label, from the origin scan object.

        Returns
        -------
        str
            The x-axis label.
        """
        return self.origin.x_label

    @x_label.setter
    def x_label(self, label: str | None) -> None:
        """
        Property setter for the x-axis label, to the origin scan.

        Parameters
        ----------
        label : str
            The new x-axis label.
        """
        self.origin.x_label = label

    @scanAbstract.x_unit.getter
    def x_unit(self) -> str:
        """
        Property for the x-axis unit, from the origin scan object.

        Returns
        -------
        str
            The x-axis unit.
        """
        return self.origin.x_unit

    @x_unit.setter
    def x_unit(self, unit: str) -> None:
        """
        Property setter for the x-axis unit, to the origin scan.

        Parameters
        ----------
        unit : str
            The new x-axis unit.
        """
        self.origin.x_unit = unit

    @scanAbstract.y.setter
    def y(
        self, vals: npt.NDArray | list[list[int | float]] | list[int | float]
    ) -> None:
        """
        Property setter for the y data.

        Similar to @scan_abstract.y, but avoids resetting the `y_units`
        and `y_labels` attributes when setting the y data dimension, as
        labels should always be inherited from the original scan object.

        Parameters
        ----------
        vals : npt.NDArray
            The new y data.
        """
        # Set Y values
        if isinstance(vals, np.ndarray):
            if len(vals.shape) == 2:
                self._y = vals.copy()
            else:
                # Add additional axis to make y 2D instead of 1D
                self._y = vals.copy()[:, np.newaxis]
        elif isinstance(vals, list):
            if isinstance(vals[0], list):
                self._y = np.array(vals)
            else:
                # Assuming int/floats, add additional axis as well.
                self._y = np.array(vals)[:, np.newaxis]
        else:
            raise ValueError("Is not a list or a np.ndarray.")

        # Remove errs if number of yvals have changed.
        if self.y_errs is not None and self.y_errs.shape != self._y.shape:
            self.y_errs = None

        # Do not remove unit/labels if number of yvals have changed!

    @scanAbstract.y_labels.getter
    def y_labels(self) -> list[str]:
        """
        Property for the y-axis labels, from the origin scan object.

        Parameters
        ----------
        labels : list[str] | None
            The new y-axis labels, propagated to the origin scan object.

        Returns
        -------
        list[str]
            The y-axis labels.
        """
        return self.origin.y_labels

    @y_labels.setter
    def y_labels(self, labels: list[str] | None) -> None:
        self.origin.y_labels = labels

    @scanAbstract.y_units.getter
    def y_units(self) -> list[str]:
        """
        Property for the y-axis units, from the origin scan object.

        Parameters
        ----------
        units : list[str]
            The new y-axis units, propagated to the origin scan object.

        Returns
        -------
        list[str]
            The y-axis units.
        """
        return self.origin.y_units

    @y_units.setter
    def y_units(self, units: list[str] | None) -> None:
        self.origin.y_units = units

    @property
    @abc.abstractmethod
    def _config_class(self) -> type[configBase]:
        """
        The settings class for the normalisation configuration.

        Allows loading from a settings object.

        Returns
        -------
        type[configBase]
            The configuration class for the normalisation settings.
        """
        pass

    @property
    @abc.abstractmethod
    def settings(self) -> configBase:
        """
        The accumulated normalisation settings.

        Returns
        -------
        norm_config
            The normalisation settings.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def from_config(scan: scanAbstract, config: type[configBase]) -> "scanAbstractNorm":
        """
        Load the normalisation settings from a configuration object.

        Parameters
        ----------
        scan : scan_abstract
            The original scan object to apply the normalisation to.
        config : normConfig
            The normalisation configuration object.

        Returns
        -------
        scanAbstractNorm
            A new instance of a scanAbstractNorm subclass initialised with the provided configuration.
        """
        classes = scanAbstractNorm.__subclasses__()
        for cls in classes:
            if config.__class__ == cls._config_class:
                if cls.from_config != scanAbstractNorm.from_config:
                    new_obj = cls.from_config(scan, config)
                    return new_obj
                else:
                    raise NotImplementedError(
                        f"Class {cls} has not implemented the `from_config` method."
                    )
        raise TypeError(
            f"No matching class found for the configuration class {config.__class__}."
        )


class scanNorm(scanAbstractNorm):
    """
    General class for a normalisation of a scan_base.

    Normalisation is usually performed in reference to some flux measurement,
    such as the current of a mesh or photodiode. This class requires
    `norm_channel` or `norm_data` to be defined, but not both.

    Parameters
    ----------
    scan : Type[scan_base]
        The initial scan object to collect and normalise y dataseries.
    norm_channel : str | int | None, optional
        The y-channel name corresponding to normalisation data.
        Can provide a string name or an integer index, but will be stored as a string.
    norm_method : normMethod, optional
        The normalisation method to apply to the normalisation channel. See `configChannel.normMethod` for options.
    apply_to : list[int] | list[str] | None, optional
        The list of y-channel indexes to apply the normalisation to.

    Raises
    ------
    ValueError
        If both `norm_channel` and `norm_data` are simultaneously `None` or not-`None`. Only one can be defined.
    ValueError
        If `norm_channel` is not found in y_labels.

    See Also
    --------
    norm_settings.configChannel.normMethod
        The normalisation method enumerate used for the normalisation channel.
    """

    normMethod = configChannel.normMethod

    @overrides.overrides
    def __init__(
        self,
        scan: scanBase,
        norm_channel: str | int,
        norm_method: normMethod = configChannel.normMethod.FLUX,
        apply_to: list[int] | list[str] | None = None,
    ) -> None:
        # Set reference for original scan object
        super().__init__(scan, apply_to)
        print(self._apply_to)

        self._norm_idx: int
        """The index of the normalisation channel in the origin scan"""

        # Store initial information.
        if isinstance(norm_channel, str):
            self._norm_idx = self._origin.y_labels.index(norm_channel)
        else:
            self._norm_idx = norm_channel
            norm_channel = self._origin.y_labels[norm_channel]

        # Save the normalisation channel settings as an object.
        self._config = configChannel(norm_method, norm_channel)
        """The normalisation configuration object used for serialising settings."""

        # Process normalisation data
        self.load_and_normalise()

    @scanAbstractNorm._config_class.getter
    @overrides.overrides
    def _config_class(self) -> type[configBase]:
        """
        Return the settings class for the normalisation configuration.

        Required to allows loading from a settings object.

        Returns
        -------
        type[configChannel]
            The configuration class for the normalisation settings.
        """
        return configChannel

    @scanAbstractNorm.settings.getter
    @overrides.overrides
    def settings(self) -> configChannel:
        """
        A settings object capturing the settings for the normalisation channel.

        Returns
        -------
        configChannel
            The normalisation settings object, containing the channel and method.
        """
        return self._config

    @overrides.overrides
    @staticmethod
    def from_config(scan: scanAbstract, config: configChannel) -> scanNorm:  # type: ignore # TODO: implement fix for subtype override.
        """
        Load the normalisation settings from a configuration object.

        Parameters
        ----------
        scan : scan_abstract
            The original scan object to apply the normalisation to.
        config : configChannel
            The normalisation configuration object.

        Returns
        -------
        scanNorm
            A new instance of scanNorm initialised with the provided configuration.
        """
        return scanNorm(
            scan=scan, norm_channel=config.channel, norm_method=config.method
        )

    @property
    def method(self) -> normMethod:
        """
        The normalisation method used for the normalisation channel.

        Parameters
        ----------
        method : normMethod
            The new normalisation method enumerate. Can be
            - 0: `normMethod.NONE` for no normalisation.
            - 1: `normMethod.BACKGROUND` for background subtraction.
            - 2: `normMethod.FLUX` for normalisation by amplitude variation.

        Returns
        -------
        normMethod
            The normalisation method.
        """
        return self._config.method

    @method.setter
    def method(self, method: normMethod) -> None:
        self._config.method = method

    @property
    def channel(self) -> str | None:
        """
        The normalisation channel name.

        Parameters
        ----------
        channel : str
            The new normalisation channel name.

        Returns
        -------
        str
            The normalisation channel name.
        """
        return self._config.channel

    @channel.setter
    def channel(self, channel: str) -> None:
        self._config.channel = channel

    @overrides.overrides
    def _load_from_origin(self) -> None:
        """
        Re-load from source, refreshing data, errors, labels and units for x,y variables.

        Overrides `scan_abstract_norm._load_from_origin` depending on the `_norm_channel` attribute being None.
        """
        # Y Reloading
        if self.channel is not None:
            # Collect index of normalisation data
            self._norm_idx = ind = self._origin.y_labels.index(self.channel)
            # Collect normalisaton data:
            self._norm_data = self._origin.y[:, ind]
            self._norm_data_errs = (
                self._origin.y_errs[:, ind] if self._origin.y_errs is not None else None
            )

            # Collect Y data, removing index of normalisation channel.
            self._y = np.delete(
                self._origin.y, ind, 1
            )  # copy data, removing the index from the existing set of data.
            self._y_errs = (
                np.delete(self._origin.y_errs, ind, 1)
                if self._origin.y_errs is not None
                else None
            )

            # Collect X data normally.
            self._x = self._origin.x.copy()
            self._x_errs = (
                self._origin.x_errs.copy() if self._origin.x_errs is not None else None
            )
        else:
            # Load x and y data regularly.
            super()._load_from_origin()

    @overrides.overrides
    def _apply_normalisation(self) -> None:
        """
        Perform the selected normalisation.

        `self.method` determines the method to apply to the provided y normalisation data.
        """
        match self.method:
            case self.normMethod.NONE:
                # Do nothing for no normalisation
                pass
            case self.normMethod.BACKGROUND:
                # Subtract the normalisation data from the y data.
                self._normalisation_subtract()
            case self.normMethod.FLUX:
                # Scale the y data by the normalisation data.
                self._normalisation_scale()
            case _:
                raise ValueError("Normalisation method not defined.")
        return

    def _normalisation_scale(self) -> None:
        """
        Scale the y data by the normalisation data.

        Applies the amplitude normalisation to `apply_to` channels, otherwise to all channels.
        """
        # Scale the y data, only in the apply_to channels.
        for i, label in enumerate(self.y_labels):
            if self.apply_to is None or label in self.apply_to:
                self._y[:, i] /= self._norm_data

    def _normalisation_subtract(self) -> None:
        """
        Subtract the background from the selected y data.

        Applies the background subtraction to `apply_to` channels, otherwise to all channels.
        """
        # Subtract the normalisation data from the y data.
        for i, label in enumerate(self.y_labels):
            if self.apply_to is None or label in self.apply_to:
                self._y[:, i] -= self._norm_data

    @scanAbstractNorm.y_labels.getter
    def y_labels(self) -> list[str]:
        """
        Property for the y-axis labels, from the origin scan object.

        When the normalisation channel is defined, the labels are adjusted to remove the normalisation channel.

        Parameters
        ----------
        labels : list[str] | None
            The new y-axis labels.

        Returns
        -------
        list[str]
            The y-axis labels.
        """
        origin_labels = self._origin.y_labels
        idx = self._norm_idx
        if idx is not None:
            return origin_labels[0:idx] + origin_labels[idx + 1 :]
        else:
            return origin_labels

    @y_labels.setter
    def y_labels(self, labels: list[str] | None) -> None:
        idx = self._norm_idx
        if idx is not None and labels is not None and self.channel is not None:
            # Include the normalisation channel in the labels when setting.
            self._origin.y_labels = labels[0:idx] + [self.channel] + labels[idx:]
        else:
            # Otherwise set labels normally.
            self._origin.y_labels = labels

    @scanAbstractNorm.y_units.getter
    def y_units(self) -> list[str] | None:
        """
        Property for the y-axis units, from the origin scan object.

        When the normalisation channel is defined, the units are adjusted to remove the normalisation channel.

        Parameters
        ----------
        labels : list[str] | None
            The new y-axis units.

        Returns
        -------
        list[str] | None
            The y-axis units.
        """
        origin_units = self._origin.y_units
        idx = self._norm_idx
        if idx is not None and origin_units is not None:
            return origin_units[0:idx] + origin_units[idx + 1 :]
        else:
            return origin_units

    @y_units.setter
    def y_units(self, units: list[str] | None) -> None:
        idx = self._norm_idx
        if idx is not None:
            if units is not None:
                # Include the normalisation channel in the units when setting.
                self._origin.y_units = units[0:idx] + [self.channel] + units[idx:]
                return
        # Otherwise, set units normally.
        self._origin.y_units = units
        return

    def copy(self) -> Self:
        """
        A copy of the scan object.

        Returns
        -------
        scan_abstract
            A copy of the scan object.
        """
        return scanNorm(
            scan=self._origin,
            norm_channel=self.channel,
            norm_method=self.method,
            apply_to=self._apply_to,
        )


class scanNormExt(scanAbstractNorm):
    """
    A normalised scan normalised by a channel external to the object.
    """

    def __init__(self):
        pass

    @property
    def norm_data(self) -> npt.NDArray:
        """
        Property for normalisation data for the normalised scan.

        Can also be used to define errors. Setting values creates a clone.
        Upon defining new normalisation data, self.load_and_normalise are called.

        Parameters
        ----------
        values : npt.NDArray | tuple[npt.NDArray, npt.NDArray]
            A set of data values setting normalisation data and errors.
            If NDArray is 1D, assumes error values are None.
            To define error values, use tuple (data, errors) or 2D array.

        Returns
        -------
        npt.NDArray
            A 1D array of normalisation values.

        See Also
        --------
        scan_normalised.norm_data_errors
            Property for getting/setting normalisation errors.
        """
        return self._norm_data

    @norm_data.setter
    def norm_data(
        self, values: tuple[npt.NDArray, npt.NDArray | None] | npt.NDArray
    ) -> None:
        if isinstance(values, tuple):
            if (
                len(values) == 2
                and isinstance(values[0], np.ndarray)
                and isinstance(values[1], (np.ndarray, NoneType))
            ):
                norm_data, norm_data_errs = values
                if norm_data_errs is not None:
                    if norm_data.shape == norm_data_errs.shape:
                        self._norm_data = norm_data.copy()
                        self._norm_data_errs = norm_data_errs.copy()
                    else:
                        raise ValueError(
                            f"Shape of normalisation data {norm_data.shape} and errors {norm_data_errs.shape} to not match."
                        )
                else:
                    self._norm_data = norm_data.copy()
                    self._norm_data_errs = None
            else:
                raise ValueError(
                    "Tuple does not match required properties; two objects of numpy arrays, the latter allowed to be None."
                )
        elif isinstance(values, np.ndarray):
            self._norm_data = values.copy()
        else:
            raise TypeError(
                "`values` parameter is not a single numpy array or a tuple of 2 numpy arrays."
            )
        self.load_and_normalise()

    @property
    def norm_data_errs(self) -> npt.NDArray:
        """
        Property for the normalisation error data of the normalised scan.

        Does not have a setter, but can be set together with
        normalisation data using the `norm_data` property.

        Returns
        -------
        npt.NDArray
            A 1D array of errors.

        See Also
        --------
        scan_normalised.norm_data
            Property for getting/setting normalisation data and errors.
        """
        return self._norm_data_errs

    def _apply_normalisation(self) -> None:
        """
        Normalise the y data (and y_err data if present).

        If errors provided for normalisation data, incorporates into calculating new y_err data,
        using a sum of squares as follows:
            x = data,
            z = normalisation data
            y = f(x, z) = x/z,
            unc_y = u(y) = sqrt( (df/dx * unc(x)) ^2 + (df/dz * unc(z) )^2)
            df/dx = 1/z
            df/dz = -x/z**2
        """

        # Scale normalisation data point #0 to 1, so it affects from unity.
        scaled_norm = self._norm_data / self._norm_data[0]
        scaled_norm_errs = (
            self._norm_data_errs / self._norm_data[0]
            if self._norm_data_errs is not None
            else None
        )
        # Scale y, yerr data by normalisation.
        if len(self._y.shape) == 1:
            self._y /= scaled_norm
        else:
            self._y /= scaled_norm[:, np.newaxis]

        # Scale y_errs if defined
        if self._y_errs is not None:
            if self._norm_data_errs is not None:
                # Two error sources, use sum of square errors if errors is defined:
                square_y_errs = np.square(1 / scaled_norm * self._y_errs)
                square_norm_errs = np.square(
                    self._y / scaled_norm**2 * scaled_norm_errs
                )
                self._y_errs = np.sqrt(square_y_errs + square_norm_errs)
            else:
                # No norm data errors
                self._y_errs /= scaled_norm
        elif self._norm_data_errs is not None:
            # Create y_errs from norm_data_errors.
            self._y_errs = self.y / scaled_norm**2 * scaled_norm_errs
        # pass for no definition of errors.
        return


class scanNormBackgroundChannel(scanNorm):
    """
    Normalising a scan_base to a particular channel of a background scan.

    Useful for NEXAFS such as carbon K edges. If a mesh current is used to normalise the signal,
    it will not necessarily normalise to the true absorption signal. A downstream photodiode, or alternative
    intensity channel, can be used to capture a true background signal. This channel can then
    permit double normalisation (i.e., with the mesh current signal) on the measured spectra.

    Parameters
    ----------
    scan : scanAbstract
        The initial scan object to collect and normalise.
    background_scan : scanAbstract
        The background scan object to use for normalisation.
    norm_channel : str
        The y-channel name corresponding to normalisation data in the background scan.
        This channel will be used to normalise the y data of the scan object.
    """

    @overrides.overrides
    def __init__(
        self,
        scan: scanAbstract,
        background_scan: scanAbstract,
        norm_channel: str,
    ) -> None:
        # Save background scan and channel information
        self._background_scan = background_scan
        self._background_channel = norm_channel
        # Collect normalisation data
        norm_channel_index = background_scan.y_labels.index(norm_channel)
        norm_data = background_scan.y[:, norm_channel_index]
        norm_data_errs = (
            background_scan.y_errs[:, norm_channel_index]
            if background_scan.y_errs is not None
            else None
        )
        super().__init__(scan, norm_data=norm_data, norm_data_errs=norm_data_errs)

    @overrides.overrides
    def _load_from_origin(self):
        # Reload background_scan data
        if hasattr(self._background_scan, "_load_from_origin"):
            self._background_scan.load_from_origin()
        # Reload scan data
        super()._load_from_origin()


# Infer the appropriate setting options for inputs
CHANNEL_METHOD = normMethod
PREEDGE_NORM_TYPE = normConfigEdges.PREEDGE_NORM_TYPE
POSTEDGE_NORM_TYPE = normConfigEdges.POSTEDGE_NORM_TYPE


class scanNormEdges(scanAbstractNorm):
    """
    Normalising a scan_base across pre &/ post edges.

    Uses two band definitions to normalise the y data centred around a chemical resonant edge
    such as the K-edge of carbon.

    Parameters
    ----------
    scan : Type[scan_base] | scan_base
        The initial scan object to collect and normalise y dataseries.
    pre_edge_domain : _type_, optional
        Data to define the domain of pre-edge normalisation. Can take the following data forms:
        - A list of indices (integers) to subset the y data.
        - A tuple of floats to define the inclusive-domain of the x data.
        - None, to not perform pre-edge normalisation. Overrides pre_edge_normalisation enumerate.
    post_edge_domain : _type_, optional
        Data to define the domain of post-edge normalisation. Same format as pre-edge.
        If None overrides post_edge_normalisation enumerate.
    pre_edge_norm_method : EDGE_NORM_TYPE, optional
        Normalisation type for pre-edge, by default EDGE_NORM_TYPE.LINEAR.
    post_edge_norm_method : EDGE_NORM_TYPE, optional
        Normalisation type for post-edge, by default EDGE_NORM_TYPE.LINEAR.
    pre_edge_level : float, optional
        Normalisation level for pre-edge, by default 0.1.
    post_edge_level : float, optional
        Normalisation level for post-edge, by default 1.0.
    apply_to : list[str] | None, optional
        The list of y-channel labels to apply the normalisation to.

    Attributes
    ----------
    EDGE_NORM_TYPE : enumerate
        Enumerate types for edge normalisation.

    Raises
    ------
    ValueError
        If both `norm_channel` and `norm_data` are simultaneously None or not-None. Only one can be defined.
    ValueError
        If `norm_channel` is not found in y_labels.
    """

    LIN_FN = lambda x, m, c: m * x + c
    """
    A function to define a linear fit.

    Parameters
    ----------
    x : ArrayLike
        The x value.
    m : float
        The gradient of the line.
    c : float
        The y-intercept of the line.
    """

    EXP_FN_OFFSET = lambda x, a, b, c: a * np.exp(b * (x - x.min())) + c
    """
    A function to define an exponential fit, adjusted to the minimum x value.

    Parameters
    ----------
    x : ArrayLike
        The x value.
    a : float
        The amplitude of the exponential.
    b : float
        The decay constant of the exponential.
    c : float
        The offset of the exponential.
    """

    PREEDGE_NORM_TYPE = normConfigEdges.PREEDGE_NORM_TYPE
    POSTEDGE_NORM_TYPE = normConfigEdges.POSTEDGE_NORM_TYPE

    def __init__(
        self,
        scan: scanAbstract,
        pre_edge_domain: tuple[float, float] | list[int] | None,
        post_edge_domain: tuple[float, float] | list[int] | None,
        pre_edge_norm_method: PREEDGE_NORM_TYPE = PREEDGE_NORM_TYPE.CONSTANT,
        post_edge_norm_method: POSTEDGE_NORM_TYPE = POSTEDGE_NORM_TYPE.CONSTANT,
        pre_edge_level: float | None = None,
        post_edge_level: float | None = None,
        apply_to: list[str] | None = None,
    ) -> None:
        # Initialize the scan_abstract_norm class.
        super().__init__(scan=scan, apply_to=apply_to)
        # Check level defaults
        if pre_edge_level is None:
            pre_edge_level = normConfigEdges.DEFAULT_PRE_EDGE_LEVEL_CONSTANT
        if post_edge_level is None:
            post_edge_level = normConfigEdges.DEFAULT_POST_EDGE_LEVEL_CONSTANT
        # Create a settings object to store the normalisation settings.
        self._config = normConfigEdges(
            pre_edge_domain=pre_edge_domain,
            post_edge_domain=post_edge_domain,
            pre_edge_norm_method=pre_edge_norm_method,
            post_edge_norm_method=post_edge_norm_method,
            pre_edge_level=pre_edge_level,
            post_edge_level=post_edge_level,
            apply_to=apply_to,
        )

        # Define variables for edge fitting
        self.pre_edge_fit_params: list | None = None
        self.post_edge_fit_params: list | None = None

        # Perform normalisation
        self.load_and_normalise()
        return

    @scanAbstractNorm._config_class.getter
    @overrides.overrides
    def _config_class(self) -> type[configBase]:
        """
        The linked configuration class for the normalisation settings.

        Returns
        -------
        type[normConfigEdges]
            The configuration class for the normalisation settings.
        """
        return normConfigEdges

    @scanAbstractNorm.settings.getter
    @overrides.overrides
    def settings(self) -> normConfigEdges:
        """
        A settings object associated with the settings for a normalisation channel.

        Returns
        -------
        normConfigEdges
            The normalisation settings.
        """
        return self._config

    @overrides.overrides
    @staticmethod
    def from_config(scan: scanAbstract, config: normConfigEdges) -> scanNormEdges:  # type: ignore # TODO: implement fix for subtype override.
        """
        Load the normalisation settings from a configuration object.

        Parameters
        ----------
        scan : scan_abstract
            The original scan object to apply the normalisation to.
        config : normConfigEdges
            The normalisation configuration object.

        Returns
        -------
        scanNormEdges
            A new instance of scanNormEdges with the provided configuration.
        """
        return scanNormEdges(
            scan=scan,
            pre_edge_domain=config.pre_edge_domain,
            post_edge_domain=config.post_edge_domain,
            pre_edge_norm_method=config.pre_edge_norm_method,
            post_edge_norm_method=config.post_edge_norm_method,
            pre_edge_level=config.pre_edge_level,
            post_edge_level=config.post_edge_level,
            apply_to=config.apply_to,
        )

    @overrides.overrides
    def copy(self) -> scanNormEdges:
        clone = scanNormEdges(
            self._origin,
            pre_edge_domain=self._config._pre_edge_domain,
            post_edge_domain=self._config._post_edge_domain,
            pre_edge_norm_method=self._config._pre_edge_norm_method,
            post_edge_norm_method=self._config._post_edge_norm_method,
            pre_edge_level=self._config._pre_edge_level,
            post_edge_level=self._config._post_edge_level,
            apply_to=self._config._apply_to,
        )
        return clone

    @property
    def DEFAULT_PRE_EDGE_LEVEL_CONSTANT(self) -> float:
        return self._config.DEFAULT_PRE_EDGE_LEVEL_CONSTANT

    @property
    def DEFAULT_PRE_EDGE_LEVEL_LINEAR(self) -> float:
        return self._config.DEFAULT_PRE_EDGE_LEVEL_LINEAR

    @property
    def DEFAULT_PRE_EDGE_LEVEL_EXP(self) -> float:
        return self._config.DEFAULT_PRE_EDGE_LEVEL_EXP

    @property
    def DEFAULT_POST_EDGE_LEVEL_CONSTANT(self) -> float:
        return self._config.DEFAULT_POST_EDGE_LEVEL_CONSTANT

    @overrides.overrides
    def _apply_normalisation(self) -> None:
        """
        Scale y data by the normalisation regions.

        Applies the pre-edge and post-edge normalisation methods to the y data.
        """
        # Define fitting functions
        lin_fn = self.LIN_FN
        exp_fn = self.EXP_FN_OFFSET
        # Perform dual normalisation, sequentially to avoid duplicate code.

        ## PRE-EDGE
        if (
            self.pre_edge_norm_method is not normConfigEdges.PREEDGE_NORM_TYPE.NONE
            and self.pre_edge_domain is not None
        ):
            # Collect pre-edge indexes
            if isinstance(self.pre_edge_domain, list):
                # Use the list of indexes
                pre_inds = self.pre_edge_domain
                if len(pre_inds) == 0:
                    raise ValueError("Pre-edge index list is empty.")
            elif isinstance(self.pre_edge_domain, tuple):
                # Use the domain range
                pre_inds = np.asarray(
                    (self.x >= self.pre_edge_domain[0])
                    & (self.x <= self.pre_edge_domain[1])
                ).nonzero()
                # Check indexes of each tuple element
                if len(pre_inds[0]) == 0:
                    raise ValueError(
                        f"Pre-edge domain ({self.pre_edge_domain[0]} to {self.pre_edge_domain[1]}) contains no datapoints."
                    )
            else:
                raise AttributeError(
                    "Pre-edge domain is not defined correctly. Should be a list of indexes or the range in a tuple."
                )

            # Calculate pre-edge on all y values and normalise
            match self.pre_edge_norm_method:
                case normConfigEdges.PREEDGE_NORM_TYPE.CONSTANT:
                    if self.apply_to:
                        inds = self.apply_to_indexes()
                    else:
                        mean = np.mean(self.y[pre_inds], axis=0)
                        stdev = np.std(self.y[pre_inds], axis=0)

                        self.y[:, :] += -mean[np.newaxis, :] + self.pre_edge_level
                        self.pre_edge_fit_params = mean.tolist()

                    # if self.y_errs is not None:
                    #     # Add the standard deviation
                    #     self.y_errs = np.sqrt(np.square(self.y_errs) + stdev ** 2)

                case normConfigEdges.PREEDGE_NORM_TYPE.LINEAR:
                    params: list = []
                    for i in range(self.y.shape[1]):
                        popt, pcov = sopt.curve_fit(
                            lin_fn, self.x[pre_inds], self.y[:, i][pre_inds]
                        )
                        self.pre_edge_fit_params = popt.tolist()
                        self.y[:, i] += -lin_fn(self.x, *popt) + self.pre_edge_level
                        params += popt.tolist()

                    self.pre_edge_fit_params = params

                case normConfigEdges.PREEDGE_NORM_TYPE.EXPONENTIAL:
                    params: list = []
                    for i in range(self.y.shape[1]):
                        # Subtract offset to fit
                        xfit = self.x[pre_inds] - self.x[pre_inds].min()
                        x = self.x - self.x[pre_inds].min()
                        a0 = self.y[:, i][pre_inds].max() - self.y[:, i][pre_inds].min()
                        # Fit
                        popt, pcov = sopt.curve_fit(
                            exp_fn,
                            xfit,
                            self.y[:, i][pre_inds],
                            p0=(a0, -0.5, 0),
                            bounds=[(-np.inf, -np.inf, -np.inf), (np.inf, 0, np.inf)],
                            maxfev=10000,
                        )
                        self.y[:, i] += -exp_fn(x, *popt) + self.pre_edge_level
                        params += popt.tolist()

                    self.pre_edge_fit_params = params

                case normConfigEdges.PREEDGE_NORM_TYPE.NONE:
                    # Do nothing for no pre-edge normalisation
                    pass
                case _:
                    # Should never reach here, and dual normalisation excludes None type.
                    raise ValueError("Pre-edge normalisation type not defined.")

        ## POST-EDGE
        if (
            self.post_edge_norm_method is not normConfigEdges.POSTEDGE_NORM_TYPE.NONE
            and self.post_edge_domain is not None
        ):
            # Collect post-edge indexes:
            if isinstance(self.post_edge_domain, list):
                post_inds = self.post_edge_domain
                if len(post_inds) == 0:
                    raise ValueError("Post-edge index list is empty.")
            elif isinstance(self.post_edge_domain, tuple):
                post_inds = np.where(
                    (self.x >= self.post_edge_domain[0])
                    & (self.x <= self.post_edge_domain[1])
                )
                # Check indexes of each tuple element
                if len(post_inds[0]) == 0:
                    raise ValueError(
                        f"Post-edge domain ({self.post_edge_domain[0]} to {self.post_edge_domain[1]}) contains no datapoints."
                    )
            else:
                raise AttributeError(
                    "Post-edge domain is not defined correctly. Should be a list of indexes or the range in a tuple."
                )

            if len(post_inds) == 0:
                raise ValueError("Post-edge domain is empty.")

            # Calculate post-edge and normalise. Note, y data is already normalised by pre-edge.
            match self.post_edge_norm_method:
                case normConfigEdges.POSTEDGE_NORM_TYPE.CONSTANT:
                    # Get the mean of the post-edge from the pre-edge level.
                    postave = np.mean(self.y[post_inds] - self.pre_edge_level, axis=0)
                    # Scale to difference from pre-edge level to post-edge level. Ignore zero values.
                    scale = np.zeros(postave.shape)
                    scale[postave != 0] = (
                        self.post_edge_level - self.pre_edge_level
                    ) / postave[postave != 0]
                    self.y = (
                        self.y - self.pre_edge_level
                    ) * scale + self.pre_edge_level
                    pass
                case normConfigEdges.POSTEDGE_NORM_TYPE.NONE:
                    # Do nothing
                    pass
                case _:
                    # Should never reach here, and dual normalisation excludes None type.
                    raise ValueError("Post-edge normalisation type not defined.")

        return

    @property
    def pre_edge_domain(self) -> tuple[float, float] | list[int] | None:
        """
        A property defining the pre-edge domain of normalisation.

        If setting and `pre_edge_normalisation` is `NONE`, will set new normalisation enumerate to `LINEAR`.

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
        return self._config._pre_edge_domain

    @pre_edge_domain.setter
    def pre_edge_domain(self, vals: list[int] | tuple[float, float] | None) -> None:
        self._config._pre_edge_domain = vals

    @pre_edge_domain.deleter
    def pre_edge_domain(self):
        del self._config._pre_edge_domain

    @property
    def pre_edge_norm_method(self) -> normConfigEdges.PREEDGE_NORM_TYPE:
        """
        Property to define the type of normalisation performed on the pre-edge.

        Parameters
        ----------
        vals : normConfigEdges.EDGE_NORM_TYPE
            LINEAR, EXPONENTIAL or NONE.

        Returns
        -------
        normConfigEdges.EDGE_NORM_TYPE
            The current normalisation type.
        """
        return self._config.pre_edge_norm_method

    @pre_edge_norm_method.setter
    def pre_edge_norm_method(self, vals: normConfigEdges.PREEDGE_NORM_TYPE) -> None:
        self._config.pre_edge_norm_method = vals

    @pre_edge_norm_method.deleter
    def pre_edge_norm_method(self) -> None:
        del self._config.pre_edge_norm_method

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
        return self._config.pre_edge_level

    @pre_edge_level.setter
    def pre_edge_level(self, level: float) -> None:
        self._config.pre_edge_level = level

    @pre_edge_level.deleter
    def pre_edge_level(self) -> None:
        del self._config.pre_edge_level

    @property
    def post_edge_domain(self) -> list[int] | tuple[float, float] | None:
        """
        A property defining the post-edge domain of normalisation.

        If setting and `post_edge_normalisation` is `NONE`, will set new normalisation enumerate to `LINEAR`.

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
        return self._config._post_edge_domain

    @post_edge_domain.setter
    def post_edge_domain(self, vals: list[int] | tuple[float, float] | None) -> None:
        self._config.post_edge_domain = vals

    @post_edge_domain.deleter
    def post_edge_domain(self):
        del self._config.post_edge_domain

    @property
    def post_edge_norm_method(self) -> normConfigEdges.POSTEDGE_NORM_TYPE:
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
        return self._config._post_edge_norm_method

    @post_edge_norm_method.setter
    def post_edge_norm_method(self, method: normConfigEdges.POSTEDGE_NORM_TYPE) -> None:
        self._config.post_edge_norm_method = method

    @post_edge_norm_method.deleter
    def post_edge_norm_method(self) -> None:
        self._post_edge_normalisation = None

    @property
    def post_edge_level(self) -> float:
        """
        Property to define the normalisation level (constant) for the post-edge.

        Parameters
        ----------
        vals : float
            The normalisation average.

        Returns
        -------
        float
            The normalisation level.
        """
        return self._config.post_edge_level

    @post_edge_level.setter
    def post_edge_level(self, vals: float) -> None:
        # if (
        #     self.post_edge_normalisation
        #     is normConfigEdges.POSTEDGE_NORM_TYPE.EXPONENTIAL
        # ):
        #     if vals <= 0:
        #         raise ValueError(
        #             "Exponential normalisation requires a positive, non-zero level."
        #         )
        self._config.post_edge_level = vals

    @post_edge_level.deleter
    def post_edge_level(self) -> None:
        if (
            # self.post_edge_normalisation
            # is normConfigEdges.POSTEDGE_NORM_TYPE.EXPONENTIAL
            False
        ):
            self._post_edge_level = normConfigEdges.DEFAULT_PRE_EDGE_LEVEL_EXP
        # elif self.pre_edge_normalisation is normConfigEdges.EDGE_NORM_TYPE.LINEAR:
        # self._pre_edge_level = normConfigEdges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
        else:
            self._post_edge_level = normConfigEdges.DEFAULT_PRE_EDGE_LEVEL_LINEAR


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # Create a basic scan object form test data
    MEX2_MDA_PATH = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..\\..\\tests\\test_data\\au\\MEX2\\MEX2_5640.mda",
        )
    )
    SXR_MDA_PATH = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..\\..\\tests\\test_data\\au\\SXR\\sxr129598.mda",
        )
    )
    from pyNexafs.parsers.au import MEX2_NEXAFS, SXR_NEXAFS

    # MEX2 example
    parser_mex2 = MEX2_NEXAFS(
        MEX2_MDA_PATH, relabel=True, energy_bin_domain=(3300, 3700)
    )
    scan_mex2 = parser_mex2.to_scan()

    # SXR Oxygen example
    parser_sxr = SXR_NEXAFS(SXR_MDA_PATH, relabel=True)
    scan_sxr = parser_sxr.to_scan()
    assert scan_sxr.y_labels[2] == "I0 VF", scan_sxr.y_labels[2]

    examples = [
        {
            "scan": scan_mex2,
            "norm_channel": "I0",
            "pre_edge_domain": (2460, 2465),
            "post_edge_domain": (2520, 2530),
            "idx": 1,
        },
        {
            "scan": scan_sxr,
            "norm_channel": scan_sxr.y_labels[2],
            "pre_edge_domain": (515, 525),
            "post_edge_domain": (570, 580),
            "idx": 0,
        },
    ]

    # plt.subplots(1,2, figsize=(10,5))
    # plt.plot(scan_sxr.x, scan_sxr.y[:,2], label="I0")
    # plt.show()

    for ex in examples[1:]:
        scan = ex["scan"]
        norm_channel = ex["norm_channel"]
        pre_edge_domain = ex["pre_edge_domain"]
        post_edge_domain = ex["post_edge_domain"]
        idx = ex["idx"]

        # Normalise to a channel
        norm = scanNorm(scan, norm_channel=norm_channel)

        # Normalise to the edges
        edge_c = normConfigEdges(
            scan, pre_edge_domain=pre_edge_domain, post_edge_domain=post_edge_domain
        )
        edge_l = normConfigEdges(
            scan,
            pre_edge_domain=pre_edge_domain,
            post_edge_domain=post_edge_domain,
            pre_edge_normalisation=normConfigEdges.PREEDGE_NORM_TYPE.LINEAR,
        )

        edge_e = normConfigEdges(
            scan,
            pre_edge_domain=pre_edge_domain,
            post_edge_domain=post_edge_domain,
            pre_edge_normalisation=normConfigEdges.PREEDGE_NORM_TYPE.EXPONENTIAL,
        )

        # Plot each normalisation step
        fig, ax = plt.subplots(1, 3, sharex=True, figsize=(15, 5))
        ax[0].plot(
            scan.x,
            scan.y[:, idx],
            label=f"{type(scan_mex2).__name__}\n{scan.y_labels[idx]}",
        )

        l = ax[1].plot(
            norm.x, norm.y[:, idx], label=f"{type(norm).__name__}\n{norm.y_labels[idx]}"
        )
        ax[1].plot(
            scan.x,
            scan.y[:, idx],
            "--",
            label=f"{type(scan_mex2).__name__}\n{scan.y_labels[idx]}",
        )  # c=l[0].get_color()

        l = ax[2].plot(
            edge_c.x,
            edge_c.y[:, idx],
            label=f"{type(edge_c).__name__}\nConstant\n{edge_c.y_labels[idx]}",
        )
        l = ax[2].plot(
            edge_l.x,
            edge_l.y[:, idx],
            label=f"{type(edge_l).__name__}\nLinear\n{edge_l.y_labels[idx]}",
        )
        l = ax[2].plot(
            edge_e.x,
            edge_e.y[:, idx],
            label=f"{type(edge_e).__name__}\nExponential\n{edge_e.y_labels[idx]}",
        )
        ax[1].plot(
            edge_e.x,
            normConfigEdges.EXP_FN_OFFSET(
                edge_e.x, *edge_e.pre_edge_fit_params[3 * idx : 3 * (idx + 1)]
            ),
            "--",
            c=l[0].get_color(),
            label="Exponential Subtraction",
        )

        ax[2].set_xlabel("Energy (eV)")
        for a in ax:
            a.set_ylabel("Intensity (a.u.)")
            a.legend()

        ax[0].set_title("Original")
        ax[1].set_title("Normalised to I0")
        ax[2].set_title("Normalised to Edges")
        # ax[2].set_yscale("log")
        fig.tight_layout()
        plt.show()
